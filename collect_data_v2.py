"""
Balanced data collection for the 2-skill v2 pipeline.

Key differences from the original collect_data.py
===================================================

OLD (bugs / design flaws):
  - Collected 2,000,000 simulation steps total, split unevenly across skill combos.
    → skill combos with short rallies got far more data than long-rally combos.
  - Skills randomised mid-rally (side_target changed at every net crossing).
    → no stable (skill1, skill2) label per rally.
  - Did not store the 'done' flag or rally winner.
    → labeling could not distinguish won from truncated rallies.
  - Used the raw 116-dim obs as state.
    → the v2 pipeline uses a 76-dim encoded state via nash_skills/v2/state_encoder.py.
  - Counted truncated episodes (step-cap hit) toward target_rallies.
    → 80%+ of stored rallies had winner=0 → all-zero labels → Q/potential models flat.

NEW design:
  - Only count done=True episodes toward target_rallies; discard truncated episodes.
    Every stored rally has a real winner (1 or 2) with nonzero discounted returns.
  - Skills are fixed for the entire rally (no mid-rally randomisation).
  - Record winner (1=ego, 2=opp) from ball velocity at done step.
  - Store 76-dim encoded state (includes gantry, joint angles, ball, skill) in 'states'.
  - Store 116-dim raw obs in 'raw_obs' for detect_winner compatibility.
  - Print a balance summary after collection.

2-skill space: ['left', 'right']  →  4 combos × TARGET_RALLIES rallies (done only).

Output format (pickle list of dicts)
--------------------------------------
Each entry:
    {
        'skill1' : str,                  # ego skill name  ('left' or 'right')
        'skill2' : str,                  # opp skill name  ('left' or 'right')
        'states' : list[np.ndarray],     # 76-dim encoded ego state at each net crossing
        'raw_obs': list[np.ndarray],     # 116-dim raw obs (for detect_winner, uses obs[39])
        'winner' : int,                  # 1=ego, 2=opp  (never 0 — truncated discarded)
    }

Run:
    MUJOCO_GL=cgl venv/bin/python collect_data_v2.py
    MUJOCO_GL=cgl venv/bin/python collect_data_v2.py \
        --rallies 100 --output data/rallies_v2_2skill.pkl
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import itertools
import pickle as pkl

import numpy as np
from stable_baselines3 import PPO

from mujoco_env_comp import KukaTennisEnv
from nash_skills.v2.labeling import detect_winner, summarise_balance, check_balance
from nash_skills.v2.state_encoder import encode_ego

# --------------------------------------------------------------------------- #
# Defaults                                                                     #
# --------------------------------------------------------------------------- #
PPO_MODEL_PATH        = "logs/best_model_tracker1/best_model"
DEFAULT_OUTPUT        = "data/rallies_v2_2skill.pkl"
TARGET_RALLIES        = 100   # rallies per skill pair (4 pairs → 400 total)
MAX_STEPS_PER_EPISODE = 800   # step cap per episode
HISTORY               = 4

# 2-skill space — same convention as mujoco_env_comp.py side_target
SKILL_NAMES_2 = ["left", "right"]
SKILL_TARGETS = {"left": -1.0, "right": 1.0}
# --------------------------------------------------------------------------- #


def _build_ppo_obs(obs, info, player: int) -> np.ndarray:
    """Build the 68-dim PPO input for `player` (1=ego, 2=opp)."""
    H = HISTORY
    ppo_obs = np.zeros(9 + 9 + 7 + 7 + 9 * H, dtype=np.float32)
    if player == 1:
        ppo_obs[:9]    = obs[:9]
        ppo_obs[9:18]  = obs[18:27]
        ppo_obs[18:21] = info["diff_pos"]
        ppo_obs[21:25] = info["diff_quat"]
        ppo_obs[25:32] = info["target"]
        ppo_obs[32:]   = obs[42: 42 + H * 9]
    else:
        ppo_obs[:9]    = obs[9:18]
        ppo_obs[9:18]  = obs[27:36]
        ppo_obs[18:21] = info["diff_pos_opp"]
        ppo_obs[21:25] = info["diff_quat_opp"]
        ppo_obs[25:32] = info["target_opp"]
        ppo_obs[32:]   = obs[42 + H * 9: 42 + 2 * H * 9]
    return ppo_obs


def _set_skills(env: KukaTennisEnv, skill1: str, skill2: str) -> None:
    """Apply the skill targets to the environment."""
    env.side_target     = SKILL_TARGETS[skill1]
    env.side_target_opp = SKILL_TARGETS[skill2]


def collect(
    target_rallies: int = TARGET_RALLIES,
    output_path: str = DEFAULT_OUTPUT,
    ppo_path: str = PPO_MODEL_PATH,
) -> list:
    """
    Collect `target_rallies` complete rallies for each of the 4 skill pairs.

    Returns the full list of rally dicts (also saved to output_path).
    """
    env   = KukaTennisEnv(proc_id=1)
    model = PPO.load(ppo_path)

    all_rallies = []

    for skill1, skill2 in itertools.product(SKILL_NAMES_2, SKILL_NAMES_2):
        print(f"\n=== Collecting: ego={skill1}  opp={skill2} ===")
        _set_skills(env, skill1, skill2)
        obs, info = env.reset()

        completed        = 0
        steps_this_combo = 0
        prev_ball_x      = obs[36]

        # Per-episode state
        curr_states = []   # encoded (76-dim)
        curr_raw    = []   # raw obs (116-dim)
        steps_in_ep = 0

        while completed < target_rallies:
            # Build PPO obs for each player and act
            ppo1 = _build_ppo_obs(obs, info, player=1)
            ppo2 = _build_ppo_obs(obs, info, player=2)
            a1, _ = model.predict(ppo1, deterministic=True)
            a2, _ = model.predict(ppo2, deterministic=True)

            action = np.zeros(18, dtype=np.float32)
            action[:9]  = a1[:9]
            action[9:]  = a2[:9]

            obs, _reward, done, _, info = env.step(action)

            curr_ball_x  = obs[36]
            steps_in_ep  += 1
            steps_this_combo += 1

            # Record encoded 76-dim state + raw 116-dim obs at each net crossing.
            # encode_ego includes gantry pos, joint angles, ball state, and skill indices.
            # raw_obs is kept for detect_winner which reads ball_vel_x from obs[39].
            if (prev_ball_x - 1.5) * (curr_ball_x - 1.5) < 0:
                curr_states.append(encode_ego(obs, info))  # 76-dim encoded state
                curr_raw.append(obs.copy())                # 116-dim raw obs for detect_winner

            prev_ball_x = curr_ball_x

            # Episode ended with a real done signal → winner is known
            if done:
                winner = detect_winner(curr_raw, done=True)
                if len(curr_states) > 0:
                    all_rallies.append({
                        "skill1":  skill1,
                        "skill2":  skill2,
                        "states":  curr_states,
                        "raw_obs": curr_raw,
                        "winner":  winner,
                    })
                    completed += 1
                    if completed % 20 == 0:
                        print(f"  {completed}/{target_rallies} rallies  "
                              f"({steps_this_combo} steps so far)")

                # Reset for next episode — keep skills fixed
                curr_states = []
                curr_raw    = []
                steps_in_ep = 0
                _set_skills(env, skill1, skill2)
                obs, info = env.reset()
                prev_ball_x = obs[36]

            # Step cap hit without done → discard this episode silently and reset
            elif steps_in_ep >= MAX_STEPS_PER_EPISODE:
                curr_states = []
                curr_raw    = []
                steps_in_ep = 0
                _set_skills(env, skill1, skill2)
                obs, info = env.reset()
                prev_ball_x = obs[36]

        print(f"  Done: {completed} rallies collected "
              f"({steps_this_combo} total steps)")

    env.close()

    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pkl.dump(all_rallies, f)
    print(f"\nSaved {len(all_rallies)} rallies to {output_path}")

    # Balance report
    counts = summarise_balance(all_rallies)
    is_ok, ratio = check_balance(all_rallies, threshold=3.0)
    print(f"\nBalance check: max/min ratio = {ratio:.2f} "
          f"({'OK' if is_ok else 'IMBALANCED'})")
    print("Per-pair counts:")
    for (s1, s2), cnt in sorted(counts.items()):
        print(f"  {s1:8s} vs {s2:8s}: {cnt}")

    return all_rallies


# --------------------------------------------------------------------------- #
# CLI entry point                                                               #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect balanced 2-skill rally data for the v2 Nash pipeline."
    )
    parser.add_argument("--rallies", type=int, default=TARGET_RALLIES,
                        help=f"Target number of rallies per skill pair (default: {TARGET_RALLIES})")
    parser.add_argument("--output",  type=str, default=DEFAULT_OUTPUT,
                        help=f"Output pickle path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--ppo",     type=str, default=PPO_MODEL_PATH,
                        help="Path to PPO model checkpoint")
    args = parser.parse_args()

    collect(
        target_rallies=args.rallies,
        output_path=args.output,
        ppo_path=args.ppo,
    )
