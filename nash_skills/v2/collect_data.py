"""
Data collection for the v2 high-level Nash pipeline.

Key differences from the old collect_data_5skill.py
====================================================

OLD (bugs / design flaws):
  - Collected a fixed number of SIMULATION STEPS per skill pair, not rallies.
    → Short-rally pairs (e.g. right_short vs right_short) produced 78x more
      entries than long-rally pairs (center_safe vs center_safe).
  - Stored only the ball-crossing obs (one state per crossing).
    → Fine for the 68-dim PPO slice but lost joint-angle info needed for v2.
  - Did not store the episode `done` flag or rally winner.
    → Labeling could not distinguish won from truncated rallies.
  - Used the old 116-dim raw obs as the state.
    → The v2 pipeline uses a richer 76-dim encoded state via state_encoder.py.

NEW design:
  - Collect exactly TARGET_RALLIES_PER_COMBO complete rallies per skill pair.
    This guarantees a balanced dataset regardless of rally length.
  - Store both the encoded state (76-dim) and the raw obs (116-dim, for inspection).
  - Record winner (1/2/0) and the done flag at episode end.
  - Print a balance summary after collection.
  - Cap maximum rallies per episode to avoid degenerate infinite rallies.

Output format (pickle list of dicts)
--------------------------------------
Each entry:
    {
        'skill1' : str,                  # ego skill name
        'skill2' : str,                  # opp skill name
        'states' : list[np.ndarray],     # encoded states, shape (76,) each
        'raw_obs': list[np.ndarray],     # raw 116-dim obs (for debugging)
        'winner' : int,                  # 1=ego, 2=opp, 0=truncated
    }

Run:
    MUJOCO_GL=cgl venv/bin/python nash_skills/v2/collect_data.py
    MUJOCO_GL=cgl venv/bin/python nash_skills/v2/collect_data.py \
        --rallies 50 --output data/rallies_v2.pkl
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import itertools
import pickle as pkl

import numpy as np
from stable_baselines3 import PPO

from nash_skills.env_wrapper import SkillEnv
from nash_skills.skills import SKILL_NAMES
from nash_skills.v2.state_encoder import encode_ego, encode_opp
from nash_skills.v2.labeling import detect_winner, summarise_balance, check_balance

# --------------------------------------------------------------------------- #
# Defaults                                                                     #
# --------------------------------------------------------------------------- #
PPO_MODEL_PATH        = "logs/best_model_tracker1/best_model"
DEFAULT_OUTPUT        = "data/rallies_v2.pkl"
TARGET_RALLIES        = 50    # rallies per skill pair (25 pairs → 1250 total)
MAX_STEPS_PER_EPISODE = 800   # step cap per episode (headless, no real-time)
HISTORY               = 4
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


def collect(
    target_rallies: int = TARGET_RALLIES,
    output_path: str = DEFAULT_OUTPUT,
    ppo_path: str = PPO_MODEL_PATH,
) -> list:
    """
    Collect `target_rallies` complete rallies for each of the 25 skill pairs.

    Returns the full list of rally dicts (also saved to output_path).
    """
    env   = SkillEnv(proc_id=1, history=HISTORY)
    model = PPO.load(ppo_path)

    all_rallies = []

    for skill1, skill2 in itertools.product(SKILL_NAMES, SKILL_NAMES):
        print(f"\n=== Collecting: ego={skill1}  opp={skill2} ===")
        env.set_skills(skill1, skill2)
        obs, info = env.reset()

        completed = 0
        steps_this_combo = 0
        prev_ball_x = obs[36]

        # Per-episode state
        curr_states  = []   # encoded (76-dim)
        curr_raw     = []   # raw obs (116-dim)
        steps_in_ep  = 0

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

            curr_ball_x = obs[36]
            steps_in_ep += 1
            steps_this_combo += 1

            # Record state at each net crossing
            if (prev_ball_x - 1.5) * (curr_ball_x - 1.5) < 0:
                curr_states.append(encode_ego(obs, info))
                curr_raw.append(obs.copy())

            prev_ball_x = curr_ball_x

            # Episode ended (done) or step-capped (truncated)
            if done or steps_in_ep >= MAX_STEPS_PER_EPISODE:
                winner = detect_winner(curr_raw, done=done)
                if len(curr_states) > 0:
                    all_rallies.append({
                        "skill1":  skill1,
                        "skill2":  skill2,
                        "states":  curr_states,
                        "raw_obs": curr_raw,
                        "winner":  winner,
                    })
                    completed += 1
                    if completed % 10 == 0:
                        print(f"  {completed}/{target_rallies} rallies  "
                              f"({steps_this_combo} steps so far)")

                # Reset for next episode
                curr_states = []
                curr_raw    = []
                steps_in_ep = 0
                env.set_skills(skill1, skill2)
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
    is_ok, ratio = check_balance(all_rallies, threshold=5.0)
    print(f"\nBalance check: max/min ratio = {ratio:.2f} "
          f"({'OK' if is_ok else 'IMBALANCED — consider increasing target_rallies'})")
    print("Per-pair counts:")
    for (s1, s2), cnt in sorted(counts.items()):
        print(f"  {s1:12s} vs {s2:12s}: {cnt}")

    return all_rallies


# --------------------------------------------------------------------------- #
# CLI entry point                                                               #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect balanced high-level rally data for the v2 Nash pipeline."
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
