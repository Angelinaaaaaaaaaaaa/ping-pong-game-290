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
from collections import Counter
from typing import Iterable, List, Optional, Sequence, Tuple

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


def _validate_skill(name: str) -> str:
    if name not in SKILL_NAMES:
        valid = ", ".join(SKILL_NAMES)
        raise ValueError(f"Unknown skill '{name}'. Valid skills: {valid}")
    return name


def _selected_pairs(
    skill1: Optional[str] = None,
    skill2: Optional[str] = None,
    num_shards: int = 1,
    shard_index: int = 0,
) -> List[Tuple[str, str]]:
    """
    Return the skill pairs assigned to this collector run.

    Defaults to all 25 pairs.  ``--skill1`` / ``--skill2`` restrict the Cartesian
    product, and ``--num-shards`` / ``--shard-index`` split the resulting ordered
    pair list so multiple processes can collect independent chunks.
    """
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")

    skills1: Sequence[str] = [_validate_skill(skill1)] if skill1 else SKILL_NAMES
    skills2: Sequence[str] = [_validate_skill(skill2)] if skill2 else SKILL_NAMES
    pairs = list(itertools.product(skills1, skills2))
    return [pair for idx, pair in enumerate(pairs) if idx % num_shards == shard_index]


def _save_rallies_atomic(rallies: list, output_path: str, label: str) -> None:
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "wb") as f:
        pkl.dump(rallies, f)
    os.replace(tmp_path, output_path)
    print(f"  [checkpoint:{label}] saved {len(rallies)} rallies to {output_path}", flush=True)


def _load_existing_rallies(output_path: str) -> list:
    if not os.path.exists(output_path):
        return []
    with open(output_path, "rb") as f:
        rallies = pkl.load(f)
    if not isinstance(rallies, list):
        raise ValueError(f"Expected a list in existing output {output_path}")
    print(f"Resuming from {output_path}: loaded {len(rallies)} existing rallies")
    return rallies


def _count_pairs(rallies: list) -> Counter:
    counts = Counter()
    for entry in rallies:
        if "skill1" in entry and "skill2" in entry:
            counts[(entry["skill1"], entry["skill2"])] += 1
    return counts


def collect(
    target_rallies: int = TARGET_RALLIES,
    output_path: str = DEFAULT_OUTPUT,
    ppo_path: str = PPO_MODEL_PATH,
    pairs: Optional[Iterable[Tuple[str, str]]] = None,
    checkpoint_every: int = 100,
    resume: bool = True,
) -> list:
    """
    Collect `target_rallies` complete rallies for each of the 25 skill pairs.

    Returns the full list of rally dicts (also saved to output_path).
    """
    env   = SkillEnv(proc_id=1, history=HISTORY)
    model = PPO.load(ppo_path)

    selected = list(pairs) if pairs is not None else list(itertools.product(SKILL_NAMES, SKILL_NAMES))
    if not selected:
        raise ValueError("No skill pairs selected for collection.")

    print(f"Selected {len(selected)} skill pair(s):")
    for idx, (skill1, skill2) in enumerate(selected):
        print(f"  [{idx:02d}] {skill1} vs {skill2}")

    all_rallies = _load_existing_rallies(output_path) if resume else []
    existing_counts = _count_pairs(all_rallies)
    if all_rallies:
        print("Existing selected-pair counts:")
        for skill1, skill2 in selected:
            print(f"  {skill1:12s} vs {skill2:12s}: "
                  f"{existing_counts.get((skill1, skill2), 0)}/{target_rallies}")

    for skill1, skill2 in selected:
        already_collected = existing_counts.get((skill1, skill2), 0)
        if already_collected >= target_rallies:
            print(f"\n=== Skipping: ego={skill1}  opp={skill2} "
                  f"({already_collected}/{target_rallies} already collected) ===")
            continue

        print(f"\n=== Collecting: ego={skill1}  opp={skill2} ===")
        if already_collected > 0:
            print(f"  Resume: {already_collected}/{target_rallies} already collected; "
                  f"collecting {target_rallies - already_collected} more")
        env.set_skills(skill1, skill2)
        obs, info = env.reset()

        completed = already_collected
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
                terminal_raw = curr_raw + [obs.copy()] if done else curr_raw
                winner = detect_winner(terminal_raw, done=done, info=info)
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
                    if checkpoint_every > 0 and completed % checkpoint_every == 0:
                        _save_rallies_atomic(
                            all_rallies,
                            output_path,
                            label=f"{skill1}_vs_{skill2}_{completed}",
                        )

                # Reset for next episode
                curr_states = []
                curr_raw    = []
                steps_in_ep = 0
                env.set_skills(skill1, skill2)
                obs, info = env.reset()
                prev_ball_x = obs[36]

        print(f"  Done: {completed} rallies collected "
              f"({steps_this_combo} total steps)")
        _save_rallies_atomic(
            all_rallies,
            output_path,
            label=f"{skill1}_vs_{skill2}_done",
        )

    env.close()

    _save_rallies_atomic(all_rallies, output_path, label="final")
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
    parser.add_argument("--skill1",  type=str, default=None,
                        help="Optional ego skill filter. Default: all ego skills")
    parser.add_argument("--skill2",  type=str, default=None,
                        help="Optional opponent skill filter. Default: all opponent skills")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Split selected skill pairs across this many shards")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Collect only pairs whose ordered index mod num_shards equals this value")
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="Save partial output every N collected rallies per skill pair; 0 disables periodic checkpoints")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore an existing output file and start this shard from scratch")
    args = parser.parse_args()

    selected_pairs = _selected_pairs(
        skill1=args.skill1,
        skill2=args.skill2,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )

    collect(
        target_rallies=args.rallies,
        output_path=args.output,
        ppo_path=args.ppo,
        pairs=selected_pairs,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
    )
