"""
Aligned decision-state collection for the 5-skill v3 Nash pipeline.

This script is based on nash_skills/v2/collect_data.py, but fixes the
state/skill timing for randomized collection modes. At each net crossing it:

  1. observes the crossing state S,
  2. samples the next skill pair using the existing mode logic,
  3. stores S with that next pair in both state[-2:] and skill_pairs,
  4. applies that pair for the following transition.

The saved rally dict remains compatible with the current v3 trainer:

    {
        "skill1": str,
        "skill2": str,
        "states": list[np.ndarray],       # encoded 76-dim states
        "raw_obs": list[np.ndarray],      # raw 116-dim observations
        "skill_pairs": list[(str, str)],  # (P1 skill, P2 skill)
        "winner": 1 or 2,
    }

Run:
    MUJOCO_GL=egl MPLCONFIGDIR=/tmp/matplotlib \
        venv/bin/python nash_skills/v2/collect_data_v3_aligned.py \
        --mode random --rallies 10 --output data/rallies_5skill_v3_aligned.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle as pkl
import sys
import time
from collections import Counter
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from nash_skills.skills import SKILL_NAMES, SKILL_PROFILE_NAMES, N_SKILLS, skill_index
from nash_skills.v2.collect_data import (
    HISTORY,
    MAX_STEPS_PER_EPISODE,
    PPO_MODEL_PATH,
    TARGET_RALLIES,
    _build_ppo_obs,
    _extract_successful_returns,
    _log,
    accepted_initial_pair_counts,
    active_decision_state_pair_counts,
    aggregate_attempt_stats,
    episode_rng,
    initial_skill_pair,
    next_skill_pair,
    setting_attempt_stats,
    setting_display_name,
    settings_for_mode,
    write_collection_metadata,
)
from nash_skills.v2.labeling import check_balance, detect_winner, summarise_balance
from nash_skills.v2.state_encoder import encode_ego


PPO = None
SkillEnv = None

DEFAULT_OUTPUT = "data/rallies_5skill_v3_aligned.pkl"


def _skill_value(skill: str) -> float:
    return skill_index(skill) / (N_SKILLS - 1)


def _encode_state_for_pair(obs: np.ndarray, info: dict[str, Any], skill_pair: tuple[str, str]) -> np.ndarray:
    state = encode_ego(obs, info)
    state[-2] = _skill_value(skill_pair[0])
    state[-1] = _skill_value(skill_pair[1])
    return state


def _raw_obs_for_pair(obs: np.ndarray, skill_pair: tuple[str, str]) -> np.ndarray:
    raw = obs.copy()
    raw[-2] = _skill_value(skill_pair[0])
    raw[-1] = _skill_value(skill_pair[1])
    return raw


def _debug_crossing_log(
    *,
    quiet: bool,
    debug_crossings: bool,
    attempt_id: int,
    crossing_index: int,
    old_pair: tuple[str, str],
    new_pair: tuple[str, str],
    stored_pair: tuple[str, str],
) -> None:
    if not debug_crossings:
        return
    _log(
        "  crossing "
        f"attempt={attempt_id} index={crossing_index} "
        f"old_pair={old_pair} new_pair={new_pair} stored_pair={stored_pair}",
        quiet=quiet,
        flush=True,
    )


def collect(
    target_rallies: int = TARGET_RALLIES,
    output_path: str = DEFAULT_OUTPUT,
    ppo_path: str = PPO_MODEL_PATH,
    max_steps_per_episode: int = MAX_STEPS_PER_EPISODE,
    max_attempts_per_pair: int | None = None,
    progress_every: int = 10,
    mode: str = "random",
    fixed_player: int = 1,
    fixed_skill: str | None = None,
    reset_mode: str = "ready",
    skill_profile: str = "aggressive",
    gantry_speed_scale: float = 1.0,
    seed: int = 0,
    quiet: bool = False,
    debug_crossings: bool = False,
    accepted_progress_callback=None,
) -> list[dict[str, Any]]:
    """
    Collect aligned decision-state rallies.

    Random and fixed_random modes resample at each net crossing before storing
    the state/skill pair. Grid mode is still supported and remains equivalent
    to the original fixed-pair collection because next_skill_pair() returns the
    current pair.
    """
    global SkillEnv
    if SkillEnv is None:
        from nash_skills.env_wrapper import SkillEnv as _SkillEnv

        SkillEnv = _SkillEnv

    env = SkillEnv(
        proc_id=1,
        history=HISTORY,
        reset_mode=reset_mode,
        skill_profile=skill_profile,
        gantry_speed_scale=gantry_speed_scale,
    )

    global PPO
    if PPO is None:
        from stable_baselines3 import PPO as _PPO

        PPO = _PPO
    model = PPO.load(ppo_path)

    all_rallies: list[dict[str, Any]] = []
    pair_summaries: list[dict[str, Any]] = []
    incomplete_pairs: list[str] = []
    attempt_rows: list[dict[str, Any]] = []
    mode_index = {"fixed_random": 0, "random": 1, "grid": 2}[mode]
    settings = settings_for_mode(mode, fixed_player, fixed_skill)

    try:
        for setting in settings:
            _log(f"\n=== Aligned collecting: {mode} {setting['setting']} ===", quiet=quiet, flush=True)
            completed = 0
            attempts = 0
            discarded = 0
            steps_this_combo = 0
            steps_completed_attempts = 0
            pair_start_time = time.monotonic()
            last_reason = "not-started"
            last_winner = None

            while completed < target_rallies and (
                max_attempts_per_pair is None or attempts < max_attempts_per_pair
            ):
                attempt_id = len(attempt_rows)
                rng = episode_rng(seed, mode_index, int(setting["setting_index"]), attempts)
                skill1, skill2 = initial_skill_pair(mode, setting, rng)
                skill_sequence = [(skill1, skill2)]
                env.set_skills(skill1, skill2)
                obs, info = env.reset()
                prev_ball_x = obs[36]

                curr_states: list[np.ndarray] = []
                curr_raw: list[np.ndarray] = []
                curr_skill_pairs: list[tuple[str, str]] = []
                steps_in_ep = 0
                done = False

                while not done and steps_in_ep < max_steps_per_episode:
                    ppo1 = _build_ppo_obs(obs, info, player=1)
                    ppo2 = _build_ppo_obs(obs, info, player=2)
                    a1, _ = model.predict(ppo1, deterministic=True)
                    a2, _ = model.predict(ppo2, deterministic=True)

                    action = np.zeros(18, dtype=np.float32)
                    action[:9] = a1[:9]
                    action[9:] = a2[:9]

                    obs, _reward, done, _, info = env.step(action)
                    curr_ball_x = obs[36]
                    steps_in_ep += 1
                    steps_this_combo += 1

                    if (prev_ball_x - 1.5) * (curr_ball_x - 1.5) < 0:
                        old_pair = (skill1, skill2)
                        new_pair = next_skill_pair(mode, setting, rng, old_pair)
                        stored_pair = new_pair

                        curr_states.append(_encode_state_for_pair(obs, info, stored_pair))
                        curr_raw.append(_raw_obs_for_pair(obs, stored_pair))
                        curr_skill_pairs.append(stored_pair)
                        skill_sequence.append(stored_pair)
                        _debug_crossing_log(
                            quiet=quiet,
                            debug_crossings=debug_crossings,
                            attempt_id=attempt_id,
                            crossing_index=len(curr_states) - 1,
                            old_pair=old_pair,
                            new_pair=new_pair,
                            stored_pair=stored_pair,
                        )

                        skill1, skill2 = new_pair
                        env.set_skills(skill1, skill2)

                    prev_ball_x = curr_ball_x

                attempts += 1
                steps_completed_attempts += steps_in_ep
                terminal_raw = curr_raw + [obs.copy()]
                winner = detect_winner(terminal_raw, done=done, info=info)
                truncated = not done
                net_crossings = len(curr_states)
                successful_returns = _extract_successful_returns(info)
                last_winner = winner
                accepted = False

                if not truncated and len(curr_states) > 0 and winner in (1, 2):
                    all_rallies.append(
                        {
                            "skill1": curr_skill_pairs[0][0],
                            "skill2": curr_skill_pairs[0][1],
                            "skill_pairs": curr_skill_pairs,
                            "states": curr_states,
                            "raw_obs": curr_raw,
                            "winner": winner,
                        }
                    )
                    completed += 1
                    accepted = True
                    last_reason = "accepted-done"
                    if accepted_progress_callback is not None:
                        accepted_progress_callback(
                            {
                                "mode": mode,
                                "setting": setting["setting"],
                                "display_name": setting_display_name(mode, setting),
                                "target": target_rallies,
                                "accepted": completed,
                                "attempts": attempts,
                                "discarded": discarded,
                                "elapsed": time.monotonic() - pair_start_time,
                                "steps_accepted_total": sum(
                                    int(row["steps"])
                                    for row in attempt_rows
                                    if row["setting"] == setting["setting"] and row["accepted"]
                                )
                                + steps_in_ep,
                                "net_crossings_accepted_total": sum(
                                    int(row["net_crossings"])
                                    for row in attempt_rows
                                    if row["setting"] == setting["setting"] and row["accepted"]
                                )
                                + net_crossings,
                            }
                        )
                    if completed % 10 == 0:
                        _log(
                            f"  {completed}/{target_rallies} rallies "
                            f"({steps_this_combo} steps so far)",
                            quiet=quiet,
                            flush=True,
                        )
                else:
                    discarded += 1
                    if truncated:
                        last_reason = "discarded-step-cap"
                        winner = 0
                        last_winner = 0
                    elif len(curr_states) == 0:
                        last_reason = "discarded-done-no-crossing"
                    else:
                        last_reason = "discarded-done-inconclusive"

                attempt_rows.append(
                    {
                        "attempt_id": attempt_id,
                        "mode": mode,
                        "setting": setting["setting"],
                        "seed": seed,
                        "episode_index": attempts - 1,
                        "accepted": accepted,
                        "truncated": truncated,
                        "discard_reason": "" if accepted else last_reason,
                        "steps": steps_in_ep,
                        "net_crossings": net_crossings,
                        "recorded_states": net_crossings,
                        "decision_count": net_crossings,
                        "successful_returns": successful_returns,
                        "winner": winner,
                        "skill_sequence": json.dumps(skill_sequence),
                    }
                )

                if progress_every > 0 and attempts % progress_every == 0:
                    elapsed = time.monotonic() - pair_start_time
                    avg_steps = steps_completed_attempts / attempts
                    _log(
                        f"  progress setting={setting['setting']} "
                        f"attempts={attempts} accepted={completed}/{target_rallies} "
                        f"discarded={discarded} elapsed={elapsed:.1f}s "
                        f"avg_steps/attempt={avg_steps:.1f} "
                        f"last={last_reason} winner={last_winner}",
                        quiet=quiet,
                        flush=True,
                    )

            elapsed = time.monotonic() - pair_start_time
            avg_steps = steps_completed_attempts / attempts if attempts else 0.0
            summary = {
                "setting": setting["setting"],
                "display_name": setting_display_name(mode, setting),
                "skill1": setting.get("skill1") or "",
                "skill2": setting.get("skill2") or "",
                "attempts": attempts,
                "accepted": completed,
                "discarded": discarded,
                "elapsed": elapsed,
                "avg_steps_per_attempt": avg_steps,
                "last_reason": last_reason,
                "last_winner": last_winner,
            }
            summary.update(setting_attempt_stats(attempt_rows, setting["setting"]))
            pair_summaries.append(summary)
            if completed < target_rallies:
                incomplete_pairs.append(setting["setting"])
            _log(
                f"  Summary {setting['setting']}: "
                f"attempts={attempts} accepted={completed}/{target_rallies} "
                f"discarded={discarded} elapsed={elapsed:.1f}s "
                f"avg_steps/attempt={avg_steps:.1f} "
                f"last={last_reason} winner={last_winner}",
                quiet=quiet,
                flush=True,
            )
    finally:
        env.close()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pkl.dump(all_rallies, f)
    _log(f"\nSaved {len(all_rallies)} aligned rallies to {output_path}", quiet=quiet)

    if incomplete_pairs:
        _log(
            "\nWARNING: incomplete aligned collection. "
            "This output is diagnostic/smoke-test data, not a valid full dataset.",
            quiet=quiet,
        )
        for setting_name in incomplete_pairs:
            _log(f"  {setting_name}", quiet=quiet)

    aggregate_stats = aggregate_attempt_stats(attempt_rows)
    metadata_summary = {
        "collector": "collect_data_v3_aligned",
        "alignment": "stored skill_pairs and state[-2:] are sampled before storage and applied after crossing",
        "mode": mode,
        "target_rallies": target_rallies,
        "attempts": aggregate_stats["total_attempts"],
        "accepted": aggregate_stats["accepted_rallies"],
        "truncated_discarded": sum(1 for row in attempt_rows if row["truncated"]),
        "discarded": aggregate_stats["discarded_truncated_rallies"],
        "reset_mode": reset_mode,
        "skill_profile": skill_profile,
        "gantry_speed_scale": gantry_speed_scale,
        "seed": seed,
        "attempt_stats": aggregate_stats,
        "settings": pair_summaries,
    }
    write_collection_metadata(output_path, attempt_rows, metadata_summary)

    _log("\nAligned collection summary:", quiet=quiet)
    _log(f"  total attempts: {aggregate_stats['total_attempts']}", quiet=quiet)
    _log(f"  accepted rallies: {aggregate_stats['accepted_rallies']}", quiet=quiet)
    _log(f"  discarded/truncated rallies: {aggregate_stats['discarded_truncated_rallies']}", quiet=quiet)
    _log(f"  avg recorded states per accepted rally: {aggregate_stats['avg_recorded_states_per_accepted_rally']:.1f}", quiet=quiet)

    initial_counts: Counter[tuple[str, str]] = accepted_initial_pair_counts(all_rallies)
    active_counts: Counter[tuple[str, str]] = active_decision_state_pair_counts(all_rallies)
    is_ok, ratio = check_balance(all_rallies, threshold=5.0)
    _log(
        f"\nBalance check: max/min ratio = {ratio:.2f} "
        f"({'OK' if is_ok else 'IMBALANCED — consider increasing target_rallies'})",
        quiet=quiet,
    )
    if mode == "grid":
        counts = summarise_balance(all_rallies)
        for (s1, s2), cnt in sorted(counts.items()):
            _log(f"  {s1:12s} vs {s2:12s}: {cnt}", quiet=quiet)
    else:
        _log("Accepted rallies by first stored skill pair:", quiet=quiet)
        for (s1, s2), cnt in sorted(initial_counts.items()):
            _log(f"  {s1:12s} vs {s2:12s}: {cnt}", quiet=quiet)
        _log("Recorded decision states by aligned stored skill pair:", quiet=quiet)
        for (s1, s2), cnt in sorted(active_counts.items()):
            _log(f"  {s1:12s} vs {s2:12s}: {cnt}", quiet=quiet)

    return all_rallies


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect v3 aligned high-level rally data with next-skill timing."
    )
    parser.add_argument("--rallies", type=int, default=TARGET_RALLIES)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--ppo", type=str, default=PPO_MODEL_PATH)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_EPISODE)
    parser.add_argument("--max-attempts-per-pair", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--mode", choices=["grid", "random", "fixed_random"], default="random")
    parser.add_argument("--fixed-player", type=int, choices=[1, 2], default=1)
    parser.add_argument("--fixed-skill", choices=SKILL_NAMES, default=None)
    parser.add_argument("--reset-mode", choices=["clean", "ready", "carryover"], default="ready")
    parser.add_argument("--skill-profile", choices=SKILL_PROFILE_NAMES, default="aggressive")
    parser.add_argument("--gantry-speed-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--debug-crossings",
        action="store_true",
        help="Print crossing index, old pair, newly selected pair, and stored pair.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    collect(
        target_rallies=args.rallies,
        output_path=args.output,
        ppo_path=args.ppo,
        max_steps_per_episode=args.max_steps,
        max_attempts_per_pair=args.max_attempts_per_pair,
        progress_every=args.progress_every,
        mode=args.mode,
        fixed_player=args.fixed_player,
        fixed_skill=args.fixed_skill,
        reset_mode=args.reset_mode,
        skill_profile=args.skill_profile,
        gantry_speed_scale=args.gantry_speed_scale,
        seed=args.seed,
        debug_crossings=args.debug_crossings,
    )


if __name__ == "__main__":
    main()
