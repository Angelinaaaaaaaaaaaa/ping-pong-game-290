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
  - Collect exactly TARGET_RALLIES complete rallies per skill pair.
    This guarantees a balanced dataset regardless of rally length.
  - Store both the encoded state (76-dim) and the raw obs (116-dim, for inspection).
  - Record winner (1/2) at episode end; discard truncated episodes.
  - Print a balance summary after collection.
  - Cap maximum steps per episode to avoid degenerate infinite rallies.

Output format (pickle list of dicts)
--------------------------------------
Each entry:
    {
        'skill1' : str,                  # ego skill name
        'skill2' : str,                  # opp skill name
        'states' : list[np.ndarray],     # encoded states, shape (76,) each
        'raw_obs': list[np.ndarray],     # raw 116-dim obs (for debugging)
        'winner' : int,                  # 1=ego, 2=opp (truncated episodes discarded)
    }

Run:
    MUJOCO_GL=egl MPLCONFIGDIR=/tmp/matplotlib \
        venv/bin/python nash_skills/v2/collect_data.py
    MUJOCO_GL=egl MPLCONFIGDIR=/tmp/matplotlib \
        venv/bin/python nash_skills/v2/collect_data.py \
        --rallies 50 --output data/rallies_5skill_v2.pkl
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import csv
import itertools
import json
import pickle as pkl
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from diagnostic_fixed_skill import build_ppo_obs
from nash_skills.skills import SKILL_NAMES, SKILL_PROFILE_NAMES
from nash_skills.v2.state_encoder import encode_ego
from nash_skills.v2.labeling import detect_winner, summarise_balance, check_balance

PPO = None
SkillEnv = None

# --------------------------------------------------------------------------- #
# Defaults                                                                     #
# --------------------------------------------------------------------------- #
PPO_MODEL_PATH        = "logs/best_model_tracker1/best_model"
DEFAULT_OUTPUT        = "data/rallies_5skill_v2.pkl"
TARGET_RALLIES        = 50    # rallies per skill pair (25 pairs → 1250 total)
MAX_STEPS_PER_EPISODE = 800   # step cap per episode (headless, no real-time)
HISTORY               = 4
# --------------------------------------------------------------------------- #


def _build_ppo_obs(obs, info, player: int) -> np.ndarray:
    """Build the PPO input using the same helper as diagnostics."""
    return build_ppo_obs(obs, info, player=player)


def choose_skill(policy_type: str, fixed_skill: str | None, rng: np.random.Generator) -> str:
    if policy_type == "fixed":
        if fixed_skill is None:
            raise ValueError("fixed policy requires fixed_skill")
        return fixed_skill
    if policy_type == "random":
        return str(rng.choice(SKILL_NAMES))
    raise ValueError(f"Unknown policy_type: {policy_type}")


def episode_rng(seed: int, mode_index: int, setting_index: int, episode_index: int) -> np.random.Generator:
    value = (
        (seed + 1_000_003) * 1_000_003
        + mode_index * 100_003
        + setting_index * 10_007
        + episode_index
    ) % (2**32 - 1)
    return np.random.default_rng(value)


def policy_types_for_mode(mode: str, fixed_player: int | None) -> tuple[str, str]:
    if mode == "random":
        return "random", "random"
    if mode == "fixed_random":
        if fixed_player == 1:
            return "fixed", "random"
        if fixed_player == 2:
            return "random", "fixed"
        raise ValueError("--fixed-player must be 1 or 2 for fixed_random")
    if mode == "grid":
        return "fixed", "fixed"
    raise ValueError(f"Unsupported mode: {mode}")


def settings_for_mode(mode: str, fixed_player: int, fixed_skill: str | None) -> list[dict[str, Any]]:
    if mode == "grid":
        return [
            {
                "setting": f"{skill1}_vs_{skill2}",
                "skill1": skill1,
                "skill2": skill2,
                "fixed_player": None,
                "fixed_skill": None,
                "setting_index": idx,
            }
            for idx, (skill1, skill2) in enumerate(itertools.product(SKILL_NAMES, SKILL_NAMES))
        ]
    if mode == "random":
        return [{"setting": "random_vs_random", "skill1": None, "skill2": None, "fixed_player": None, "fixed_skill": None, "setting_index": 0}]
    if mode == "fixed_random":
        skills = [fixed_skill] if fixed_skill is not None else list(SKILL_NAMES)
        return [
            {
                "setting": f"p{fixed_player}_fixed_{skill}",
                "skill1": skill if fixed_player == 1 else None,
                "skill2": skill if fixed_player == 2 else None,
                "fixed_player": fixed_player,
                "fixed_skill": skill,
                "setting_index": skill_index_offset(fixed_player, skill),
            }
            for skill in skills
        ]
    raise ValueError(f"Unsupported mode: {mode}")


def skill_index_offset(fixed_player: int, skill: str) -> int:
    base = 0 if fixed_player == 1 else len(SKILL_NAMES)
    return base + SKILL_NAMES.index(skill)


def initial_skill_pair(mode: str, setting: dict[str, Any], rng: np.random.Generator) -> tuple[str, str]:
    if mode == "grid":
        return str(setting["skill1"]), str(setting["skill2"])
    p1_policy, p2_policy = policy_types_for_mode(mode, setting.get("fixed_player"))
    return (
        choose_skill(p1_policy, setting.get("fixed_skill"), rng),
        choose_skill(p2_policy, setting.get("fixed_skill"), rng),
    )


def next_skill_pair(mode: str, setting: dict[str, Any], rng: np.random.Generator, current: tuple[str, str]) -> tuple[str, str]:
    if mode == "grid":
        return current
    p1_policy, p2_policy = policy_types_for_mode(mode, setting.get("fixed_player"))
    return (
        choose_skill(p1_policy, setting.get("fixed_skill"), rng),
        choose_skill(p2_policy, setting.get("fixed_skill"), rng),
    )


def metadata_paths(output_path: str) -> tuple[Path, Path]:
    path = Path(output_path)
    return path.with_name(f"{path.stem}_metadata.csv"), path.with_name(f"{path.stem}_metadata.json")


def _mean(values: list[int]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _median(values: list[int]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _percent(part: int, total: int) -> float:
    return 100.0 * part / total if total else 0.0


def crossing_bucket_percentages(crossings: list[int]) -> dict[str, float]:
    total = len(crossings)
    return {
        "crossings_0_pct": _percent(sum(v == 0 for v in crossings), total),
        "crossings_1_pct": _percent(sum(v == 1 for v in crossings), total),
        "crossings_2plus_pct": _percent(sum(v >= 2 for v in crossings), total),
    }


def aggregate_attempt_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted_rows = [row for row in rows if row["accepted"]]
    accepted_crossings = [int(row["net_crossings"]) for row in accepted_rows]
    bucket_pcts = crossing_bucket_percentages(accepted_crossings)
    return {
        "total_attempts": len(rows),
        "accepted_rallies": len(accepted_rows),
        "discarded_truncated_rallies": len(rows) - len(accepted_rows),
        "avg_net_crossings_per_attempted_rally": _mean([int(row["net_crossings"]) for row in rows]),
        "avg_net_crossings_per_accepted_rally": _mean(accepted_crossings),
        "median_net_crossings_per_accepted_rally": _median(accepted_crossings),
        **bucket_pcts,
        "avg_recorded_states_per_accepted_rally": _mean([int(row["recorded_states"]) for row in accepted_rows]),
        "avg_physics_steps_per_accepted_rally": _mean([int(row["steps"]) for row in accepted_rows]),
    }


def setting_attempt_stats(rows: list[dict[str, Any]], setting: str) -> dict[str, Any]:
    setting_rows = [row for row in rows if row["setting"] == setting]
    accepted_rows = [row for row in setting_rows if row["accepted"]]
    accepted_crossings = [int(row["net_crossings"]) for row in accepted_rows]
    bucket_pcts = crossing_bucket_percentages(accepted_crossings)
    return {
        "attempts": len(setting_rows),
        "accepted": len(accepted_rows),
        "discarded": len(setting_rows) - len(accepted_rows),
        "avg_crossings_accepted": _mean(accepted_crossings),
        "median_crossings_accepted": _median(accepted_crossings),
        **bucket_pcts,
        "avg_steps_accepted": _mean([int(row["steps"]) for row in accepted_rows]),
    }


def accepted_initial_pair_counts(rallies: list[dict[str, Any]]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for rally in rallies:
        counts[(rally["skill1"], rally["skill2"])] += 1
    return counts


def active_decision_state_pair_counts(rallies: list[dict[str, Any]]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for rally in rallies:
        for _state, pair in zip(rally.get("states", []), rally.get("skill_pairs", [])):
            counts[tuple(pair)] += 1
    return counts


def setting_display_name(mode: str, setting: dict[str, Any]) -> str:
    if mode == "fixed_random":
        fixed_player = int(setting["fixed_player"])
        random_player = 2 if fixed_player == 1 else 1
        return f"P{fixed_player} fixed {setting['fixed_skill']} vs P{random_player} random"
    if mode == "random":
        return "P1 random vs P2 random"
    return f"{setting['skill1']} vs {setting['skill2']}"


def _extract_successful_returns(info: dict[str, Any]) -> int | None:
    for key in ("successful_returns", "return_success_count", "success_returns"):
        if key in info:
            return int(info[key])
    return None


def _log(message: str = "", *, quiet: bool = False, flush: bool = False) -> None:
    if not quiet:
        print(message, flush=flush)


def write_collection_metadata(output_path: str, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    csv_path, json_path = metadata_paths(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "attempt_id",
        "mode",
        "setting",
        "seed",
        "episode_index",
        "accepted",
        "truncated",
        "discard_reason",
        "steps",
        "net_crossings",
        "recorded_states",
        "decision_count",
        "successful_returns",
        "winner",
        "skill_sequence",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"summary": summary, "attempts": rows}, indent=2))


def collect(
    target_rallies: int = TARGET_RALLIES,
    output_path: str = DEFAULT_OUTPUT,
    ppo_path: str = PPO_MODEL_PATH,
    max_steps_per_episode: int = MAX_STEPS_PER_EPISODE,
    max_attempts_per_pair: int | None = None,
    progress_every: int = 10,
    mode: str = "grid",
    fixed_player: int = 1,
    fixed_skill: str | None = None,
    reset_mode: str = "ready",
    skill_profile: str = "aggressive",
    gantry_speed_scale: float = 1.0,
    seed: int = 0,
    quiet: bool = False,
    accepted_progress_callback=None,
) -> list:
    """
    Collect `target_rallies` complete rallies for each of the 25 skill pairs.

    Returns the full list of rally dicts (also saved to output_path).
    """
    global SkillEnv
    if SkillEnv is None:
        from nash_skills.env_wrapper import SkillEnv as _SkillEnv
        SkillEnv = _SkillEnv

    env   = SkillEnv(
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

    all_rallies = []
    pair_summaries = []
    incomplete_pairs = []
    attempt_rows = []
    mode_index = {"fixed_random": 0, "random": 1, "grid": 2}[mode]
    settings = settings_for_mode(mode, fixed_player, fixed_skill)

    try:
        for setting_index, setting in enumerate(settings):
            _log(f"\n=== Collecting: {mode} {setting['setting']} ===", quiet=quiet, flush=True)
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
                rng = episode_rng(seed, mode_index, int(setting["setting_index"]), attempts)
                skill1, skill2 = initial_skill_pair(mode, setting, rng)
                skill_sequence = [(skill1, skill2)]
                env.set_skills(skill1, skill2)
                obs, info = env.reset()
                prev_ball_x = obs[36]

                curr_states = []
                curr_raw = []
                curr_skill_pairs = []
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
                        curr_states.append(encode_ego(obs, info))
                        curr_raw.append(obs.copy())
                        curr_skill_pairs.append((skill1, skill2))
                        skill1, skill2 = next_skill_pair(mode, setting, rng, (skill1, skill2))
                        env.set_skills(skill1, skill2)
                        skill_sequence.append((skill1, skill2))

                    prev_ball_x = curr_ball_x

                attempts += 1
                steps_completed_attempts += steps_in_ep
                terminal_raw = curr_raw + [obs.copy()]
                winner = detect_winner(terminal_raw, done=done, info=info)
                truncated = not done
                net_crossings = len(curr_states)
                recorded_states = len(curr_states)
                successful_returns = _extract_successful_returns(info)
                last_winner = winner
                accepted = False

                if not truncated and len(curr_states) > 0 and winner in (1, 2):
                    all_rallies.append({
                        "skill1": curr_skill_pairs[0][0],
                        "skill2": curr_skill_pairs[0][1],
                        "skill_pairs": curr_skill_pairs,
                        "states": curr_states,
                        "raw_obs": curr_raw,
                        "winner": winner,
                    })
                    completed += 1
                    accepted = True
                    last_reason = "accepted-done"
                    if accepted_progress_callback is not None:
                        accepted_progress_callback({
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
                            ) + steps_in_ep,
                            "net_crossings_accepted_total": sum(
                                int(row["net_crossings"])
                                for row in attempt_rows
                                if row["setting"] == setting["setting"] and row["accepted"]
                            ) + net_crossings,
                        })
                    if completed % 10 == 0:
                        _log(
                            f"  {completed}/{target_rallies} rallies  "
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

                attempt_rows.append({
                    "attempt_id": len(attempt_rows),
                    "mode": mode,
                    "setting": setting["setting"],
                    "seed": seed,
                    "episode_index": attempts - 1,
                    "accepted": accepted,
                    "truncated": truncated,
                    "discard_reason": "" if accepted else last_reason,
                    "steps": steps_in_ep,
                    "net_crossings": net_crossings,
                    "recorded_states": recorded_states,
                    "decision_count": recorded_states,
                    "successful_returns": successful_returns,
                    "winner": winner,
                    "skill_sequence": json.dumps(skill_sequence),
                })

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

    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pkl.dump(all_rallies, f)
    _log(f"\nSaved {len(all_rallies)} rallies to {output_path}", quiet=quiet)

    _log("\nDiagnostic pair summary:", quiet=quiet)
    for s in pair_summaries:
        _log(
            f"  {s['display_name']}: "
            f"attempts={s['attempts']:4d} accepted={s['accepted']:4d} "
            f"discarded={s['discarded']:4d} elapsed={s['elapsed']:.1f}s "
            f"avg_steps/attempt={s['avg_steps_per_attempt']:.1f} "
            f"last={s['last_reason']} winner={s['last_winner']}",
            quiet=quiet,
        )

    if incomplete_pairs:
        _log(
            "\nWARNING: incomplete collection. "
            "This output is diagnostic/smoke-test data, not a valid full dataset.",
            quiet=quiet,
        )
        _log("Incomplete pairs:", quiet=quiet)
        for setting in incomplete_pairs:
            _log(f"  {setting}", quiet=quiet)

    aggregate_stats = aggregate_attempt_stats(attempt_rows)
    metadata_summary = {
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

    _log("\nCollection summary:", quiet=quiet)
    _log(f"  total attempts: {aggregate_stats['total_attempts']}", quiet=quiet)
    _log(f"  accepted rallies: {aggregate_stats['accepted_rallies']}", quiet=quiet)
    _log(f"  discarded/truncated rallies: {aggregate_stats['discarded_truncated_rallies']}", quiet=quiet)
    _log(f"  avg net crossings per attempted rally: {aggregate_stats['avg_net_crossings_per_attempted_rally']:.1f}", quiet=quiet)
    _log(f"  avg net crossings per accepted rally: {aggregate_stats['avg_net_crossings_per_accepted_rally']:.1f}", quiet=quiet)
    _log(f"  median net crossings per accepted rally: {aggregate_stats['median_net_crossings_per_accepted_rally']:.1f}", quiet=quiet)
    _log(f"  accepted rallies with 0 crossings: {aggregate_stats['crossings_0_pct']:.1f}%", quiet=quiet)
    _log(f"  accepted rallies with 1 crossing: {aggregate_stats['crossings_1_pct']:.1f}%", quiet=quiet)
    _log(f"  accepted rallies with 2+ crossings: {aggregate_stats['crossings_2plus_pct']:.1f}%", quiet=quiet)
    _log(f"  avg recorded states per accepted rally: {aggregate_stats['avg_recorded_states_per_accepted_rally']:.1f}", quiet=quiet)
    _log(f"  avg physics steps per accepted rally: {aggregate_stats['avg_physics_steps_per_accepted_rally']:.1f}", quiet=quiet)

    _log("\nCollection setting summaries:", quiet=quiet)
    for s in pair_summaries:
        _log(f"  {s['display_name']}:", quiet=quiet)
        _log(f"    attempts={s['attempts']}", quiet=quiet)
        _log(f"    accepted={s['accepted']}", quiet=quiet)
        _log(f"    discarded={s['discarded']}", quiet=quiet)
        _log(f"    avg_crossings_accepted={s['avg_crossings_accepted']:.1f}", quiet=quiet)
        _log(f"    median_crossings_accepted={s['median_crossings_accepted']:.1f}", quiet=quiet)
        _log(f"    crossings_0={s['crossings_0_pct']:.1f}%", quiet=quiet)
        _log(f"    crossings_1={s['crossings_1_pct']:.1f}%", quiet=quiet)
        _log(f"    crossings_2plus={s['crossings_2plus_pct']:.1f}%", quiet=quiet)
        _log(f"    avg_steps_accepted={s['avg_steps_accepted']:.1f}", quiet=quiet)

    # Balance report
    initial_counts = accepted_initial_pair_counts(all_rallies)
    active_counts = active_decision_state_pair_counts(all_rallies)
    counts = summarise_balance(all_rallies)
    is_ok, ratio = check_balance(all_rallies, threshold=5.0)
    _log(
        f"\nBalance check: max/min ratio = {ratio:.2f} "
        f"({'OK' if is_ok else 'IMBALANCED — consider increasing target_rallies'})",
        quiet=quiet,
    )
    if mode == "grid":
        _log("Accepted rallies by fixed skill pair (grid mode; pair stays fixed for the full rally):", quiet=quiet)
        for (s1, s2), cnt in sorted(counts.items()):
            _log(f"  {s1:12s} vs {s2:12s}: {cnt}", quiet=quiet)
    else:
        _log("Accepted rallies by initial skill pair:", quiet=quiet)
        for (s1, s2), cnt in sorted(initial_counts.items()):
            _log(f"  {s1:12s} vs {s2:12s}: {cnt}", quiet=quiet)
        _log("Recorded decision states by active skill pair:", quiet=quiet)
        for (s1, s2), cnt in sorted(active_counts.items()):
            _log(f"  {s1:12s} vs {s2:12s}: {cnt}", quiet=quiet)

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
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_EPISODE,
                        help=f"Maximum steps per rally attempt (default: {MAX_STEPS_PER_EPISODE})")
    parser.add_argument("--max-attempts-per-pair", type=int, default=None,
                        help="Stop each skill pair after this many attempts, even if target rallies are not reached")
    parser.add_argument("--progress-every", type=int, default=10,
                        help="Print progress every N completed attempts per skill pair; <=0 disables attempt progress")
    parser.add_argument("--mode", choices=["grid", "random", "fixed_random"], default="grid",
                        help="Skill sampling mode: grid keeps fixed pairs; random resamples both; fixed_random fixes one player")
    parser.add_argument("--fixed-player", type=int, choices=[1, 2], default=1,
                        help="Fixed player for --mode fixed_random")
    parser.add_argument("--fixed-skill", choices=SKILL_NAMES, default=None,
                        help="Optional fixed skill for --mode fixed_random; defaults to all skills")
    parser.add_argument("--reset-mode", choices=["clean", "ready", "carryover"], default="ready",
                        help="SkillEnv reset mode")
    parser.add_argument("--skill-profile", choices=SKILL_PROFILE_NAMES, default="aggressive",
                        help="Skill target profile")
    parser.add_argument("--gantry-speed-scale", type=float, default=1.0,
                        help="Gantry movement speed multiplier")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed used for deterministic high-level skill sampling")
    args = parser.parse_args()

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
    )
