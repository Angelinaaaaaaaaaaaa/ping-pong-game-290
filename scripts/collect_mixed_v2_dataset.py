#!/usr/bin/env python3
"""Collect and merge the mixed v2 training dataset."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import pickle as pkl
import shutil
import statistics
import sys
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nash_skills.skills import SKILL_NAMES, SKILL_PROFILE_NAMES
from nash_skills.v2 import collect_data

DEFAULT_OUTPUT = "data/rallies_5skill_v2_mixed_10k.pkl"


def fixed_random_settings() -> list[dict[str, Any]]:
    settings = []
    for fixed_player in (1, 2):
        for skill in SKILL_NAMES:
            settings.append({
                "mode": "fixed_random",
                "fixed_player": fixed_player,
                "fixed_skill": skill,
                "setting": f"p{fixed_player}_fixed_{skill}",
                "display_name": f"P{fixed_player} fixed {skill}",
            })
    return settings


def allocate_fixed_random(total: int) -> list[dict[str, Any]]:
    settings = fixed_random_settings()
    base, remainder = divmod(total, len(settings))
    for index, setting in enumerate(settings):
        setting["target"] = base + (1 if index < remainder else 0)
    return settings


def derived_seed(base_seed: int, mode: str, setting_index: int = 0) -> int:
    mode_offset = {"random": 17, "fixed_random": 10_009}[mode]
    return int((base_seed + 1_000_003) * 1_000_003 + mode_offset + setting_index * 10_007) % (2**32 - 1)


def output_metadata_paths(output_path: str) -> tuple[Path, Path]:
    path = Path(output_path)
    return path.with_name(f"{path.stem}_metadata.csv"), path.with_name(f"{path.stem}_metadata.json")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def should_suppress_line(line: str) -> bool:
    return "Returned successfully by ego" in line or "Returned successfully by opp" in line or line.startswith("Returned successfully ")


class ReturnLineFilter(io.TextIOBase):
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if not should_suppress_line(line):
                self.wrapped.write(line + "\n")
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            if not should_suppress_line(self._buffer):
                self.wrapped.write(self._buffer)
            self._buffer = ""
        self.wrapped.flush()


@contextmanager
def suppress_return_lines(enabled: bool):
    if not enabled:
        yield
        return
    original = sys.stdout
    filtered = ReturnLineFilter(original)
    sys.stdout = filtered
    try:
        yield
    finally:
        filtered.flush()
        sys.stdout = original


class ProgressPrinter:
    def __init__(self, interval: int):
        self.interval = max(1, interval)

    def __call__(self, snapshot: dict[str, Any]) -> None:
        accepted = int(snapshot["accepted"])
        target = int(snapshot["target"])
        if accepted % self.interval != 0 and accepted != target:
            return
        elapsed = float(snapshot["elapsed"])
        rate = accepted / elapsed if elapsed > 0 else 0.0
        remaining = max(0, target - accepted)
        eta = remaining / rate if rate > 0 else 0.0
        avg_steps = float(snapshot["steps_accepted_total"]) / accepted if accepted else 0.0
        avg_crossings = float(snapshot["net_crossings_accepted_total"]) / accepted if accepted else 0.0
        mode = snapshot["mode"]
        if mode == "fixed_random":
            label = f"fixed_random: {snapshot['display_name']}"
        else:
            label = "random_vs_random"
        print(
            f"[{label}] accepted={accepted}/{target} attempts={snapshot['attempts']} "
            f"discarded={snapshot['discarded']}"
        )
        print(
            f"elapsed={format_duration(elapsed)} rate={rate:.2f} accepted/s "
            f"eta={format_duration(eta)}"
        )
        print(f"avg_steps={avg_steps:.1f} avg_crossings={avg_crossings:.1f}", flush=True)


def load_temp_result(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with path.open("rb") as f:
        rallies = pkl.load(f)
    _csv_path, json_path = collect_data.metadata_paths(str(path))
    metadata = json.loads(json_path.read_text())
    return rallies, metadata


def collect_segment(
    *,
    target: int,
    temp_path: Path,
    seed: int,
    progress: ProgressPrinter,
    verbose_returns: bool,
    max_steps: int,
    reset_mode: str,
    skill_profile: str,
    gantry_speed_scale: float,
    max_attempts_per_setting: int | None,
    mode: str,
    fixed_player: int = 1,
    fixed_skill: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with suppress_return_lines(enabled=not verbose_returns):
        collect_data.collect(
            target_rallies=target,
            output_path=str(temp_path),
            max_steps_per_episode=max_steps,
            max_attempts_per_pair=max_attempts_per_setting,
            progress_every=0,
            mode=mode,
            fixed_player=fixed_player,
            fixed_skill=fixed_skill,
            reset_mode=reset_mode,
            skill_profile=skill_profile,
            gantry_speed_scale=gantry_speed_scale,
            seed=seed,
            quiet=True,
            accepted_progress_callback=progress,
        )
    return load_temp_result(temp_path)


def validate_combined(
    rallies: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    random_target: int,
    fixed_target: int,
    fixed_targets: dict[str, int],
) -> None:
    expected_total = random_target + fixed_target
    if len(rallies) != expected_total:
        raise RuntimeError(f"accepted rally mismatch: got {len(rallies)}, expected {expected_total}")
    for index, rally in enumerate(rallies):
        if len(rally.get("states", [])) != len(rally.get("skill_pairs", [])):
            raise RuntimeError(f"rally {index} has misaligned states and skill_pairs")
        if rally.get("winner") not in (1, 2):
            raise RuntimeError(f"rally {index} has invalid winner {rally.get('winner')!r}")
    accepted_attempts = [row for row in attempts if row.get("accepted")]
    mode_counts = Counter(row["mode"] for row in accepted_attempts)
    if mode_counts["random"] != random_target:
        raise RuntimeError(f"random accepted mismatch: got {mode_counts['random']}, expected {random_target}")
    if mode_counts["fixed_random"] != fixed_target:
        raise RuntimeError(f"fixed_random accepted mismatch: got {mode_counts['fixed_random']}, expected {fixed_target}")
    fixed_counts = Counter(row["setting"] for row in accepted_attempts if row["mode"] == "fixed_random")
    missing = [setting for setting, target in fixed_targets.items() if target > 0 and fixed_counts[setting] != target]
    if missing:
        raise RuntimeError(f"fixed-random setting count mismatch: {missing}")


def write_combined_metadata(output_path: str, summary: dict[str, Any], attempts: list[dict[str, Any]]) -> None:
    csv_path, json_path = output_metadata_paths(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in attempts for key in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(attempts)
    json_path.write_text(json.dumps({"summary": summary, "attempts": attempts}, indent=2))


def final_summary(
    rallies: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    random_target: int,
    fixed_target: int,
    fixed_targets: dict[str, int],
    elapsed: float,
) -> dict[str, Any]:
    accepted_attempts = [row for row in attempts if row.get("accepted")]
    crossings = [len(rally["states"]) for rally in rallies]
    active_counts = collect_data.active_decision_state_pair_counts(rallies)
    fixed_counts = Counter(row["setting"] for row in accepted_attempts if row["mode"] == "fixed_random")
    bucket = collect_data.crossing_bucket_percentages(crossings)
    summary = {
        "random_accepted": random_target,
        "fixed_random_accepted": fixed_target,
        "total_attempts": len(attempts),
        "total_discarded_truncated": len(attempts) - len(accepted_attempts),
        "elapsed_seconds": elapsed,
        "accepted_per_second": len(rallies) / elapsed if elapsed > 0 else 0.0,
        "avg_net_crossings": statistics.mean(crossings) if crossings else 0.0,
        "median_net_crossings": statistics.median(crossings) if crossings else 0.0,
        **bucket,
        "fixed_random_setting_targets": fixed_targets,
        "fixed_random_setting_counts": dict(fixed_counts),
        "active_skill_pair_decision_state_counts": {f"{s1} vs {s2}": count for (s1, s2), count in sorted(active_counts.items())},
    }
    print("\nFinal mixed dataset summary:")
    print(f"  random-vs-random accepted: {summary['random_accepted']}")
    print(f"  fixed-vs-random accepted: {summary['fixed_random_accepted']}")
    print(f"  total attempts: {summary['total_attempts']}")
    print(f"  discarded/truncated attempts: {summary['total_discarded_truncated']}")
    print(f"  total elapsed: {format_duration(elapsed)}")
    print(f"  overall rate: {summary['accepted_per_second']:.2f} accepted/s")
    print(f"  avg net crossings: {summary['avg_net_crossings']:.1f}")
    print(f"  median net crossings: {summary['median_net_crossings']:.1f}")
    print(f"  0 crossings: {summary['crossings_0_pct']:.1f}%")
    print(f"  1 crossing: {summary['crossings_1_pct']:.1f}%")
    print(f"  2+ crossings: {summary['crossings_2plus_pct']:.1f}%")
    print("  fixed-random accepted counts:")
    for setting in fixed_targets:
        print(f"    {setting}: {fixed_counts[setting]}")
    print("  recorded decision-state counts by active skill pair:")
    for label, count in summary["active_skill_pair_decision_state_counts"].items():
        print(f"    {label}: {count}")
    return summary


def run(args) -> list[dict[str, Any]]:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="mixed_v2_collect_"))
    start = time.monotonic()
    all_rallies: list[dict[str, Any]] = []
    all_attempts: list[dict[str, Any]] = []
    fixed_plan = allocate_fixed_random(args.fixed_random_rallies)
    fixed_targets = {setting["setting"]: int(setting["target"]) for setting in fixed_plan}
    progress = ProgressPrinter(args.progress_every)
    success = False

    try:
        random_path = temp_root / "random_vs_random.pkl"
        random_rallies, random_metadata = collect_segment(
            target=args.random_rallies,
            temp_path=random_path,
            seed=derived_seed(args.seed, "random"),
            progress=progress,
            verbose_returns=args.verbose_returns,
            max_steps=args.max_steps,
            reset_mode=args.reset_mode,
            skill_profile=args.skill_profile,
            gantry_speed_scale=args.gantry_speed_scale,
            max_attempts_per_setting=args.max_attempts_per_setting,
            mode="random",
        )
        all_rallies.extend(random_rallies)
        all_attempts.extend(random_metadata["attempts"])

        for index, setting in enumerate(fixed_plan):
            if setting["target"] == 0:
                continue
            temp_path = temp_root / f"{setting['setting']}.pkl"
            rallies, metadata = collect_segment(
                target=setting["target"],
                temp_path=temp_path,
                seed=derived_seed(args.seed, "fixed_random", index),
                progress=progress,
                verbose_returns=args.verbose_returns,
                max_steps=args.max_steps,
                reset_mode=args.reset_mode,
                skill_profile=args.skill_profile,
                gantry_speed_scale=args.gantry_speed_scale,
                max_attempts_per_setting=args.max_attempts_per_setting,
                mode="fixed_random",
                fixed_player=setting["fixed_player"],
                fixed_skill=setting["fixed_skill"],
            )
            all_rallies.extend(rallies)
            all_attempts.extend(metadata["attempts"])

        for global_id, row in enumerate(all_attempts):
            row["global_attempt_id"] = global_id

        validate_combined(all_rallies, all_attempts, args.random_rallies, args.fixed_random_rallies, fixed_targets)
        elapsed = time.monotonic() - start
        summary = final_summary(all_rallies, all_attempts, args.random_rallies, args.fixed_random_rallies, fixed_targets, elapsed)

        with output.open("wb") as f:
            pkl.dump(all_rallies, f)
        write_combined_metadata(str(output), summary, all_attempts)
        print(f"\nSaved combined pickle: {output}")
        print(f"Saved metadata: {output_metadata_paths(str(output))[0]} and {output_metadata_paths(str(output))[1]}")
        success = True
        return all_rallies
    finally:
        if args.keep_intermediate or not success:
            print(f"Intermediate files kept in: {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect a mixed random/fixed-random v2 dataset.")
    parser.add_argument("--random-rallies", type=int, default=7000)
    parser.add_argument("--fixed-random-rallies", type=int, default=3000)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--reset-mode", choices=["clean", "ready", "carryover"], default="ready")
    parser.add_argument("--skill-profile", choices=SKILL_PROFILE_NAMES, default="aggressive")
    parser.add_argument("--gantry-speed-scale", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--max-attempts-per-setting", type=int, default=None)
    parser.add_argument("--verbose-returns", action="store_true")
    parser.add_argument("--keep-intermediate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
