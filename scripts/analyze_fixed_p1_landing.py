#!/usr/bin/env python3
"""Aggregate fixed-P1 skill diagnostics with landing-error analysis."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nash_skills.skills import SKILL_NAMES, SKILL_PROFILE_NAMES, world_target_xy


SUMMARY_FIELDS = [
    "fixed_p1_skill",
    "episodes",
    "p1_win_rate",
    "truncation_rate",
    "median_rally_length",
    "mean_returns",
    "mean_net_crossings",
    "landing_count",
    "expected_x",
    "expected_y",
    "actual_x_mean",
    "actual_y_mean",
    "actual_x_std",
    "actual_y_std",
    "actual_x_min",
    "actual_x_max",
    "actual_y_min",
    "actual_y_max",
    "landing_error_mean",
    "landing_error_median",
    "landing_error_std",
]


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any, default: float | None = None) -> float | None:
    if value in ("", None):
        return default
    return float(value)


def as_int(value: Any, default: int = 0) -> int:
    if value in ("", None):
        return default
    return int(float(value))


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def summarize(episodes: list[dict[str, Any]], contacts: list[dict[str, Any]], skill_profile: str) -> list[dict[str, Any]]:
    episodes_by_skill = defaultdict(list)
    contacts_by_skill = defaultdict(list)
    for row in episodes:
        episodes_by_skill[row["player1_skill"]].append(row)
    for row in contacts:
        if row.get("player") == "player1":
            contacts_by_skill[row["player1_skill"]].append(row)

    rows = []
    for skill in SKILL_NAMES:
        eps = episodes_by_skill.get(skill, [])
        cts = contacts_by_skill.get(skill, [])
        n = len(eps)
        wins = sum(row.get("winner") == "player1" for row in eps)
        trunc = sum(row.get("winner") == "truncated" or as_bool(row.get("reached_step_limit")) for row in eps)
        steps = [as_int(row.get("physics_steps")) for row in eps]
        returns = [as_int(row.get("successful_returns")) for row in eps]
        crossings = [as_int(row.get("net_crossings", row.get("decision_state_count"))) for row in eps]
        xs = [as_float(row.get("x_land")) for row in cts if as_float(row.get("x_land")) is not None]
        ys = [as_float(row.get("y_land")) for row in cts if as_float(row.get("y_land")) is not None]
        errs = [as_float(row.get("error_dist")) for row in cts if as_float(row.get("error_dist")) is not None]
        expected_x, expected_y = world_target_xy(1, skill, profile=skill_profile)
        rows.append({
            "fixed_p1_skill": skill,
            "episodes": n,
            "p1_win_rate": wins / n if n else "",
            "truncation_rate": trunc / n if n else "",
            "median_rally_length": statistics.median(steps) if steps else "",
            "mean_returns": statistics.mean(returns) if returns else "",
            "mean_net_crossings": statistics.mean(crossings) if crossings else "",
            "landing_count": len(errs),
            "expected_x": expected_x,
            "expected_y": expected_y,
            "actual_x_mean": statistics.mean(xs) if xs else "",
            "actual_y_mean": statistics.mean(ys) if ys else "",
            "actual_x_std": statistics.pstdev(xs) if len(xs) > 1 else 0.0 if xs else "",
            "actual_y_std": statistics.pstdev(ys) if len(ys) > 1 else 0.0 if ys else "",
            "actual_x_min": min(xs) if xs else "",
            "actual_x_max": max(xs) if xs else "",
            "actual_y_min": min(ys) if ys else "",
            "actual_y_max": max(ys) if ys else "",
            "landing_error_mean": statistics.mean(errs) if errs else "",
            "landing_error_median": statistics.median(errs) if errs else "",
            "landing_error_std": statistics.pstdev(errs) if len(errs) > 1 else 0.0 if errs else "",
        })
    return rows


def plot_targets(path: Path, skill_profile: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for skill in SKILL_NAMES:
        x, y = world_target_xy(1, skill, profile=skill_profile)
        ax.scatter([x], [y], marker="x", s=90)
        ax.text(x + 0.01, y + 0.01, skill)
    ax.axvline(1.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("world x")
    ax.set_ylabel("world y")
    ax.set_title(f"Expected P1 skill targets ({skill_profile})")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_actual_landings(path: Path, contacts: list[dict[str, Any]], skill_profile: str, with_vectors: bool) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    all_expected = []
    all_actual = []
    for skill in SKILL_NAMES:
        pts = np.array(
            [
                [float(row["x_land"]), float(row["y_land"])]
                for row in contacts
                if row.get("player") == "player1"
                and row.get("player1_skill") == skill
                and row.get("x_land") not in ("", None)
                and row.get("y_land") not in ("", None)
            ],
            dtype=float,
        )
        exp_x, exp_y = world_target_xy(1, skill, profile=skill_profile)
        ax.scatter([exp_x], [exp_y], marker="x", s=100, color="black")
        ax.text(exp_x + 0.01, exp_y + 0.01, f"{skill} target")
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], alpha=0.45, s=18, label=skill)
            all_actual.extend(pts.tolist())
            all_expected.extend([[exp_x, exp_y]] * len(pts))
    if with_vectors and all_actual:
        actual = np.asarray(all_actual, dtype=float)
        expected = np.asarray(all_expected, dtype=float)
        sample_idx = np.linspace(0, len(actual) - 1, min(len(actual), 80), dtype=int)
        ax.quiver(
            expected[sample_idx, 0],
            expected[sample_idx, 1],
            actual[sample_idx, 0] - expected[sample_idx, 0],
            actual[sample_idx, 1] - expected[sample_idx, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.0025,
            alpha=0.25,
            color="gray",
        )
    ax.axvline(1.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("world landing x")
    ax.set_ylabel("world landing y")
    ax.set_title("P1 actual landing locations")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(path: Path, summary: list[dict[str, Any]], input_dirs: list[Path], plots: list[Path]) -> None:
    lines = [
        "# Fixed P1 Landing Diagnostic",
        "",
        "## Inputs",
        *[f"- `{p}`" for p in input_dirs],
        "",
        "## Summary",
        "| fixed P1 skill | episodes | P1 win | trunc | median steps | mean returns | mean crossings | landings | expected | actual mean | mean error | median error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['fixed_p1_skill']} | {row['episodes']} | {float(row['p1_win_rate'] or 0):.3f} | "
            f"{float(row['truncation_rate'] or 0):.3f} | {row['median_rally_length']} | "
            f"{float(row['mean_returns'] or 0):.3f} | {float(row['mean_net_crossings'] or 0):.3f} | "
            f"{row['landing_count']} | ({float(row['expected_x']):.3f}, {float(row['expected_y']):.3f}) | "
            f"({float(row['actual_x_mean'] or 0):.3f}, {float(row['actual_y_mean'] or 0):.3f}) | "
            f"{float(row['landing_error_mean'] or 0):.3f} | {float(row['landing_error_median'] or 0):.3f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "- Landing distribution uses every successful P1 return in rallies with that fixed P1 skill.",
        "- Landing error is Euclidean distance from the expected P1 world target for that skill.",
        "",
        "## Outputs",
        "- `fixed_p1_landing_summary.csv`",
        *[f"- `{p.relative_to(path.parent)}`" for p in plots],
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skill-profile", choices=SKILL_PROFILE_NAMES, default="aggressive")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dirs = [Path(p) for p in args.input_dirs]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    episodes = []
    contacts = []
    for input_dir in input_dirs:
        episodes.extend(read_csv(input_dir / "episodes.csv"))
        contacts.extend(read_csv(input_dir / "contacts.csv"))
    summary = summarize(episodes, contacts, args.skill_profile)
    write_csv(out / "fixed_p1_landing_summary.csv", summary, SUMMARY_FIELDS)
    plot_dir = out / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plots = [
        plot_dir / "expected_skill_targets.png",
        plot_dir / "actual_landing_locations.png",
        plot_dir / "landing_error_vectors.png",
    ]
    plot_targets(plots[0], args.skill_profile)
    plot_actual_landings(plots[1], contacts, args.skill_profile, with_vectors=False)
    plot_actual_landings(plots[2], contacts, args.skill_profile, with_vectors=True)
    (out / "metadata.json").write_text(json.dumps({
        "input_dirs": [str(p) for p in input_dirs],
        "output_dir": str(out),
        "skill_profile": args.skill_profile,
        "episodes": len(episodes),
        "contacts": len(contacts),
    }, indent=2))
    write_report(out / "fixed_p1_landing_report.md", summary, input_dirs, plots)
    print(f"Wrote fixed-P1 landing analysis to {out}")


if __name__ == "__main__":
    main()
