#!/usr/bin/env python3
"""Summarize fixed-P1 vs random-opponent landing diagnostics."""

from __future__ import annotations

import argparse
import csv
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


FIELDS = [
    "fixed_p1_skill",
    "episodes",
    "p1_win_rate",
    "truncation_rate",
    "median_rally_length",
    "mean_net_crossings",
    "landing_count",
    "expected_x",
    "expected_y",
    "actual_x_mean",
    "actual_y_mean",
    "actual_x_std",
    "actual_y_std",
    "landing_error_mean",
    "landing_error_median",
    "landing_error_std",
]


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def f(value: Any, default: float | None = None) -> float | None:
    if value in ("", None):
        return default
    return float(value)


def summarize(rallies: list[dict[str, Any]], contacts: list[dict[str, Any]], skill_profile: str) -> list[dict[str, Any]]:
    p1_rallies = [r for r in rallies if r.get("mode") == "fixed_vs_random" and r.get("fixed_player") == "1"]
    by_skill = defaultdict(list)
    by_skill_contacts = defaultdict(list)
    for row in p1_rallies:
        by_skill[row["fixed_skill"]].append(row)
    for row in contacts:
        if row.get("mode") == "fixed_vs_random" and row.get("fixed_player") == "1" and row.get("player") == "1":
            by_skill_contacts[row["fixed_skill"]].append(row)

    rows = []
    for skill in SKILL_NAMES:
        rs = by_skill.get(skill, [])
        cs = by_skill_contacts.get(skill, [])
        n = len(rs)
        lengths = [int(float(r["rally_length"])) for r in rs]
        decisions = [int(float(r["num_decisions"])) for r in rs]
        xs = [f(c.get("x_land")) for c in cs if f(c.get("x_land")) is not None]
        ys = [f(c.get("y_land")) for c in cs if f(c.get("y_land")) is not None]
        errs = [f(c.get("error_dist")) for c in cs if f(c.get("error_dist")) is not None]
        expected_x, expected_y = world_target_xy(1, skill, profile=skill_profile)
        rows.append({
            "fixed_p1_skill": skill,
            "episodes": n,
            "p1_win_rate": sum(r["winner"] == "player1" for r in rs) / n if n else "",
            "truncation_rate": sum(str(r["truncated"]).lower() == "true" for r in rs) / n if n else "",
            "median_rally_length": statistics.median(lengths) if lengths else "",
            "mean_net_crossings": statistics.mean(decisions) if decisions else "",
            "landing_count": len(errs),
            "expected_x": expected_x,
            "expected_y": expected_y,
            "actual_x_mean": statistics.mean(xs) if xs else "",
            "actual_y_mean": statistics.mean(ys) if ys else "",
            "actual_x_std": statistics.pstdev(xs) if len(xs) > 1 else 0.0 if xs else "",
            "actual_y_std": statistics.pstdev(ys) if len(ys) > 1 else 0.0 if ys else "",
            "landing_error_mean": statistics.mean(errs) if errs else "",
            "landing_error_median": statistics.median(errs) if errs else "",
            "landing_error_std": statistics.pstdev(errs) if len(errs) > 1 else 0.0 if errs else "",
        })
    return rows


def plot(path: Path, contacts: list[dict[str, Any]], skill_profile: str, vectors: bool) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    actual_all = []
    expected_all = []
    for skill in SKILL_NAMES:
        pts = np.array(
            [
                [float(c["x_land"]), float(c["y_land"])]
                for c in contacts
                if c.get("mode") == "fixed_vs_random"
                and c.get("fixed_player") == "1"
                and c.get("fixed_skill") == skill
                and c.get("player") == "1"
                and c.get("x_land") not in ("", None)
                and c.get("y_land") not in ("", None)
            ],
            dtype=float,
        )
        ex, ey = world_target_xy(1, skill, profile=skill_profile)
        ax.scatter([ex], [ey], marker="x", s=100, color="black")
        ax.text(ex + 0.01, ey + 0.01, f"{skill} target")
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=22, alpha=0.55, label=skill)
            actual_all.extend(pts.tolist())
            expected_all.extend([[ex, ey]] * len(pts))
    if vectors and actual_all:
        actual = np.asarray(actual_all)
        expected = np.asarray(expected_all)
        idx = np.linspace(0, len(actual) - 1, min(len(actual), 80), dtype=int)
        ax.quiver(
            expected[idx, 0],
            expected[idx, 1],
            actual[idx, 0] - expected[idx, 0],
            actual[idx, 1] - expected[idx, 1],
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
    ax.set_title("Fixed P1 vs random P2 landings")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(path: Path, summary: list[dict[str, Any]], plots: list[Path]) -> None:
    lines = [
        "# Fixed P1 Vs Random Landing Diagnostic",
        "",
        "| fixed P1 skill | episodes | P1 win | trunc | median steps | mean crossings | landings | expected | actual mean | mean error | median error |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['fixed_p1_skill']} | {row['episodes']} | {float(row['p1_win_rate'] or 0):.3f} | "
            f"{float(row['truncation_rate'] or 0):.3f} | {row['median_rally_length']} | "
            f"{float(row['mean_net_crossings'] or 0):.3f} | {row['landing_count']} | "
            f"({float(row['expected_x']):.3f}, {float(row['expected_y']):.3f}) | "
            f"({float(row['actual_x_mean'] or 0):.3f}, {float(row['actual_y_mean'] or 0):.3f}) | "
            f"{float(row['landing_error_mean'] or 0):.3f} | {float(row['landing_error_median'] or 0):.3f} |"
        )
    lines += ["", "## Outputs", "- `fixed_vs_random_p1_landing_summary.csv`", *[f"- `{p.relative_to(path.parent)}`" for p in plots]]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skill-profile", choices=SKILL_PROFILE_NAMES, default="aggressive")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rallies = read_csv(input_dir / "rallies.csv")
    contacts = read_csv(input_dir / "contacts.csv")
    summary = summarize(rallies, contacts, args.skill_profile)
    write_csv(out / "fixed_vs_random_p1_landing_summary.csv", summary)
    plot_dir = out / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plots = [plot_dir / "actual_landing_locations.png", plot_dir / "landing_error_vectors.png"]
    plot(plots[0], contacts, args.skill_profile, vectors=False)
    plot(plots[1], contacts, args.skill_profile, vectors=True)
    write_report(out / "fixed_vs_random_p1_landing_report.md", summary, plots)
    print(f"Wrote fixed-vs-random P1 landing analysis to {out}")


if __name__ == "__main__":
    main()
