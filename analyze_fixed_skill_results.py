#!/usr/bin/env python3
"""Combine and analyze fixed-player-2 skill diagnostic outputs.

This script is offline-only: it reads existing diagnostic CSVs and does not
import or initialize MuJoCo.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nash_skills.skills import SKILL_NAMES, get_skill, skill_index

EPISODE_FIELDS = [
    "source_dir",
    "source_row",
    "key",
    "seed",
    "episode_index",
    "skill_profile",
    "player1_skill",
    "player2_skill",
    "winner",
    "termination_reason",
    "reached_step_limit",
    "physics_steps",
    "decision_state_count",
    "net_crossings",
    "successful_returns",
    "return_bucket",
    "raw_obs_ids",
    "state_ids",
    "player1_target_xy",
    "player2_target_xy",
    "validation_ok",
    "validation_errors",
    "combined_key",
]

SUMMARY_FIELDS = [
    "player1_skill",
    "player2_skill",
    "episode_count",
    "expected_episode_count",
    "missing_episode_count",
    "duplicate_episode_count",
    "player1_wins",
    "player2_wins",
    "truncated",
    "truncation_rate",
    "truncation_ci_low",
    "truncation_ci_high",
    "player1_win_rate_all",
    "player1_win_rate_all_ci_low",
    "player1_win_rate_all_ci_high",
    "completed_count",
    "completed_player1_win_rate",
    "completed_player1_win_rate_ci_low",
    "completed_player1_win_rate_ci_high",
    "physics_steps_mean",
    "physics_steps_median",
    "physics_steps_std",
    "physics_steps_max",
    "decision_state_count_mean",
    "decision_state_count_median",
    "net_crossings_mean",
    "successful_returns_mean",
    "zero_return_rate",
    "one_return_rate",
    "two_plus_return_rate",
    "suspicion_score",
    "flags",
]

SUSPICIOUS_FIELDS = [
    "rank",
    "player1_skill",
    "player2_skill",
    "episode_count",
    "completed_count",
    "truncation_rate",
    "completed_player1_win_rate",
    "physics_steps_mean",
    "physics_steps_median",
    "physics_steps_max",
    "decision_state_count_mean",
    "net_crossings_mean",
    "successful_returns_mean",
    "zero_return_rate",
    "one_return_rate",
    "two_plus_return_rate",
    "suspicion_score",
    "flags",
]


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def parse_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    return float(value)


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def target_xy(skill: str) -> list[float]:
    side, x_target = get_skill(skill)
    return [float(x_target), float(side * 0.38)]


def combined_key(row: dict[str, Any]) -> str:
    return f"{row['seed']}:{row['player1_skill']}:{row['player2_skill']}:{row['episode_index']}"


def read_episode_csv(input_dir: Path) -> list[dict[str, Any]]:
    path = input_dir / "episodes.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing episodes.csv in {input_dir}")

    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"seed", "episode_index", "player1_skill", "player2_skill", "winner", "physics_steps"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for idx, row in enumerate(reader, start=2):
            item = {field: row.get(field, "") for field in EPISODE_FIELDS if field not in {"source_dir", "source_row", "combined_key"}}
            item["source_dir"] = str(input_dir)
            item["source_row"] = idx
            item["seed"] = parse_int(item["seed"])
            item["episode_index"] = parse_int(item["episode_index"])
            item["physics_steps"] = parse_int(item["physics_steps"])
            item["decision_state_count"] = parse_int(item.get("decision_state_count", 0))
            item["net_crossings"] = parse_int(item.get("net_crossings", item["decision_state_count"]))
            item["successful_returns"] = parse_int(item.get("successful_returns", 0))
            item["return_bucket"] = item.get("return_bucket") or (
                "0" if item["successful_returns"] == 0 else "1" if item["successful_returns"] == 1 else "2+"
            )
            item["reached_step_limit"] = parse_bool(item.get("reached_step_limit", False))
            item["validation_ok"] = parse_bool(item.get("validation_ok", True))
            item["combined_key"] = combined_key(item)
            rows.append(item)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, 6)
    return value


def sorted_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: (skill_index(r["player2_skill"]), skill_index(r["player1_skill"]), r["seed"], r["episode_index"]))


def expected_count_from_rows(rows: list[dict[str, Any]], explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    counts = Counter((r["player1_skill"], r["player2_skill"]) for r in rows)
    if not counts:
        return 0
    count_counts = Counter(counts.values())
    return count_counts.most_common(1)[0][0]


def detect_issues(rows: list[dict[str, Any]], expected_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    key_counts = Counter(r["combined_key"] for r in rows)
    duplicates = [r for r in rows if key_counts[r["combined_key"]] > 1]

    pair_groups = defaultdict(list)
    for row in rows:
        pair_groups[(row["player1_skill"], row["player2_skill"])].append(row)

    missing_or_incomplete = []
    for p1 in SKILL_NAMES:
        for p2 in SKILL_NAMES:
            pair_rows = pair_groups.get((p1, p2), [])
            missing = max(0, expected_count - len(pair_rows))
            if not pair_rows or missing:
                missing_or_incomplete.append({
                    "player1_skill": p1,
                    "player2_skill": p2,
                    "episode_count": len(pair_rows),
                    "expected_episode_count": expected_count,
                    "missing_episode_count": missing,
                })

    invalid = []
    for row in rows:
        reasons = []
        if row["player1_skill"] not in SKILL_NAMES:
            reasons.append("invalid_player1_skill")
        if row["player2_skill"] not in SKILL_NAMES:
            reasons.append("invalid_player2_skill")
        if row["winner"] not in {"player1", "player2", "truncated"}:
            reasons.append("invalid_winner")
        if row["physics_steps"] < 0:
            reasons.append("negative_physics_steps")
        if not row["validation_ok"]:
            reasons.append("validation_failed")
        if reasons:
            item = dict(row)
            item["issue_reasons"] = ";".join(reasons)
            invalid.append(item)

    return duplicates, missing_or_incomplete, invalid


def summarize(rows: list[dict[str, Any]], expected_count: int, duplicate_keys: set[str]) -> list[dict[str, Any]]:
    pair_groups = defaultdict(list)
    for row in rows:
        pair_groups[(row["player1_skill"], row["player2_skill"])].append(row)

    completed_rates = []
    medians = []
    for pair_rows in pair_groups.values():
        completed = sum(r["winner"] in {"player1", "player2"} for r in pair_rows)
        if pair_rows:
            completed_rates.append(completed / len(pair_rows))
            medians.extend(r["physics_steps"] for r in pair_rows)
    global_step_median = statistics.median(medians) if medians else 0

    summary = []
    for p1 in SKILL_NAMES:
        for p2 in SKILL_NAMES:
            rs = pair_groups.get((p1, p2), [])
            n = len(rs)
            p1_wins = sum(r["winner"] == "player1" for r in rs)
            p2_wins = sum(r["winner"] == "player2" for r in rs)
            truncated = sum(r["winner"] == "truncated" or r["reached_step_limit"] for r in rs)
            completed = p1_wins + p2_wins
            steps = [r["physics_steps"] for r in rs]
            decisions = [r["decision_state_count"] for r in rs if r.get("decision_state_count") is not None]
            crossings = [r["net_crossings"] for r in rs if r.get("net_crossings") is not None]
            returns = [r["successful_returns"] for r in rs if r.get("successful_returns") is not None]
            dup_count = sum(1 for r in rs if r["combined_key"] in duplicate_keys)

            trunc_ci = wilson_interval(truncated, n)
            p1_ci = wilson_interval(p1_wins, n)
            completed_ci = wilson_interval(p1_wins, completed)
            trunc_rate = truncated / n if n else None
            completed_win = p1_wins / completed if completed else None
            step_mean = statistics.mean(steps) if steps else None
            step_median = statistics.median(steps) if steps else None

            flags = []
            score = 0
            missing = max(0, expected_count - n)
            if n == 0:
                flags.append("missing_pair")
                score += 100
            if missing:
                flags.append("incomplete_pair")
                score += 20 + missing / max(expected_count, 1)
            if dup_count:
                flags.append("duplicate_episodes")
                score += 20
            if completed == 0 and n:
                flags.append("zero_completed_rallies")
                score += 50
            if trunc_rate is not None and trunc_rate >= 0.75:
                flags.append("high_truncation")
                score += 30 + trunc_rate * 10
            elif trunc_rate is not None and trunc_rate >= 0.5:
                flags.append("moderate_truncation")
                score += 15 + trunc_rate * 5
            if step_median is not None and global_step_median and step_median > global_step_median * 1.25:
                flags.append("long_duration")
                score += 10
            if step_mean is not None and step_mean >= 0.9 * max((max(steps) if steps else 0), 1) and trunc_rate and trunc_rate > 0.5:
                flags.append("near_step_cap_mean")
                score += 10

            summary.append({
                "player1_skill": p1,
                "player2_skill": p2,
                "episode_count": n,
                "expected_episode_count": expected_count,
                "missing_episode_count": missing,
                "duplicate_episode_count": dup_count,
                "player1_wins": p1_wins,
                "player2_wins": p2_wins,
                "truncated": truncated,
                "truncation_rate": trunc_rate,
                "truncation_ci_low": trunc_ci[0],
                "truncation_ci_high": trunc_ci[1],
                "player1_win_rate_all": p1_wins / n if n else None,
                "player1_win_rate_all_ci_low": p1_ci[0],
                "player1_win_rate_all_ci_high": p1_ci[1],
                "completed_count": completed,
                "completed_player1_win_rate": completed_win,
                "completed_player1_win_rate_ci_low": completed_ci[0],
                "completed_player1_win_rate_ci_high": completed_ci[1],
                "physics_steps_mean": step_mean,
                "physics_steps_median": step_median,
                "physics_steps_std": statistics.pstdev(steps) if len(steps) > 1 else 0.0,
                "physics_steps_max": max(steps) if steps else None,
                "decision_state_count_mean": statistics.mean(decisions) if decisions else None,
                "decision_state_count_median": statistics.median(decisions) if decisions else None,
                "net_crossings_mean": statistics.mean(crossings) if crossings else None,
                "successful_returns_mean": statistics.mean(returns) if returns else None,
                "zero_return_rate": sum(v == 0 for v in returns) / n if n else None,
                "one_return_rate": sum(v == 1 for v in returns) / n if n else None,
                "two_plus_return_rate": sum(v >= 2 for v in returns) / n if n else None,
                "suspicion_score": score,
                "flags": ";".join(flags),
            })
    return summary


def matrix_from_summary(summary: list[dict[str, Any]], field: str) -> np.ndarray:
    mat = np.full((len(SKILL_NAMES), len(SKILL_NAMES)), np.nan)
    for row in summary:
        i = skill_index(row["player1_skill"])
        j = skill_index(row["player2_skill"])
        value = row.get(field)
        if value is not None and value != "":
            mat[i, j] = float(value)
    return mat


def save_heatmap(path: Path, summary: list[dict[str, Any]], field: str, title: str, cmap: str, vmin=None, vmax=None) -> None:
    data = matrix_from_summary(summary, field)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(SKILL_NAMES)), SKILL_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(len(SKILL_NAMES)), SKILL_NAMES)
    ax.set_xlabel("player 2 fixed skill")
    ax.set_ylabel("player 1 skill")
    ax.set_title(title)
    for i in range(len(SKILL_NAMES)):
        for j in range(len(SKILL_NAMES)):
            value = data[i, j]
            label = "NA" if np.isnan(value) else f"{value:.2f}"
            ax.text(j, i, label, ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def pair_label(row: dict[str, Any]) -> str:
    return f"{row['player1_skill']} vs {row['player2_skill']}"


def plot_outcomes(path: Path, summary: list[dict[str, Any]]) -> None:
    labels = [pair_label(r) for r in summary]
    x = np.arange(len(summary))
    p1 = np.array([r["player1_wins"] for r in summary])
    p2 = np.array([r["player2_wins"] for r in summary])
    tr = np.array([r["truncated"] for r in summary])
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x, p1, label="player1 win")
    ax.bar(x, p2, bottom=p1, label="player2 win")
    ax.bar(x, tr, bottom=p1 + p2, label="truncated")
    ax.set_xticks(x, labels, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("episodes")
    ax.set_title("Outcome comparison by skill pair")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_duration_distribution(path: Path, rows: list[dict[str, Any]]) -> None:
    groups = []
    labels = []
    for p2 in SKILL_NAMES:
        for p1 in SKILL_NAMES:
            values = [r["physics_steps"] for r in rows if r["player1_skill"] == p1 and r["player2_skill"] == p2]
            if values:
                groups.append(values)
                labels.append(f"{p1}\nvs {p2}")
    fig, ax = plt.subplots(figsize=(14, 6))
    if groups:
        try:
            ax.boxplot(groups, tick_labels=labels, showfliers=False)
        except TypeError:
            ax.boxplot(groups, labels=labels, showfliers=False)
    step_caps = sorted({r["physics_steps"] for r in rows if r["reached_step_limit"]})
    if step_caps:
        ax.axhline(max(step_caps), color="red", linestyle="--", label=f"observed step cap {max(step_caps)}")
        ax.legend()
    ax.set_ylabel("physics steps")
    ax.set_title("Rally-duration distribution by skill pair")
    ax.tick_params(axis="x", labelrotation=75, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_fixed_skill_comparison(path: Path, summary: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    x = np.arange(len(SKILL_NAMES))
    for ax, field, title in [
        (axes[0], "truncation_rate", "Truncation by fixed player 2 skill"),
        (axes[1], "completed_player1_win_rate", "Completed win rate by fixed player 2 skill"),
    ]:
        for p1 in SKILL_NAMES:
            vals = []
            for p2 in SKILL_NAMES:
                row = next(r for r in summary if r["player1_skill"] == p1 and r["player2_skill"] == p2)
                vals.append(0.0 if row[field] is None else row[field])
            ax.plot(x, vals, marker="o", label=p1)
        ax.set_xticks(x, SKILL_NAMES, rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(title)
    axes[0].set_ylabel("rate")
    axes[1].legend(title="player 1 skill", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_target_coordinates(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for skill in SKILL_NAMES:
        x, y = target_xy(skill)
        ax.scatter([x], [y], s=70)
        ax.text(x + 0.01, y + 0.01, skill)
    ax.set_xlabel("target x")
    ax.set_ylabel("target y")
    ax.set_title("Canonical skill target coordinates")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def generate_plots(out: Path, rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> list[str]:
    plot_dir = out / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plots = []
    heatmaps = [
        ("heatmap_truncation_rate.png", "truncation_rate", "Truncation rate", "Reds", 0, 1),
        ("heatmap_player1_win_rate.png", "player1_win_rate_all", "Player 1 win rate, all rallies", "Blues", 0, 1),
        ("heatmap_completed_win_rate.png", "completed_player1_win_rate", "Player 1 win rate, completed only", "Blues", 0, 1),
        ("heatmap_mean_physics_steps.png", "physics_steps_mean", "Mean physics steps", "viridis", None, None),
        ("heatmap_median_physics_steps.png", "physics_steps_median", "Median physics steps", "viridis", None, None),
        ("heatmap_mean_decision_states.png", "decision_state_count_mean", "Mean decision-state count", "magma", None, None),
    ]
    for filename, field, title, cmap, vmin, vmax in heatmaps:
        path = plot_dir / filename
        save_heatmap(path, summary, field, title, cmap, vmin, vmax)
        plots.append(str(path))

    for filename, func in [
        ("outcome_comparison.png", lambda p: plot_outcomes(p, summary)),
        ("rally_duration_distribution.png", lambda p: plot_duration_distribution(p, rows)),
        ("fixed_skill_rate_comparison.png", lambda p: plot_fixed_skill_comparison(p, summary)),
        ("target_coordinates.png", plot_target_coordinates),
    ]:
        path = plot_dir / filename
        func(path)
        plots.append(str(path))
    return plots


def write_report(
    path: Path,
    input_dirs: list[Path],
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    duplicates: list[dict[str, Any]],
    missing_or_incomplete: list[dict[str, Any]],
    invalid: list[dict[str, Any]],
    suspicious: list[dict[str, Any]],
    plots: list[str],
) -> None:
    high_priority = [
        r for r in suspicious
        if "high_truncation" in r["flags"] or "zero_completed_rallies" in r["flags"] or "long_duration" in r["flags"]
    ][:10]
    lines = [
        "# Combined Fixed-Skill Analysis",
        "",
        "## Inputs",
        *[f"- `{p}`" for p in input_dirs],
        "",
        "## Data Checks",
        f"- Combined episode rows: {len(rows)}",
        f"- Skill pairs present: {sum(1 for r in summary if r['episode_count'] > 0)}/25",
        f"- Duplicate episode rows: {len(duplicates)}",
        f"- Missing or incomplete pairs: {len(missing_or_incomplete)}",
        f"- Invalid or failed-validation rows: {len(invalid)}",
        "",
        "## Pairs Needing Deeper Trajectory Investigation",
    ]
    if high_priority:
        for row in high_priority:
            lines.append(
                f"- `{row['player1_skill']}` vs `{row['player2_skill']}`: "
                f"trunc={format_cell(row['truncation_rate'])}, "
                f"completed={row['completed_count']}, "
                f"median_steps={format_cell(row['physics_steps_median'])}, "
                f"flags={row['flags']}"
            )
    else:
        lines.append("- No high-priority suspicious pairs found by the configured thresholds.")

    lines += [
        "",
        "## Interpretation Notes",
        "- Truncated rallies are counted separately and are not treated as player 1 or player 2 losses.",
        "- `player1_win_rate_all` uses all episodes as the denominator.",
        "- `completed_player1_win_rate` uses only completed player1/player2 wins as the denominator.",
        "- High truncation or zero completed rallies can create selection bias if truncated rallies are discarded before model training.",
        "",
        "## Outputs",
        "- `combined_episodes.csv`",
        "- `combined_summary.csv`",
        "- `suspicious_pairs.csv`",
        "- `missing_or_incomplete_pairs.csv`",
        "- `duplicate_episodes.csv`",
        "- `invalid_rows.csv`",
        "- `metadata.json`",
        *[f"- `{Path(p).relative_to(path.parent)}`" for p in plots],
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze completed fixed-player-2 skill diagnostic folders.")
    parser.add_argument("--input-dirs", nargs="+", required=True, help="Folders containing episodes.csv from diagnostic_fixed_skill.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-episodes-per-pair", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dirs = [Path(p) for p in args.input_dirs]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for input_dir in input_dirs:
        rows.extend(read_episode_csv(input_dir))
    rows = sorted_rows(rows)

    expected = expected_count_from_rows(rows, args.expected_episodes_per_pair)
    duplicates, missing_or_incomplete, invalid = detect_issues(rows, expected)
    duplicate_keys = {r["combined_key"] for r in duplicates}
    summary = summarize(rows, expected, duplicate_keys)
    suspicious = sorted(
        [dict(r, rank=idx + 1) for idx, r in enumerate(sorted(summary, key=lambda x: x["suspicion_score"], reverse=True))],
        key=lambda r: r["rank"],
    )

    serializable_rows = [{k: format_cell(v) for k, v in row.items()} for row in rows]
    serializable_summary = [{k: format_cell(v) for k, v in row.items()} for row in summary]
    serializable_suspicious = [{k: format_cell(v) for k, v in row.items()} for row in suspicious]

    write_csv(out / "combined_episodes.csv", serializable_rows, EPISODE_FIELDS)
    write_csv(out / "combined_summary.csv", serializable_summary, SUMMARY_FIELDS)
    write_csv(out / "suspicious_pairs.csv", serializable_suspicious, SUSPICIOUS_FIELDS)
    write_csv(out / "missing_or_incomplete_pairs.csv", missing_or_incomplete, [
        "player1_skill", "player2_skill", "episode_count", "expected_episode_count", "missing_episode_count"
    ])
    write_csv(out / "duplicate_episodes.csv", duplicates, EPISODE_FIELDS)
    write_csv(out / "invalid_rows.csv", invalid, [*EPISODE_FIELDS, "issue_reasons"])

    plots = generate_plots(out, rows, summary)
    metadata = {
        "input_dirs": [str(p) for p in input_dirs],
        "output_dir": str(out),
        "episode_rows": len(rows),
        "expected_episodes_per_pair": expected,
        "pairs_present": sum(1 for r in summary if r["episode_count"] > 0),
        "duplicates": len(duplicates),
        "missing_or_incomplete_pairs": len(missing_or_incomplete),
        "invalid_rows": len(invalid),
        "skill_order": SKILL_NAMES,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    write_report(
        out / "combined_analysis_report.md",
        input_dirs,
        rows,
        summary,
        duplicates,
        missing_or_incomplete,
        invalid,
        suspicious,
        plots,
    )

    print(f"Wrote combined analysis to {out}")
    print(f"Rows={len(rows)} pairs_present={metadata['pairs_present']}/25 expected_per_pair={expected}")
    print(f"duplicates={len(duplicates)} missing_or_incomplete={len(missing_or_incomplete)} invalid={len(invalid)}")


if __name__ == "__main__":
    main()
