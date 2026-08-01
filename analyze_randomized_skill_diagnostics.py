#!/usr/bin/env python3
"""Analyze randomized skill diagnostic CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from nash_skills.skills import SKILL_NAMES


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, 6)
    return value


def normalize_rallies(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        item = dict(row)
        item["seed"] = parse_int(item.get("seed"))
        item["fixed_player"] = None if item.get("fixed_player") in (None, "") else parse_int(item["fixed_player"])
        item["truncated"] = parse_bool(item.get("truncated"))
        item["rally_length"] = parse_int(item.get("rally_length"))
        item["num_decisions"] = parse_int(item.get("num_decisions"))
        item["max_steps"] = parse_int(item.get("max_steps"))
        normalized.append(item)
    return normalized


def normalize_decisions(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        item = dict(row)
        item["decision_t"] = parse_int(item.get("decision_t"))
        item["player"] = parse_int(item.get("player"))
        item["fixed_player"] = None if item.get("fixed_player") in (None, "") else parse_int(item["fixed_player"])
        item["truncated"] = parse_bool(item.get("truncated"))
        item["rally_length"] = parse_int(item.get("rally_length"))
        normalized.append(item)
    return normalized


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def base_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    p1_wins = sum(row["winner"] == "player1" for row in rows)
    p2_wins = sum(row["winner"] == "player2" for row in rows)
    truncated = sum(row["truncated"] or row["winner"] == "truncated" for row in rows)
    lengths = [row["rally_length"] for row in rows]
    return {
        "rallies": n,
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "truncated": truncated,
        "p1_win_rate": rate(p1_wins, n),
        "p2_win_rate": rate(p2_wins, n),
        "truncation_rate": rate(truncated, n),
        "mean_rally_length": statistics.mean(lengths) if lengths else None,
        "median_rally_length": statistics.median(lengths) if lengths else None,
    }


def skill_distribution(
    decisions: list[dict[str, Any]],
    *,
    players: set[int] | None = None,
    winner: str | None = None,
    truncated: bool | None = None,
) -> Counter:
    counts = Counter()
    for row in decisions:
        if players is not None and row["player"] not in players:
            continue
        if winner is not None and row["winner"] != winner:
            continue
        if truncated is not None and row["truncated"] != truncated:
            continue
        counts[row["chosen_skill"]] += 1
    return counts


def distribution_rows(scope: str, counts: Counter) -> list[dict[str, Any]]:
    total = sum(counts.values())
    return [
        {
            "scope": scope,
            "skill": skill,
            "count": counts.get(skill, 0),
            "fraction": rate(counts.get(skill, 0), total),
        }
        for skill in SKILL_NAMES
    ]


def last_decisions(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for row in decisions:
        key = (row["rally_id"], row["player"])
        if key not in latest or row["decision_t"] > latest[key]["decision_t"]:
            latest[key] = row
    return list(latest.values())


def summarize_fixed_vs_random(
    rallies: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rally_groups = defaultdict(list)
    decision_groups = defaultdict(list)
    for row in rallies:
        if row["mode"] == "fixed_vs_random":
            rally_groups[(row["fixed_player"], row["fixed_skill"])].append(row)
    for row in decisions:
        if row["mode"] == "fixed_vs_random":
            decision_groups[(row["fixed_player"], row["fixed_skill"])].append(row)

    summary = []
    distributions = []
    last_skill_rows = []
    for fixed_player in (1, 2):
        random_player = 2 if fixed_player == 1 else 1
        for fixed_skill in SKILL_NAMES:
            rs = rally_groups.get((fixed_player, fixed_skill), [])
            ds = decision_groups.get((fixed_player, fixed_skill), [])
            metrics = base_metrics(rs)
            fixed_wins = sum(row["winner"] == f"player{fixed_player}" for row in rs)
            random_wins = sum(row["winner"] == f"player{random_player}" for row in rs)
            summary.append({
                "mode": "fixed_vs_random",
                "fixed_player": fixed_player,
                "fixed_skill": fixed_skill,
                "random_player": random_player,
                "rallies": metrics["rallies"],
                "fixed_player_wins": fixed_wins,
                "random_player_wins": random_wins,
                "truncated": metrics["truncated"],
                "fixed_player_win_rate": rate(fixed_wins, metrics["rallies"]),
                "random_player_win_rate": rate(random_wins, metrics["rallies"]),
                "truncation_rate": metrics["truncation_rate"],
                "mean_rally_length": metrics["mean_rally_length"],
                "median_rally_length": metrics["median_rally_length"],
            })
            random_decisions = [row for row in ds if row["player"] == random_player]
            distributions.extend(distribution_rows(
                f"fixed_player={fixed_player};fixed_skill={fixed_skill};random_overall",
                skill_distribution(random_decisions),
            ))
            distributions.extend(distribution_rows(
                f"fixed_player={fixed_player};fixed_skill={fixed_skill};random_wins",
                skill_distribution(random_decisions, winner=f"player{random_player}", truncated=False),
            ))
            distributions.extend(distribution_rows(
                f"fixed_player={fixed_player};fixed_skill={fixed_skill};random_losses",
                skill_distribution(random_decisions, winner=f"player{fixed_player}", truncated=False),
            ))
            distributions.extend(distribution_rows(
                f"fixed_player={fixed_player};fixed_skill={fixed_skill};truncations",
                skill_distribution(random_decisions, truncated=True),
            ))
            last_random = last_decisions(random_decisions)
            last_skill_rows.extend(distribution_rows(
                f"fixed_player={fixed_player};fixed_skill={fixed_skill};random_last_before_terminal",
                skill_distribution(last_random),
            ))
    return summary, distributions, last_skill_rows


def summarize_random_vs_random(
    rallies: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rs = [row for row in rallies if row["mode"] == "random_vs_random"]
    ds = [row for row in decisions if row["mode"] == "random_vs_random"]
    metrics = base_metrics(rs)
    summary = [{
        "mode": "random_vs_random",
        "rallies": metrics["rallies"],
        "p1_wins": metrics["p1_wins"],
        "p2_wins": metrics["p2_wins"],
        "truncated": metrics["truncated"],
        "p1_win_rate": metrics["p1_win_rate"],
        "p2_win_rate": metrics["p2_win_rate"],
        "truncation_rate": metrics["truncation_rate"],
        "mean_rally_length": metrics["mean_rally_length"],
        "median_rally_length": metrics["median_rally_length"],
    }]
    distributions = []
    for player in (1, 2):
        player_decisions = [row for row in ds if row["player"] == player]
        distributions.extend(distribution_rows(
            f"random_vs_random;player={player};overall",
            skill_distribution(player_decisions),
        ))
        distributions.extend(distribution_rows(
            f"random_vs_random;player={player};p1_wins",
            skill_distribution(player_decisions, winner="player1", truncated=False),
        ))
        distributions.extend(distribution_rows(
            f"random_vs_random;player={player};p2_wins",
            skill_distribution(player_decisions, winner="player2", truncated=False),
        ))
        distributions.extend(distribution_rows(
            f"random_vs_random;player={player};truncations",
            skill_distribution(player_decisions, truncated=True),
        ))
    last_skill_rows = distribution_rows(
        "random_vs_random;last_before_terminal",
        skill_distribution(last_decisions(ds)),
    )
    return summary, distributions, last_skill_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze randomized skill diagnostic outputs.")
    parser.add_argument("--input-dir", required=True, help="Directory containing rallies.csv and decisions.csv")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rallies = normalize_rallies(read_csv(input_dir / "rallies.csv"))
    decisions = normalize_decisions(read_csv(input_dir / "decisions.csv"))
    modes = {row["mode"] for row in rallies}
    outputs = []

    if "fixed_vs_random" in modes:
        summary, distributions, last_skills = summarize_fixed_vs_random(rallies, decisions)
        write_csv(out / "fixed_vs_random_summary.csv", [{k: format_value(v) for k, v in row.items()} for row in summary], [
            "mode", "fixed_player", "fixed_skill", "random_player", "rallies",
            "fixed_player_wins", "random_player_wins", "truncated",
            "fixed_player_win_rate", "random_player_win_rate", "truncation_rate",
            "mean_rally_length", "median_rally_length",
        ])
        write_csv(out / "fixed_vs_random_skill_distributions.csv", [{k: format_value(v) for k, v in row.items()} for row in distributions], [
            "scope", "skill", "count", "fraction",
        ])
        write_csv(out / "fixed_vs_random_last_skill_before_terminal.csv", [{k: format_value(v) for k, v in row.items()} for row in last_skills], [
            "scope", "skill", "count", "fraction",
        ])
        outputs.extend([
            "fixed_vs_random_summary.csv",
            "fixed_vs_random_skill_distributions.csv",
            "fixed_vs_random_last_skill_before_terminal.csv",
        ])

    if "random_vs_random" in modes:
        summary, distributions, last_skills = summarize_random_vs_random(rallies, decisions)
        write_csv(out / "random_vs_random_summary.csv", [{k: format_value(v) for k, v in row.items()} for row in summary], [
            "mode", "rallies", "p1_wins", "p2_wins", "truncated",
            "p1_win_rate", "p2_win_rate", "truncation_rate",
            "mean_rally_length", "median_rally_length",
        ])
        write_csv(out / "random_vs_random_skill_distributions.csv", [{k: format_value(v) for k, v in row.items()} for row in distributions], [
            "scope", "skill", "count", "fraction",
        ])
        write_csv(out / "random_vs_random_last_skill_before_terminal.csv", [{k: format_value(v) for k, v in row.items()} for row in last_skills], [
            "scope", "skill", "count", "fraction",
        ])
        outputs.extend([
            "random_vs_random_summary.csv",
            "random_vs_random_skill_distributions.csv",
            "random_vs_random_last_skill_before_terminal.csv",
        ])

    metadata = {
        "input_dir": str(input_dir),
        "output_dir": str(out),
        "rallies": len(rallies),
        "decisions": len(decisions),
        "modes": sorted(modes),
        "outputs": outputs,
        "interpretation_note": (
            "Skill distribution in wins counts every skill chosen during rallies that eventually "
            "ended in a win; it does not imply causal attribution. Last-skill-before-terminal "
            "is a separate diagnostic."
        ),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote randomized skill analysis to {out}")


if __name__ == "__main__":
    main()
