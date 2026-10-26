#!/usr/bin/env python3
"""Aggregate per-decision skill-value diagnostics from eval_matchup.py."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - analysis can still emit CSVs
    plt = None

from nash_skills.skills import SKILL_NAMES


def parse_json_list(value: str) -> list[float]:
    if value in ("", None):
        return []
    try:
        return [float(x) for x in json.loads(value)]
    except Exception:
        return []


def read_rows(path: str) -> list[dict[str, Any]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["player"] = int(row["player"])
        row["decision_index"] = int(row["decision_index"])
        row["selected_skill_idx"] = int(row["selected_skill_idx"])
        row["selected_probability"] = (
            float(row["selected_probability"])
            if row.get("selected_probability") not in ("", None)
            else float("nan")
        )
        row["player_won_bool"] = (
            row.get("player_won") == "True"
            if row.get("player_won") in ("True", "False")
            else None
        )
        row["action_values"] = parse_json_list(row.get("action_values_json", ""))
        row["q_values"] = parse_json_list(row.get("q_values_json", ""))
        row["phi_values"] = parse_json_list(row.get("phi_values_json", ""))
        row["selection_probabilities"] = parse_json_list(row.get("selection_probabilities_json", ""))
    return rows


def load_training_freq(path: str | None) -> dict[int, Counter[str]]:
    freqs = {1: Counter(), 2: Counter()}
    if not path:
        return freqs
    # Compatibility for pickles written by NumPy 2 and read by older NumPy.
    import numpy as np

    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    with open(path, "rb") as f:
        rallies = pickle.load(f)
    for rally in rallies:
        for p1, p2 in rally.get("skill_pairs", []):
            freqs[1][p1] += 1
            freqs[2][p2] += 1
    return freqs


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": "",
            "median": "",
            "std": "",
            "p10": "",
            "p90": "",
            "min": "",
            "max": "",
        }
    arr = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "p10": float(np.quantile(arr, 0.10)),
        "p90": float(np.quantile(arr, 0.90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def value_by_skill(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        for family, key in (("action", "action_values"), ("q", "q_values"), ("phi", "phi_values")):
            values = row.get(key) or []
            if len(values) != len(SKILL_NAMES):
                continue
            for skill, value in zip(SKILL_NAMES, values):
                buckets[(row["player"], row["strategy"], family, skill)].append(float(value))
    output = []
    for (player, strategy, family, skill), values in sorted(buckets.items()):
        output.append({
            "player": player,
            "player_label": "P1" if player == 1 else "P2",
            "strategy": strategy,
            "value_family": family,
            "skill": skill,
            **stats(values),
        })
    return output


def choice_frequency(rows: list[dict[str, Any]], training_freq: dict[int, Counter[str]]) -> list[dict[str, Any]]:
    selected = Counter((r["player"], r["strategy"], r["opponent_skill"], r["selected_skill"]) for r in rows)
    totals = Counter((r["player"], r["strategy"], r["opponent_skill"]) for r in rows)
    train_totals = {player: sum(counter.values()) for player, counter in training_freq.items()}
    output = []
    for player, strategy, opp in sorted(totals):
        denom = totals[(player, strategy, opp)]
        for skill in SKILL_NAMES:
            count = selected[(player, strategy, opp, skill)]
            train_count = training_freq[player][skill]
            output.append({
                "player": player,
                "player_label": "P1" if player == 1 else "P2",
                "strategy": strategy,
                "opponent_skill": opp,
                "skill": skill,
                "selected_count": count,
                "selected_fraction": count / denom if denom else "",
                "training_count": train_count,
                "training_fraction": train_count / train_totals[player] if train_totals[player] else "",
            })
    return output


def calibration_bins(rows: list[dict[str, Any]], n_bins: int) -> list[dict[str, Any]]:
    output = []
    for family, key in (("action", "action_values"), ("q", "q_values"), ("phi", "phi_values")):
        grouped: dict[tuple[int, str], list[tuple[float, bool]]] = defaultdict(list)
        for row in rows:
            values = row.get(key) or []
            won = row.get("player_won_bool")
            if won is None or len(values) != len(SKILL_NAMES):
                continue
            grouped[(row["player"], row["strategy"])].append((float(values[row["selected_skill_idx"]]), bool(won)))
        for (player, strategy), points in sorted(grouped.items()):
            if not points:
                continue
            preds = np.asarray([p for p, _won in points], dtype=float)
            edges = np.quantile(preds, np.linspace(0.0, 1.0, n_bins + 1))
            edges = np.unique(edges)
            if len(edges) <= 1:
                edges = np.asarray([preds.min() - 1e-9, preds.max() + 1e-9])
            for bin_idx in range(len(edges) - 1):
                lo, hi = float(edges[bin_idx]), float(edges[bin_idx + 1])
                if bin_idx == len(edges) - 2:
                    subset = [(p, w) for p, w in points if lo <= p <= hi]
                else:
                    subset = [(p, w) for p, w in points if lo <= p < hi]
                if not subset:
                    continue
                output.append({
                    "player": player,
                    "player_label": "P1" if player == 1 else "P2",
                    "strategy": strategy,
                    "value_family": family,
                    "bin": bin_idx,
                    "pred_min": lo,
                    "pred_max": hi,
                    "count": len(subset),
                    "mean_predicted_value": float(np.mean([p for p, _w in subset])),
                    "empirical_player_win_rate": float(np.mean([w for _p, w in subset])),
                })
    return output


def decision_stage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["player"], row["strategy"], row["decision_index"], row["selected_skill"])].append(row)
    output = []
    for (player, strategy, decision_index, skill), group in sorted(buckets.items()):
        wins = [r["player_won_bool"] for r in group if r["player_won_bool"] is not None]
        selected_values = [
            r["action_values"][r["selected_skill_idx"]]
            for r in group
            if len(r.get("action_values") or []) == len(SKILL_NAMES)
        ]
        output.append({
            "player": player,
            "player_label": "P1" if player == 1 else "P2",
            "strategy": strategy,
            "decision_index": decision_index,
            "selected_skill": skill,
            "count": len(group),
            "mean_selected_action_value": float(np.mean(selected_values)) if selected_values else "",
            "empirical_player_win_rate": float(np.mean(wins)) if wins else "",
        })
    return output


def p1_p2_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for player in (1, 2):
        sub = [r for r in rows if r["player"] == player]
        selected = Counter(r["selected_skill"] for r in sub)
        values = [
            r["action_values"][r["selected_skill_idx"]]
            for r in sub
            if len(r.get("action_values") or []) == len(SKILL_NAMES)
        ]
        wins = [r["player_won_bool"] for r in sub if r["player_won_bool"] is not None]
        output.append({
            "player": player,
            "player_label": "P1" if player == 1 else "P2",
            "decisions": len(sub),
            "mean_selected_action_value": float(np.mean(values)) if values else "",
            "median_selected_action_value": float(median(values)) if values else "",
            "empirical_player_win_rate": float(np.mean(wins)) if wins else "",
            **{f"selected_{skill}": selected[skill] for skill in SKILL_NAMES},
        })
    return output


def win_loss_skill_distribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[int, str, bool], Counter[str]] = defaultdict(Counter)
    for row in rows:
        won = row.get("player_won_bool")
        if won is None:
            continue
        buckets[(row["player"], row["strategy"], bool(won))][row["selected_skill"]] += 1

    output = []
    keys = sorted({(player, strategy) for player, strategy, _won in buckets})
    for player, strategy in keys:
        for won in (True, False):
            counts = buckets.get((player, strategy, won), Counter())
            total = sum(counts.values())
            output.append({
                "player": player,
                "player_label": "P1" if player == 1 else "P2",
                "strategy": strategy,
                "outcome": "WIN" if won else "LOSS",
                "count": total,
                **{
                    skill: counts[skill] / total if total else 0.0
                    for skill in SKILL_NAMES
                },
            })
    return output


def print_win_loss_skill_distribution(rows: list[dict[str, Any]]) -> None:
    print("\nSelected-skill distribution by final player outcome")
    for row in win_loss_skill_distribution(rows):
        parts = ", ".join(f"{skill}={row[skill]:.2f}" for skill in SKILL_NAMES)
        print(
            f"  {row['player_label']} {row['strategy']} {row['outcome']}: "
            f"{parts}  (n={row['count']})"
        )


def plot_outputs(
    rows: list[dict[str, Any]],
    value_rows: list[dict[str, Any]],
    choice_rows: list[dict[str, Any]],
    calib_rows: list[dict[str, Any]],
    plots_dir: Path,
) -> None:
    if plt is None:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)

    for player in (1, 2):
        label = "P1" if player == 1 else "P2"
        action_values = [
            [
                r["action_values"][i]
                for r in rows
                if r["player"] == player and len(r.get("action_values") or []) == len(SKILL_NAMES)
            ]
            for i in range(len(SKILL_NAMES))
        ]
        if any(action_values):
            plt.figure(figsize=(9, 4))
            plt.boxplot(action_values, labels=SKILL_NAMES, showfliers=False)
            plt.xticks(rotation=25, ha="right")
            plt.ylabel("policy action value")
            plt.title(f"{label} action-value distribution by skill")
            plt.tight_layout()
            plt.savefig(plots_dir / f"{label}_action_value_by_skill.png", dpi=160)
            plt.close()

        selected = Counter(r["selected_skill"] for r in rows if r["player"] == player)
        player_choice_rows = [r for r in choice_rows if r["player"] == player]
        baseline_opp = player_choice_rows[0]["opponent_skill"] if player_choice_rows else None
        train = {
            r["skill"]: r["training_fraction"]
            for r in player_choice_rows
            if r["opponent_skill"] == baseline_opp
        }
        total = sum(selected.values())
        x = np.arange(len(SKILL_NAMES))
        plt.figure(figsize=(9, 4))
        plt.bar(x - 0.2, [selected[s] / total if total else 0 for s in SKILL_NAMES], width=0.4, label="selected")
        plt.bar(x + 0.2, [float(train.get(s) or 0) for s in SKILL_NAMES], width=0.4, label="training")
        plt.xticks(x, SKILL_NAMES, rotation=25, ha="right")
        plt.ylabel("fraction")
        plt.title(f"{label} selected-skill frequency vs training frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"{label}_selected_vs_training.png", dpi=160)
        plt.close()

        for family in ("action", "q", "phi"):
            sub = [r for r in calib_rows if r["player"] == player and r["value_family"] == family]
            if not sub:
                continue
            plt.figure(figsize=(6, 4))
            plt.scatter(
                [float(r["mean_predicted_value"]) for r in sub],
                [float(r["empirical_player_win_rate"]) for r in sub],
                s=[max(10, float(r["count"]) * 0.8) for r in sub],
                alpha=0.7,
            )
            plt.xlabel(f"mean selected {family} value")
            plt.ylabel("empirical player win rate")
            plt.title(f"{label} {family}: predicted value vs noisy outcome")
            plt.tight_layout()
            plt.savefig(plots_dir / f"{label}_{family}_calibration.png", dpi=160)
            plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze eval_matchup per-decision diagnostics.")
    parser.add_argument("--decision-log", required=True)
    parser.add_argument("--training-rallies", default=None)
    parser.add_argument("--output-dir", default="skill_eval/decision_diagnostics")
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--bins", type=int, default=8)
    args = parser.parse_args()

    rows = read_rows(args.decision_log)
    training_freq = load_training_freq(args.training_rallies)
    out_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir) if args.plots_dir else out_dir / "plots"

    value_rows = value_by_skill(rows)
    choice_rows = choice_frequency(rows, training_freq)
    calib_rows = calibration_bins(rows, args.bins)
    stage_rows = decision_stage(rows)
    symmetry_rows = p1_p2_summary(rows)
    win_loss_rows = win_loss_skill_distribution(rows)

    write_csv(out_dir / "value_by_skill.csv", value_rows)
    write_csv(out_dir / "choice_frequency.csv", choice_rows)
    write_csv(out_dir / "calibration_bins.csv", calib_rows)
    write_csv(out_dir / "decision_stage.csv", stage_rows)
    write_csv(out_dir / "p1_p2_summary.csv", symmetry_rows)
    write_csv(out_dir / "win_loss_skill_distribution.csv", win_loss_rows)
    plot_outputs(rows, value_rows, choice_rows, calib_rows, plots_dir)

    print(f"Loaded decision rows: {len(rows)}")
    print(f"Wrote summaries to: {out_dir}")
    print_win_loss_skill_distribution(rows)
    if plt is not None:
        print(f"Wrote plots to: {plots_dir}")
    else:
        print("Plots skipped: matplotlib unavailable")
    print("\nNote: empirical win rate is a noisy rally-level calibration diagnostic, not proof that an intermediate skill caused the win.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
