#!/usr/bin/env python3
"""Compare observed V3 pair returns against predicted Q distributions."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_arch import SimpleModel  # noqa: E402
from nash_skills.skills import SKILL_NAMES  # noqa: E402
from v3.constants import DEFENSE_POSITIONS  # noqa: E402
from v3.train_q_preposition import HIDDEN, INPUT_DIM, load_examples, q_matrix, safe_load_state_dict  # noqa: E402


OBSERVED_FIELDS = [
    "attack_skill",
    "defense_preposition",
    "count",
    "mean_attacker_return",
    "median_attacker_return",
    "std_attacker_return",
    "p25_attacker_return",
    "p75_attacker_return",
    "attacker_win_rate",
]

PREDICTED_FIELDS = [
    "attack_skill",
    "defense_preposition",
    "count",
    "mean_predicted_q",
    "median_predicted_q",
    "std_predicted_q",
    "p25_predicted_q",
    "p75_predicted_q",
]


def numeric_summary(values: list[float]) -> dict[str, float | int | str]:
    if not values:
        return {
            "count": 0,
            "mean": "",
            "median": "",
            "std": "",
            "p25": "",
            "p75": "",
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def pair_key(skill: str, position: str) -> tuple[str, str]:
    return skill, position


def empty_pair_map() -> dict[tuple[str, str], list[float]]:
    return {
        pair_key(skill, position): []
        for skill in SKILL_NAMES
        for position in DEFENSE_POSITIONS
    }


def summarize_observed(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pair = empty_pair_map()
    for example in examples:
        key = pair_key(str(example["attack_skill"]), str(example["defense_preposition"]))
        if key not in by_pair:
            raise ValueError(f"Unknown V3 action pair in labels: {key}")
        by_pair[key].append(float(example["attacker_return"]))

    rows: list[dict[str, Any]] = []
    for skill in SKILL_NAMES:
        for position in DEFENSE_POSITIONS:
            values = by_pair[pair_key(skill, position)]
            summary = numeric_summary(values)
            rows.append({
                "attack_skill": skill,
                "defense_preposition": position,
                "count": summary["count"],
                "mean_attacker_return": summary["mean"],
                "median_attacker_return": summary["median"],
                "std_attacker_return": summary["std"],
                "p25_attacker_return": summary["p25"],
                "p75_attacker_return": summary["p75"],
                "attacker_win_rate": "" if not values else float(np.mean(np.asarray(values) > 0.0)),
            })
    return rows


def load_model(path: str) -> SimpleModel:
    model = SimpleModel(INPUT_DIM, HIDDEN, 1)
    model.load_state_dict(safe_load_state_dict(path))
    model.eval()
    return model


def summarize_predicted(examples: list[dict[str, Any]], model: torch.nn.Module, device: str) -> list[dict[str, Any]]:
    model.to(device)
    by_pair = empty_pair_map()

    for example in examples:
        matrix = q_matrix(model, np.asarray(example["state"], dtype=np.float32), device=device)
        for skill_index, skill in enumerate(SKILL_NAMES):
            for position_index, position in enumerate(DEFENSE_POSITIONS):
                by_pair[pair_key(skill, position)].append(float(matrix[skill_index, position_index]))

    rows: list[dict[str, Any]] = []
    for skill in SKILL_NAMES:
        for position in DEFENSE_POSITIONS:
            summary = numeric_summary(by_pair[pair_key(skill, position)])
            rows.append({
                "attack_skill": skill,
                "defense_preposition": position,
                "count": summary["count"],
                "mean_predicted_q": summary["mean"],
                "median_predicted_q": summary["median"],
                "std_predicted_q": summary["std"],
                "p25_predicted_q": summary["p25"],
                "p75_predicted_q": summary["p75"],
            })
    return rows


def write_csv(path: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def combined_rows(observed: list[dict[str, Any]], predicted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predicted_by_pair = {
        pair_key(row["attack_skill"], row["defense_preposition"]): row
        for row in predicted
    }
    rows = []
    for obs in observed:
        pred = predicted_by_pair[pair_key(obs["attack_skill"], obs["defense_preposition"])]
        mean_obs = obs["mean_attacker_return"]
        mean_pred = pred["mean_predicted_q"]
        if mean_obs == "" or mean_pred == "":
            mean_gap = ""
        else:
            mean_gap = float(mean_pred) - float(mean_obs)
        rows.append({
            "attack_skill": obs["attack_skill"],
            "defense_preposition": obs["defense_preposition"],
            "observed_count": obs["count"],
            "observed_mean_attacker_return": mean_obs,
            "observed_median_attacker_return": obs["median_attacker_return"],
            "observed_std_attacker_return": obs["std_attacker_return"],
            "observed_p25_attacker_return": obs["p25_attacker_return"],
            "observed_p75_attacker_return": obs["p75_attacker_return"],
            "observed_attacker_win_rate": obs["attacker_win_rate"],
            "predicted_count": pred["count"],
            "predicted_mean_q": mean_pred,
            "predicted_median_q": pred["median_predicted_q"],
            "predicted_std_q": pred["std_predicted_q"],
            "predicted_p25_q": pred["p25_predicted_q"],
            "predicted_p75_q": pred["p75_predicted_q"],
            "mean_q_minus_mean_return": mean_gap,
        })
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose observed returns and predicted Q values for all V3 skill x pre-position pairs."
    )
    parser.add_argument("--labels", default="data/v3/labeled_preposition_5k.pkl")
    parser.add_argument("--model", default="models/v3/q_preposition_5k.pth")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--observed-csv", default="skill_eval/v3/observed_pair_returns.csv")
    parser.add_argument("--predicted-csv", default="skill_eval/v3/predicted_pair_q.csv")
    parser.add_argument("--combined-csv", default="skill_eval/v3/observed_vs_predicted_pair_q.csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    examples = load_examples(args.labels)
    if not examples:
        raise ValueError(f"No labeled examples found in {args.labels}")

    model = load_model(args.model)
    observed = summarize_observed(examples)
    predicted = summarize_predicted(examples, model, args.device)
    combined = combined_rows(observed, predicted)

    write_csv(args.observed_csv, observed, OBSERVED_FIELDS)
    write_csv(args.predicted_csv, predicted, PREDICTED_FIELDS)
    write_csv(args.combined_csv, combined, list(combined[0].keys()))

    print(f"Loaded examples: {len(examples)}")
    print(f"Loaded model: {args.model}")
    print(f"Saved observed return summary: {args.observed_csv}")
    print(f"Saved predicted Q summary: {args.predicted_csv}")
    print(f"Saved combined comparison: {args.combined_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
