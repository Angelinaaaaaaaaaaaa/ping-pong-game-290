#!/usr/bin/env python3
"""Run the existing pair diagnostic against the V3 one-hot Q model."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_arch import SimpleModel  # noqa: E402
from nash_skills.skills import SKILL_NAMES  # noqa: E402
from v3.constants import DEFENSE_POSITIONS  # noqa: E402
from v3.diagnose_pair_q import (  # noqa: E402
    OBSERVED_FIELDS,
    PREDICTED_FIELDS,
    combined_rows,
    empty_pair_map,
    numeric_summary,
    summarize_observed,
    write_csv,
)
from v3.train_q_preposition_onehot import (  # noqa: E402
    HIDDEN,
    INPUT_DIM,
    load_examples,
    q_matrix,
    safe_load_state_dict,
)


def load_model(path: str) -> SimpleModel:
    model = SimpleModel(INPUT_DIM, HIDDEN, 1)
    model.load_state_dict(safe_load_state_dict(path))
    model.eval()
    return model


def summarize_predicted_onehot(
    examples: list[dict], model: torch.nn.Module, device: str
) -> list[dict]:
    """Aggregate one-hot Q predictions with the existing diagnostic schema."""
    model.to(device)
    by_pair = empty_pair_map()

    for example in examples:
        matrix = q_matrix(model, np.asarray(example["state"], dtype=np.float32), device=device)
        for skill_index, skill in enumerate(SKILL_NAMES):
            for position_index, position in enumerate(DEFENSE_POSITIONS):
                by_pair[(skill, position)].append(float(matrix[skill_index, position_index]))

    rows = []
    for skill in SKILL_NAMES:
        for position in DEFENSE_POSITIONS:
            summary = numeric_summary(by_pair[(skill, position)])
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose observed V3 returns against one-hot predicted Q values."
    )
    parser.add_argument("--labels", default="data/v3/labeled_preposition_5k.pkl")
    parser.add_argument("--model", default="models/v3/q_preposition_onehot.pth")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--observed-csv", default="skill_eval/v3/observed_pair_returns_onehot.csv")
    parser.add_argument("--predicted-csv", default="skill_eval/v3/predicted_pair_q_onehot.csv")
    parser.add_argument(
        "--combined-csv",
        default="skill_eval/v3/observed_vs_predicted_pair_q_onehot.csv",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    examples = load_examples(args.labels)
    if not examples:
        raise ValueError(f"No labeled examples found in {args.labels}")

    model = load_model(args.model)
    observed = summarize_observed(examples)
    predicted = summarize_predicted_onehot(examples, model, args.device)
    combined = combined_rows(observed, predicted)

    write_csv(args.observed_csv, observed, OBSERVED_FIELDS)
    write_csv(args.predicted_csv, predicted, PREDICTED_FIELDS)
    write_csv(args.combined_csv, combined, list(combined[0].keys()))

    print(f"Loaded examples: {len(examples)}")
    print(f"Loaded one-hot model: {args.model}")
    print(f"Saved observed return summary: {args.observed_csv}")
    print(f"Saved predicted Q summary: {args.predicted_csv}")
    print(f"Saved combined comparison: {args.combined_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
