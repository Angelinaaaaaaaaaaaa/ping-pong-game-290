#!/usr/bin/env python3
"""Train the V3 Q model with one-hot action encoding.

This is the corrected V3 one-hot trainer. It reuses the existing V3 dataset,
split logic, MLP shape, optimizer, checkpoint selection, and diagnostics while
using the state74/action-onehot schema.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_arch import SimpleModel  # noqa: E402
from nash_skills.skills import SKILL_NAMES  # noqa: E402
from v3.constants import DEFENSE_POSITIONS  # noqa: E402
from v3.model_input import (  # noqa: E402
    ACTION_DIM,
    MODEL_INPUT_DIM,
    MODEL_INPUT_SCHEMA_VERSION,
    build_joint_action_input,
    iter_action_pairs,
)
from v3.state_encoder import STATE_DIM, STATE_ENCODER_NAME  # noqa: E402
from v3.train_q_preposition import (  # noqa: E402
    HIDDEN,
    correlation,
    load_examples,
    safe_load_state_dict,
    save_q_matrix,
    split_indices,
    subset_loss,
    write_bin_diagnostics,
)


INPUT_DIM = MODEL_INPUT_DIM


def encode_input(
    state: np.ndarray,
    attack_skill: str | int,
    defense_preposition: str | int,
) -> np.ndarray:
    """Encode one state/action pair as 74 state + 5 + 3 one-hot dims."""
    return build_joint_action_input(state, attack_skill, defense_preposition)


def q_matrix(model: nn.Module, state: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Evaluate all 5 x 3 V3 action pairs for one 74-dim state."""
    rows = [encode_input(state, skill, position) for skill, position in iter_action_pairs()]
    model.eval()
    with torch.no_grad():
        x = torch.tensor(np.asarray(rows), dtype=torch.float32, device=device)
        values = model(x).reshape(len(SKILL_NAMES), len(DEFENSE_POSITIONS))
    return values.cpu().numpy()


def build_dataset(examples: list[dict]) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    x_rows = []
    y_rows = []
    kept = []
    for example in examples:
        x_rows.append(encode_input(
            example["state"],
            example["attack_skill"],
            example["defense_preposition"],
        ))
        y_rows.append([float(example["attacker_return"])])
        kept.append(example)
    if not x_rows:
        raise ValueError("No V3 labeled examples found")
    x = torch.tensor(np.asarray(x_rows), dtype=torch.float32)
    y = torch.tensor(np.asarray(y_rows), dtype=torch.float32)
    return x, y, kept


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    examples = load_examples(args.labels)
    x, y, kept_examples = build_dataset(examples)
    splits = split_indices(len(x), args.seed, args.train_frac, args.val_frac)

    print(f"Loaded examples: {len(x)}")
    print(f"Input dim: {INPUT_DIM} = state {STATE_DIM} + action {ACTION_DIM}")
    print(f"Action encoding: {len(SKILL_NAMES)}-dim attack one-hot + {len(DEFENSE_POSITIONS)}-dim defense one-hot")
    print(f"Splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    model = SimpleModel(INPUT_DIM, HIDDEN, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_epoch = 0

    train_idx = torch.tensor(splits["train"], dtype=torch.long)
    with open(args.log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "test_loss", "saved"])
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(x[train_idx]), y[train_idx])
            loss.backward()
            optimizer.step()

            train_loss = subset_loss(model, criterion, x, y, splits["train"])
            val_loss = subset_loss(model, criterion, x, y, splits["val"])
            test_loss = subset_loss(model, criterion, x, y, splits["test"])
            saved = 0
            if val_loss <= best_val:
                best_val = val_loss
                best_epoch = epoch + 1
                torch.save(model.state_dict(), args.model_out)
                saved = 1
            writer.writerow([epoch + 1, train_loss, val_loss, test_loss, saved])
            if (epoch + 1) % args.print_every == 0 or epoch == 0:
                print(
                    f"epoch {epoch+1}/{args.epochs} "
                    f"train={train_loss:.5f} val={val_loss:.5f} test={test_loss:.5f} saved={saved}",
                    flush=True,
                )

    model.load_state_dict(safe_load_state_dict(args.model_out))
    model.eval()
    with torch.no_grad():
        pred = model(x).cpu().numpy().reshape(-1)
    target = y.cpu().numpy().reshape(-1)
    corr = correlation(pred, target)

    write_bin_diagnostics(args.bins_csv, pred, target, args.bins)
    sample_matrix = q_matrix(model, kept_examples[0]["state"])
    save_q_matrix(args.q_matrix_csv, sample_matrix)

    metadata = {
        "labels": args.labels,
        "model_out": args.model_out,
        "input_dim": INPUT_DIM,
        "state_dim": STATE_DIM,
        "state_encoder": STATE_ENCODER_NAME,
        "action_dim": ACTION_DIM,
        "model_input_schema": MODEL_INPUT_SCHEMA_VERSION,
        "attack_encoding": "one_hot",
        "defense_encoding": "one_hot",
        "hidden": HIDDEN,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "splits": {k: len(v) for k, v in splits.items()},
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "final_train_loss": subset_loss(model, criterion, x, y, splits["train"]),
        "final_val_loss": subset_loss(model, criterion, x, y, splits["val"]),
        "final_test_loss": subset_loss(model, criterion, x, y, splits["test"]),
        "prediction_target_correlation": corr,
        "attack_skills": list(SKILL_NAMES),
        "defense_prepositions": list(DEFENSE_POSITIONS),
    }
    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_json, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved best model: {args.model_out}")
    print(f"Saved training log: {args.log_csv}")
    print(f"Saved metrics: {args.metrics_json}")
    print(f"Saved bin diagnostics: {args.bins_csv}")
    print(f"Saved sample 5x3 Q matrix: {args.q_matrix_csv}")
    print(f"Prediction/target correlation: {corr}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train V3 Q model with one-hot action encoding.")
    parser.add_argument("--labels", default="data/v3/labeled_preposition_state74_examples.pkl")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--model-out", default="models/v3/q_preposition_state74_onehot.pth")
    parser.add_argument("--log-csv", default="logs/v3/train_q_preposition_state74_onehot.csv")
    parser.add_argument("--metrics-json", default="skill_eval/v3/q_preposition_state74_onehot_metrics.json")
    parser.add_argument("--bins-csv", default="skill_eval/v3/q_preposition_state74_onehot_bins.csv")
    parser.add_argument("--q-matrix-csv", default="skill_eval/v3/q_preposition_state74_onehot_sample_matrix.csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
