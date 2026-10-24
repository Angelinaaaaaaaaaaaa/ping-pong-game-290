#!/usr/bin/env python3
"""Train V3.2 Q(state, attack_skill, defense_preposition) -> attacker_return."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle as pkl
import random
import sys
from pathlib import Path
from typing import Any

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


INPUT_DIM = MODEL_INPUT_DIM
HIDDEN = [64, 32, 16]
CHECKPOINT_EVERY = 100


def encode_input(state: np.ndarray, attack_skill: str | int, defense_preposition: str | int) -> np.ndarray:
    return build_joint_action_input(state, attack_skill, defense_preposition)


def q_matrix(model: nn.Module, state: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Evaluate all 5 x 3 V3 action pairs for one 74-dim state."""
    rows = [encode_input(state, skill, position) for skill, position in iter_action_pairs()]
    model.eval()
    with torch.no_grad():
        x = torch.tensor(np.asarray(rows), dtype=torch.float32, device=device)
        values = model(x).reshape(len(SKILL_NAMES), len(DEFENSE_POSITIONS))
    return values.cpu().numpy()


def load_examples(path: str) -> list[dict[str, Any]]:
    with open(path, "rb") as f:
        payload = pkl.load(f)
    if isinstance(payload, dict) and "examples" in payload:
        encoder = payload.get("metadata", {}).get("state_encoder")
        if encoder != STATE_ENCODER_NAME:
            raise ValueError(
                f"Incompatible V3 labels: expected state encoder {STATE_ENCODER_NAME!r}, got {encoder!r}"
            )
        metadata = payload.get("metadata", {})
        if metadata.get("state_dim") != STATE_DIM:
            raise ValueError(f"Incompatible V3 labels: expected state_dim {STATE_DIM}, got {metadata.get('state_dim')!r}")
        if metadata.get("model_input_schema") != MODEL_INPUT_SCHEMA_VERSION:
            raise ValueError(
                f"Incompatible V3 labels: expected model_input_schema {MODEL_INPUT_SCHEMA_VERSION!r}, "
                f"got {metadata.get('model_input_schema')!r}"
            )
        return list(payload["examples"])
    if isinstance(payload, list):
        raise ValueError(
            "Legacy/list V3 labels do not declare the canonical state encoder; "
            "recollect and relabel the data."
        )
    raise ValueError(f"Unsupported label pickle format: {type(payload)}")


def build_dataset(examples: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
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


def split_indices(n: int, seed: int, train_frac: float, val_frac: float) -> dict[str, list[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if n < 3:
        return {"train": indices, "val": indices, "test": indices}
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    return {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }


def subset_loss(model: nn.Module, criterion, x: torch.Tensor, y: torch.Tensor, indices: list[int]) -> float:
    if not indices:
        return float("nan")
    model.eval()
    with torch.no_grad():
        idx = torch.tensor(indices, dtype=torch.long)
        return float(criterion(model(x[idx]), y[idx]).item())


def correlation(pred: np.ndarray, truth: np.ndarray) -> float | str:
    pred = np.asarray(pred, dtype=float).reshape(-1)
    truth = np.asarray(truth, dtype=float).reshape(-1)
    if len(pred) < 2 or np.std(pred) <= 1e-12 or np.std(truth) <= 1e-12:
        return ""
    return float(np.corrcoef(pred, truth)[0, 1])


def write_bin_diagnostics(path: str, predictions: np.ndarray, targets: np.ndarray, bins: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    targets = np.asarray(targets, dtype=float).reshape(-1)
    if len(predictions) == 0:
        rows = []
    else:
        order = np.argsort(predictions)
        chunks = np.array_split(order, min(bins, len(order)))
        rows = []
        for index, chunk in enumerate(chunks):
            if len(chunk) == 0:
                continue
            t = targets[chunk]
            p = predictions[chunk]
            rows.append({
                "bin": index,
                "count": len(chunk),
                "pred_q_min": float(np.min(p)),
                "pred_q_max": float(np.max(p)),
                "pred_q_mean": float(np.mean(p)),
                "empirical_avg_return": float(np.mean(t)),
                "attacker_win_rate": float(np.mean(t > 0)),
            })
    with open(path, "w", newline="") as f:
        fieldnames = [
            "bin",
            "count",
            "pred_q_min",
            "pred_q_max",
            "pred_q_mean",
            "empirical_avg_return",
            "attacker_win_rate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_q_matrix(path: str, matrix: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["attack_skill", *DEFENSE_POSITIONS])
        for skill, row in zip(SKILL_NAMES, matrix):
            writer.writerow([skill, *[float(v) for v in row]])


def safe_load_state_dict(path: str):
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    examples = load_examples(args.labels)
    x, y, kept_examples = build_dataset(examples)
    splits = split_indices(len(x), args.seed, args.train_frac, args.val_frac)

    print(f"Loaded examples: {len(x)}")
    print(f"Input dim: {INPUT_DIM} = state {STATE_DIM} + action {ACTION_DIM}")
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
    parser = argparse.ArgumentParser(description="Train V3.2 attacker-perspective Q model.")
    parser.add_argument("--labels", default="data/v3/labeled_preposition_state74_examples.pkl")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--model-out", default="models/v3/q_preposition_state74.pth")
    parser.add_argument("--log-csv", default="logs/v3/train_q_preposition_state74.csv")
    parser.add_argument("--metrics-json", default="skill_eval/v3/q_preposition_state74_metrics.json")
    parser.add_argument("--bins-csv", default="skill_eval/v3/q_preposition_state74_bins.csv")
    parser.add_argument("--q-matrix-csv", default="skill_eval/v3/q_preposition_state74_sample_matrix.csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
