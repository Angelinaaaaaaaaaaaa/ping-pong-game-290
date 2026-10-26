#!/usr/bin/env python3
"""Train a V3 one-hot neural potential from per-state LP targets.

The model estimates:

    Phi_theta(state, attack_one_hot, defense_one_hot) -> scalar

It uses the current V3 Q model's 82-input representation and the same MLP
hidden sizes and optimizer family as the one-hot Q trainer.  Unlike the Q
model, the potential head is linear: the anchored LP targets are not required
to lie in [-1, 1].  Evaluation includes the supervised target metrics and the
per-state unilateral-deviation alpha induced by the neural potential.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
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
from v3.potential_lp import DEVIATIONS as _DEVIATIONS, deviation_q_delta as _deviation_q_delta  # noqa: E402
from v3.state_encoder import STATE_DIM, STATE_ENCODER_NAME  # noqa: E402
from v3.model_input import MODEL_INPUT_SCHEMA_VERSION, build_joint_action_input  # noqa: E402
from v3.train_q_preposition_onehot import HIDDEN, INPUT_DIM, safe_load_state_dict, split_indices  # noqa: E402


DEFAULT_LABELS = "data/v3/potential_labels_state74.pkl"
DEFAULT_MODEL = "models/v3/potential_state74.pth"
DEFAULT_LOG = "logs/v3/train_potential_state74.csv"
DEFAULT_METRICS = "skill_eval/v3/potential_state74_metrics.json"
DEFAULT_STATE_CSV = "skill_eval/v3/potential_state74_state_metrics.csv"
DEFAULT_BASELINE_MODEL = "models/v3/potential_state74_baseline.pth"
N_ACTION_PAIRS = len(SKILL_NAMES) * len(DEFENSE_POSITIONS)


def load_potential_labels(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "examples" not in payload or "state_solutions" not in payload:
        raise ValueError(f"Unsupported potential-label payload: {path}")
    metadata = payload.get("metadata", {})
    if metadata.get("state_encoder") != STATE_ENCODER_NAME:
        raise ValueError(
            f"Expected state encoder {STATE_ENCODER_NAME!r}, got {metadata.get('state_encoder')!r}"
        )
    if metadata.get("input_dim") != INPUT_DIM:
        raise ValueError(f"Expected potential input_dim {INPUT_DIM}, got {metadata.get('input_dim')!r}")
    if metadata.get("state_dim") != STATE_DIM:
        raise ValueError(f"Expected potential state_dim {STATE_DIM}, got {metadata.get('state_dim')!r}")
    if metadata.get("model_input_schema") != MODEL_INPUT_SCHEMA_VERSION:
        raise ValueError(
            f"Expected potential model_input_schema {MODEL_INPUT_SCHEMA_VERSION!r}, "
            f"got {metadata.get('model_input_schema')!r}"
        )
    return payload


def build_dataset(payload: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    examples = payload["examples"]
    states = np.asarray(examples["states"], dtype=np.float32)
    attack_indices = np.asarray(examples["attack_indices"], dtype=np.int64)
    defense_indices = np.asarray(examples["defense_indices"], dtype=np.int64)
    state_indices = np.asarray(examples["state_indices"], dtype=np.int64)
    targets = np.asarray(examples["potential_targets"], dtype=np.float32).reshape(-1, 1)

    n = len(states)
    if states.shape != (n, STATE_DIM):
        raise ValueError(f"Expected example states with shape (N, {STATE_DIM}), got {states.shape}")
    if not all(len(values) == n for values in (attack_indices, defense_indices, state_indices, targets)):
        raise ValueError("Potential example arrays have inconsistent lengths")

    rows = [
        build_joint_action_input(states[index], int(attack_indices[index]), int(defense_indices[index]))
        for index in range(n)
    ]
    x = torch.tensor(np.asarray(rows), dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    return x, y, state_indices


def state_level_splits(n_states: int, seed: int, train_frac: float, val_frac: float) -> dict[str, list[int]]:
    return split_indices(n_states, seed, train_frac, val_frac)


def expand_state_indices(state_split: list[int], n_pairs: int) -> np.ndarray:
    states = np.asarray(state_split, dtype=np.int64)
    return (states[:, None] * n_pairs + np.arange(n_pairs, dtype=np.int64)[None, :]).reshape(-1)


def subset_loss(model: nn.Module, criterion: nn.Module, x: torch.Tensor, y: torch.Tensor, indices: np.ndarray) -> float:
    if len(indices) == 0:
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


def predict_in_batches(model: nn.Module, x: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), max(1, int(batch_size))):
            outputs.append(model(x[start:start + batch_size]).cpu().numpy())
    return np.concatenate(outputs, axis=0).reshape(-1)


def freeze_batch_norm_with_training_stats(
    model: SimpleModel,
    x: torch.Tensor,
    train_indices: np.ndarray,
) -> None:
    """Use fixed input statistics, matching the V2 potential-network pattern."""
    indices = torch.tensor(train_indices, dtype=torch.long)
    train_x = x[indices]
    with torch.no_grad():
        model.batch_norm.running_mean.copy_(train_x.mean(dim=0))
        model.batch_norm.running_var.copy_(train_x.var(dim=0, unbiased=False).clamp_min(1e-5))
    model.batch_norm.eval()


def neural_alpha(q: np.ndarray, phi: np.ndarray) -> float:
    """Return max unilateral-deviation residual for a fixed neural Phi."""
    q = np.asarray(q, dtype=float)
    phi = np.asarray(phi, dtype=float)
    residuals = []
    for deviation in _DEVIATIONS:
        _, attack, defense, attack_prime, defense_prime = deviation
        q_delta = _deviation_q_delta(q, deviation)
        phi_delta = phi[attack_prime, defense_prime] - phi[attack, defense]
        residuals.append(abs(float(phi_delta) - q_delta))
    return float(max(residuals, default=0.0))


def deviation_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened source/destination indices and player masks."""
    source = []
    destination = []
    attacker = []
    defender = []
    for player, attack, defense, attack_prime, defense_prime in _DEVIATIONS:
        source.append(attack * len(DEFENSE_POSITIONS) + defense)
        destination.append(attack_prime * len(DEFENSE_POSITIONS) + defense_prime)
        attacker.append(player == "attacker")
        defender.append(player == "defender")
    return (
        np.asarray(source, dtype=np.int64),
        np.asarray(destination, dtype=np.int64),
        np.asarray(attacker, dtype=bool),
        np.asarray(defender, dtype=bool),
    )


def utility_deviation_targets(q_matrices: np.ndarray) -> np.ndarray:
    """Return attacker and defender utility changes in _DEVIATIONS order."""
    q_flat = np.asarray(q_matrices, dtype=np.float32).reshape(len(q_matrices), N_ACTION_PAIRS)
    deltas = []
    for player, attack, defense, attack_prime, defense_prime in _DEVIATIONS:
        source_index = attack * len(DEFENSE_POSITIONS) + defense
        destination_index = attack_prime * len(DEFENSE_POSITIONS) + defense_prime
        source = q_flat[:, source_index]
        destination = q_flat[:, destination_index]
        deltas.append(destination - source if player == "attacker" else source - destination)
    return np.stack(deltas, axis=1)


def _predict_state_potentials(
    model: nn.Module,
    x: torch.Tensor,
    state_indices: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    example_indices = expand_state_indices(list(map(int, state_indices)), N_ACTION_PAIRS)
    selected_x = x[torch.tensor(example_indices, dtype=torch.long)]
    return predict_in_batches(model, selected_x, batch_size).reshape(len(state_indices), N_ACTION_PAIRS)


def normalized_alpha_for_states(
    model: nn.Module,
    x: torch.Tensor,
    q_matrices: np.ndarray,
    q_ranges: np.ndarray,
    state_indices: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    phi = _predict_state_potentials(model, x, state_indices, batch_size).reshape(
        len(state_indices), len(SKILL_NAMES), len(DEFENSE_POSITIONS)
    )
    selected_q = np.asarray(q_matrices)[state_indices]
    alpha = np.asarray([neural_alpha(q, p) for q, p in zip(selected_q, phi)], dtype=float)
    ranges = np.asarray(q_ranges)[state_indices]
    normalized = np.divide(alpha, ranges, out=np.zeros_like(alpha), where=ranges > 1e-12)
    return alpha, normalized


def summarize(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p5": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
    }


def split_regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=float).reshape(-1)
    return {
        "mse": float(np.mean((prediction - target) ** 2)),
        "correlation": correlation(prediction, target),
    }


def evaluate(
    model: nn.Module,
    payload: dict[str, Any],
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    sample_indices: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prediction = predict_in_batches(model, x, batch_size)
    target = y.numpy().reshape(-1)
    prediction_mse = float(np.mean((prediction - target) ** 2))
    state_solutions = payload["state_solutions"]
    q_matrices = np.asarray(state_solutions["q_matrices"], dtype=float)
    lp_phi = np.asarray(state_solutions["phi_matrices"], dtype=float)
    lp_alphas = np.asarray(state_solutions["lp_alphas"], dtype=float)
    q_ranges = np.asarray(state_solutions["q_ranges"], dtype=float)
    n_states = len(q_matrices)
    if prediction.shape[0] != n_states * N_ACTION_PAIRS:
        raise ValueError("Prediction count does not match state-solution count")
    nn_phi = prediction.reshape(n_states, len(SKILL_NAMES), len(DEFENSE_POSITIONS))
    nn_alphas = np.asarray([neural_alpha(q, phi) for q, phi in zip(q_matrices, nn_phi)])
    lp_normalized = np.divide(lp_alphas, q_ranges, out=np.zeros_like(lp_alphas), where=q_ranges > 1e-12)
    nn_normalized = np.divide(nn_alphas, q_ranges, out=np.zeros_like(nn_alphas), where=q_ranges > 1e-12)
    source, destination, attacker_mask, defender_mask = deviation_arrays()
    phi_flat = nn_phi.reshape(n_states, N_ACTION_PAIRS)
    lp_phi_flat = lp_phi.reshape(n_states, N_ACTION_PAIRS)
    phi_deltas = phi_flat[:, destination] - phi_flat[:, source]
    lp_phi_deltas = lp_phi_flat[:, destination] - lp_phi_flat[:, source]
    utility_deltas = utility_deviation_targets(q_matrices)
    attacker_sign_agreement = np.mean(
        np.sign(phi_deltas[:, attacker_mask]) == np.sign(utility_deltas[:, attacker_mask]), axis=1
    )
    defender_sign_agreement = np.mean(
        np.sign(phi_deltas[:, defender_mask]) == np.sign(utility_deltas[:, defender_mask]), axis=1
    )

    state_rows = []
    for index in range(n_states):
        state_rows.append({
            "state_index": index,
            "q_range": float(q_ranges[index]),
            "lp_alpha": float(lp_alphas[index]),
            "neural_alpha": float(nn_alphas[index]),
            "alpha_gap": float(nn_alphas[index] - lp_alphas[index]),
            "lp_normalized_alpha": float(lp_normalized[index]),
            "neural_normalized_alpha": float(nn_normalized[index]),
            "attacker_sign_agreement": float(attacker_sign_agreement[index]),
            "defender_sign_agreement": float(defender_sign_agreement[index]),
        })

    alpha_gap = nn_alphas - lp_alphas
    metrics: dict[str, Any] = {
        "potential_prediction_mse": prediction_mse,
        "potential_prediction_correlation": correlation(prediction, target),
        "lp_alpha": summarize(lp_alphas),
        "neural_alpha": summarize(nn_alphas),
        "lp_normalized_alpha": summarize(lp_normalized),
        "neural_normalized_alpha": summarize(nn_normalized),
        "attacker_sign_agreement": float(np.mean(attacker_sign_agreement)),
        "defender_sign_agreement": float(np.mean(defender_sign_agreement)),
        "attacker_sign_agreement_by_state": attacker_sign_agreement.tolist(),
        "defender_sign_agreement_by_state": defender_sign_agreement.tolist(),
        "deviation_mse_to_lp_potential": float(np.mean((phi_deltas - lp_phi_deltas) ** 2)),
        "deviation_mse_to_utility": float(np.mean((phi_deltas - utility_deltas) ** 2)),
        "alpha_comparison": {
            "mean_gap_neural_minus_lp": float(np.mean(alpha_gap)),
            "median_gap_neural_minus_lp": float(np.median(alpha_gap)),
            "mae": float(np.mean(np.abs(alpha_gap))),
            "rmse": float(np.sqrt(np.mean(alpha_gap ** 2))),
            "correlation": correlation(nn_alphas, lp_alphas),
            "fraction_neural_alpha_ge_lp_alpha": float(np.mean(nn_alphas + 1e-7 >= lp_alphas)),
        },
        "samples": [],
    }
    for index in sample_indices:
        index = int(index)
        metrics["samples"].append({
            "state_index": index,
            "q": q_matrices[index].tolist(),
            "lp_phi": lp_phi[index].tolist(),
            "neural_phi": nn_phi[index].tolist(),
            "lp_alpha": float(lp_alphas[index]),
            "neural_alpha": float(nn_alphas[index]),
            "lp_normalized_alpha": float(lp_normalized[index]),
            "neural_normalized_alpha": float(nn_normalized[index]),
            "attacker_sign_agreement": float(attacker_sign_agreement[index]),
            "defender_sign_agreement": float(defender_sign_agreement[index]),
        })
    return metrics, state_rows


def write_state_csv(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else [
        "state_index", "q_range", "lp_alpha", "neural_alpha", "alpha_gap",
        "lp_normalized_alpha", "neural_normalized_alpha",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    payload = load_potential_labels(args.labels)
    x, y, state_indices = build_dataset(payload)
    n_states = int(payload["metadata"]["n_states"])
    n_pairs = N_ACTION_PAIRS
    if len(x) != n_states * n_pairs or not np.array_equal(state_indices, np.repeat(np.arange(n_states), n_pairs)):
        raise ValueError("Potential labels are not in the expected state-major order")
    splits = state_level_splits(n_states, args.seed, args.train_frac, args.val_frac)
    train_idx = expand_state_indices(splits["train"], n_pairs)
    val_idx = expand_state_indices(splits["val"], n_pairs)
    test_idx = expand_state_indices(splits["test"], n_pairs)

    source_np, destination_np, attacker_mask_np, defender_mask_np = deviation_arrays()
    source = torch.tensor(source_np, dtype=torch.long)
    destination = torch.tensor(destination_np, dtype=torch.long)
    solutions = payload["state_solutions"]
    lp_phi_all = torch.tensor(np.asarray(solutions["phi_matrices"], dtype=np.float32).reshape(n_states, n_pairs))
    q_matrices = np.asarray(solutions["q_matrices"], dtype=np.float32)
    utility_delta_all = torch.tensor(utility_deviation_targets(q_matrices), dtype=torch.float32)
    q_ranges = np.asarray(solutions["q_ranges"], dtype=float)
    lp_alphas = np.asarray(solutions["lp_alphas"], dtype=float)

    model = SimpleModel(INPUT_DIM, HIDDEN, 1, last_layer_activation=None)
    freeze_batch_norm_with_training_stats(model, x, train_idx)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_epoch = 0
    log_path = args.log_csv
    if args.eval_only:
        if not os.path.exists(args.model_out):
            raise FileNotFoundError(f"Cannot evaluate missing checkpoint: {args.model_out}")
        if os.path.exists(args.log_csv):
            with open(args.log_csv, newline="") as prior_log:
                for row in csv.DictReader(prior_log):
                    if row.get("saved") == "1":
                        best_epoch = int(row["epoch"])
                        best_val = float(row["val_normalized_alpha"])
        log_path = os.devnull

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_phi_loss", "train_delta_lp_loss",
            "train_delta_q_loss", "val_phi_loss", "val_normalized_alpha",
            "test_phi_loss", "saved",
        ])
        for epoch in range(0 if args.eval_only else args.epochs):
            model.train()
            # Keep the potential model's BN statistics fixed, as in the V2
            # potential workflow; otherwise each random mini-batch changes
            # the inference normalization used for validation and alpha.
            model.batch_norm.eval()
            state_permutation = np.random.permutation(splits["train"])
            train_components = np.zeros(4, dtype=float)
            train_state_count = 0
            # Keep all 15 action pairs for a state in one batch so the
            # deviation losses are formed from matched counterfactuals.
            states_per_batch = max(1, int(args.batch_size) // N_ACTION_PAIRS)
            for start in range(0, len(state_permutation), states_per_batch):
                batch_states = state_permutation[start:start + states_per_batch]
                flat_indices = expand_state_indices(list(map(int, batch_states)), N_ACTION_PAIRS)
                indices = torch.tensor(flat_indices, dtype=torch.long)
                optimizer.zero_grad()
                prediction = model(x[indices]).reshape(len(batch_states), N_ACTION_PAIRS)
                lp_phi = lp_phi_all[batch_states]
                utility_delta = utility_delta_all[batch_states]
                predicted_delta = prediction[:, destination] - prediction[:, source]
                lp_delta = lp_phi[:, destination] - lp_phi[:, source]
                phi_loss = criterion(prediction, lp_phi)
                delta_lp_loss = criterion(predicted_delta, lp_delta)
                delta_q_loss = criterion(predicted_delta, utility_delta)
                loss = phi_loss + args.lambda_lp * delta_lp_loss + args.lambda_q * delta_q_loss
                loss.backward()
                optimizer.step()
                n_batch_states = len(batch_states)
                train_components += np.asarray([
                    float(loss.item()), float(phi_loss.item()),
                    float(delta_lp_loss.item()), float(delta_q_loss.item()),
                ]) * n_batch_states
                train_state_count += n_batch_states

            train_components /= max(1, train_state_count)
            val_phi_loss = subset_loss(model, criterion, x, y, val_idx)
            test_phi_loss = subset_loss(model, criterion, x, y, test_idx)
            _, val_normalized_alpha_values = normalized_alpha_for_states(
                model, x, q_matrices, q_ranges,
                np.asarray(splits["val"], dtype=np.int64), args.eval_batch_size,
            )
            val_normalized_alpha = float(np.mean(val_normalized_alpha_values))
            saved = 0
            if val_normalized_alpha <= best_val:
                best_val = val_normalized_alpha
                best_epoch = epoch + 1
                torch.save(model.state_dict(), args.model_out)
                saved = 1
            writer.writerow([
                epoch + 1, *train_components.tolist(), val_phi_loss,
                val_normalized_alpha, test_phi_loss, saved,
            ])
            if (epoch + 1) % args.print_every == 0 or epoch == 0:
                print(
                    f"epoch {epoch + 1}/{args.epochs} "
                    f"train={train_components[0]:.6f} val_phi={val_phi_loss:.6f} "
                    f"val_norm_alpha={val_normalized_alpha:.6f} "
                    f"test_phi={test_phi_loss:.6f} saved={saved}",
                    flush=True,
                )

    model.load_state_dict(safe_load_state_dict(args.model_out))
    sample_count = min(max(0, int(args.n_samples)), n_states)
    sample_rng = np.random.default_rng(args.seed)
    sample_indices = np.sort(sample_rng.choice(n_states, size=sample_count, replace=False))
    metrics, state_rows = evaluate(model, payload, x, y, args.eval_batch_size, sample_indices)
    metrics["split_mse"] = {
        "train": subset_loss(model, criterion, x, y, train_idx),
        "validation": subset_loss(model, criterion, x, y, val_idx),
        "test": subset_loss(model, criterion, x, y, test_idx),
    }
    selected_prediction = predict_in_batches(model, x, args.eval_batch_size)
    target_values = y.numpy().reshape(-1)
    metrics["split_potential_metrics"] = {}
    for split_name, split_states in (("train", splits["train"]), ("validation", splits["val"]), ("test", splits["test"])):
        split_examples = expand_state_indices(split_states, n_pairs)
        metrics["split_potential_metrics"][split_name] = split_regression_metrics(
            selected_prediction[split_examples], target_values[split_examples]
        )
    val_alpha, val_norm_alpha = normalized_alpha_for_states(
        model, x, q_matrices, q_ranges, np.asarray(splits["val"], dtype=np.int64), args.eval_batch_size
    )
    test_alpha, test_norm_alpha = normalized_alpha_for_states(
        model, x, q_matrices, q_ranges, np.asarray(splits["test"], dtype=np.int64), args.eval_batch_size
    )
    val_state_indices = np.asarray(splits["val"], dtype=np.int64)
    test_state_indices = np.asarray(splits["test"], dtype=np.int64)
    metrics["split_alpha"] = {
        "validation": {
            "lp_alpha": summarize(lp_alphas[val_state_indices]),
            "neural_alpha": summarize(val_alpha),
            "neural_minus_lp_alpha": summarize(val_alpha - lp_alphas[val_state_indices]),
            "lp_normalized_alpha": summarize(
                np.divide(lp_alphas[val_state_indices], q_ranges[val_state_indices],
                          out=np.zeros_like(lp_alphas[val_state_indices]), where=q_ranges[val_state_indices] > 1e-12)
            ),
            "neural_normalized_alpha": summarize(val_norm_alpha),
        },
        "test": {
            "lp_alpha": summarize(lp_alphas[test_state_indices]),
            "neural_alpha": summarize(test_alpha),
            "neural_minus_lp_alpha": summarize(test_alpha - lp_alphas[test_state_indices]),
            "lp_normalized_alpha": summarize(
                np.divide(lp_alphas[test_state_indices], q_ranges[test_state_indices],
                          out=np.zeros_like(lp_alphas[test_state_indices]), where=q_ranges[test_state_indices] > 1e-12)
            ),
            "neural_normalized_alpha": summarize(test_norm_alpha),
        },
    }
    if args.baseline_model and os.path.exists(args.baseline_model):
        baseline = SimpleModel(INPUT_DIM, HIDDEN, 1, last_layer_activation=None)
        baseline.load_state_dict(safe_load_state_dict(args.baseline_model))
        baseline.eval()
        baseline_metrics, _ = evaluate(
            baseline, payload, x, y, args.eval_batch_size, np.empty(0, dtype=np.int64)
        )
        baseline_val_alpha, baseline_val_norm = normalized_alpha_for_states(
            baseline, x, q_matrices, q_ranges,
            np.asarray(splits["val"], dtype=np.int64), args.eval_batch_size,
        )
        baseline_test_alpha, baseline_test_norm = normalized_alpha_for_states(
            baseline, x, q_matrices, q_ranges,
            np.asarray(splits["test"], dtype=np.int64), args.eval_batch_size,
        )
        baseline_metrics["split_mse"] = {
            "train": subset_loss(baseline, criterion, x, y, train_idx),
            "validation": subset_loss(baseline, criterion, x, y, val_idx),
            "test": subset_loss(baseline, criterion, x, y, test_idx),
        }
        baseline_metrics["split_alpha"] = {
            "validation": {
                "neural_alpha": summarize(baseline_val_alpha),
                "neural_minus_lp_alpha": summarize(baseline_val_alpha - lp_alphas[val_state_indices]),
                "neural_normalized_alpha": summarize(baseline_val_norm),
            },
            "test": {
                "neural_alpha": summarize(baseline_test_alpha),
                "neural_minus_lp_alpha": summarize(baseline_test_alpha - lp_alphas[test_state_indices]),
                "neural_normalized_alpha": summarize(baseline_test_norm),
            },
        }
        metrics["baseline"] = {
            "model": args.baseline_model,
            "potential_prediction_mse": baseline_metrics["potential_prediction_mse"],
            "potential_prediction_correlation": baseline_metrics["potential_prediction_correlation"],
            "neural_alpha": baseline_metrics["neural_alpha"],
            "neural_normalized_alpha": baseline_metrics["neural_normalized_alpha"],
            "attacker_sign_agreement": baseline_metrics["attacker_sign_agreement"],
            "defender_sign_agreement": baseline_metrics["defender_sign_agreement"],
            "alpha_comparison": baseline_metrics["alpha_comparison"],
            "split_mse": baseline_metrics["split_mse"],
            "split_potential_metrics": {},
            "split_alpha": baseline_metrics["split_alpha"],
        }
        baseline_prediction = predict_in_batches(baseline, x, args.eval_batch_size)
        for split_name, split_states in (("train", splits["train"]), ("validation", splits["val"]), ("test", splits["test"])):
            split_examples = expand_state_indices(split_states, n_pairs)
            metrics["baseline"]["split_potential_metrics"][split_name] = split_regression_metrics(
                baseline_prediction[split_examples], target_values[split_examples]
            )
        # Include sign agreement on each held-out split for a direct, aligned
        # comparison with the baseline model.
        for split_name, split_states in (("validation", splits["val"]), ("test", splits["test"])):
            split_state_indices = np.asarray(split_states, dtype=np.int64)
            metrics["split_alpha"][split_name]["attacker_sign_agreement"] = float(
                np.mean(np.asarray(metrics["attacker_sign_agreement_by_state"])[split_state_indices])
            )
            metrics["split_alpha"][split_name]["defender_sign_agreement"] = float(
                np.mean(np.asarray(metrics["defender_sign_agreement_by_state"])[split_state_indices])
            )
            baseline_metrics["split_alpha"][split_name]["attacker_sign_agreement"] = float(
                np.mean(np.asarray(baseline_metrics["attacker_sign_agreement_by_state"])[split_state_indices])
            )
            baseline_metrics["split_alpha"][split_name]["defender_sign_agreement"] = float(
                np.mean(np.asarray(baseline_metrics["defender_sign_agreement_by_state"])[split_state_indices])
            )
            metrics["baseline"]["split_alpha"][split_name] = baseline_metrics["split_alpha"][split_name]
    metrics.pop("attacker_sign_agreement_by_state", None)
    metrics.pop("defender_sign_agreement_by_state", None)
    metrics.update({
        "metadata": {
            "labels": args.labels,
            "model_out": args.model_out,
            "state_encoder": STATE_ENCODER_NAME,
            "input_dim": INPUT_DIM,
            "model_input_schema": MODEL_INPUT_SCHEMA_VERSION,
            "hidden": HIDDEN,
            "output_activation": "linear",
            "batch_norm": "frozen_training_statistics",
            "epochs": args.epochs,
            "eval_only": args.eval_only,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "state_splits": {key: len(value) for key, value in splits.items()},
            "example_splits": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "n_states": n_states,
            "n_examples": len(x),
            "lp_anchor": "Phi[0,0] = 0",
            "lambda_lp": args.lambda_lp,
            "lambda_q": args.lambda_q,
            "baseline_model": args.baseline_model,
            "checkpoint_selection": "lowest validation mean normalized neural alpha",
        },
    })
    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)
    write_state_csv(args.state_csv, state_rows)

    print(f"Saved best potential model: {args.model_out}")
    print(f"Saved training log: {args.log_csv}")
    print(f"Saved metrics: {args.metrics_json}")
    print(f"Saved state alpha diagnostics: {args.state_csv}")
    print(f"Potential target MSE: {metrics['potential_prediction_mse']:.9f}")
    print(f"Potential/LP correlation: {metrics['potential_prediction_correlation']}")
    print(f"Attacker sign agreement: {metrics['attacker_sign_agreement']:.6f}")
    print(f"Defender sign agreement: {metrics['defender_sign_agreement']:.6f}")
    print(f"LP alpha mean/median: {metrics['lp_alpha']['mean']:.9f} / {metrics['lp_alpha']['median']:.9f}")
    print(f"Neural alpha mean/median: {metrics['neural_alpha']['mean']:.9f} / {metrics['neural_alpha']['median']:.9f}")
    print(f"Normalized LP alpha mean/median: {metrics['lp_normalized_alpha']['mean']:.9f} / {metrics['lp_normalized_alpha']['median']:.9f}")
    print(f"Normalized neural alpha mean/median: {metrics['neural_normalized_alpha']['mean']:.9f} / {metrics['neural_normalized_alpha']['median']:.9f}")
    print(f"Validation normalized neural alpha mean/median: {metrics['split_alpha']['validation']['neural_normalized_alpha']['mean']:.9f} / {metrics['split_alpha']['validation']['neural_normalized_alpha']['median']:.9f}")
    print(f"Test normalized neural alpha mean/median: {metrics['split_alpha']['test']['neural_normalized_alpha']['mean']:.9f} / {metrics['split_alpha']['test']['neural_normalized_alpha']['median']:.9f}")
    if "baseline" in metrics:
        baseline = metrics["baseline"]
        print(
            "Baseline test: "
            f"MSE={baseline['split_potential_metrics']['test']['mse']:.9f} "
            f"corr={baseline['split_potential_metrics']['test']['correlation']:.6f} "
            f"norm_alpha_median={baseline['split_alpha']['test']['neural_normalized_alpha']['median']:.6f}"
        )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the V3 one-hot neural potential from LP targets.")
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--lambda-lp", type=float, default=1.0)
    parser.add_argument("--lambda-q", type=float, default=1.0)
    parser.add_argument("--baseline-model", default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--eval-only", action="store_true", help="Evaluate an existing checkpoint without changing it")
    parser.add_argument("--model-out", default=DEFAULT_MODEL)
    parser.add_argument("--log-csv", default=DEFAULT_LOG)
    parser.add_argument("--metrics-json", default=DEFAULT_METRICS)
    parser.add_argument("--state-csv", default=DEFAULT_STATE_CSV)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
