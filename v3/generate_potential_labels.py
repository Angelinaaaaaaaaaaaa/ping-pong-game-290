#!/usr/bin/env python3
"""Generate supervised V3 potential targets from per-state LP solutions.

For every canonical state and every one of the 15 attack/defense pairs, this
script stores one training target:

    (canonical_state, attack_one_hot, defense_one_hot) -> Phi_s(a, d)

The LP convention is implemented by the shared V3 potential LP module and fixes
Phi[0, 0] = 0 for every state.  The payload also keeps the Q matrices, LP
potential matrices, alphas, and Q ranges needed for downstream diagnostics.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nash_skills.skills import SKILL_NAMES  # noqa: E402
from v3.constants import DEFENSE_POSITIONS  # noqa: E402
from v3.potential_lp import (  # noqa: E402
    predict_q_matrices,
    solve_alpha_potential,
)
from v3.q_policy import load_q_model  # noqa: E402
from v3.model_input import MODEL_INPUT_DIM, MODEL_INPUT_SCHEMA_VERSION  # noqa: E402
from v3.state_encoder import STATE_DIM, STATE_ENCODER_NAME  # noqa: E402
from v3.train_q_preposition_onehot import load_examples  # noqa: E402


DEFAULT_LABELS = "data/v3/labeled_preposition_state74_examples.pkl"
DEFAULT_MODEL = "models/v3/q_preposition_state74_onehot.pth"
DEFAULT_OUTPUT = "data/v3/potential_labels_state74.pkl"
N_ACTION_PAIRS = len(SKILL_NAMES) * len(DEFENSE_POSITIONS)


def generate_payload(
    labels_path: str,
    model_path: str,
    batch_size: int = 512,
) -> dict:
    examples = load_examples(labels_path)
    if not examples:
        raise ValueError(f"No canonical examples found in {labels_path}")

    states = np.asarray([example["state"] for example in examples], dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != STATE_DIM:
        raise ValueError(f"Expected states with shape (N, {STATE_DIM}), got {states.shape}")

    model = load_q_model(model_path, device="cpu")
    q_matrices = predict_q_matrices(model, states, batch_size=batch_size)
    fits = [solve_alpha_potential(q) for q in q_matrices]
    phi_matrices = np.asarray([fit.phi for fit in fits], dtype=np.float32)
    lp_alphas = np.asarray([fit.alpha for fit in fits], dtype=np.float32)
    q_ranges = np.asarray([fit.q_range for fit in fits], dtype=np.float32)

    # Pair order is state-major, then attack-major, then defense-major.  This
    # makes every state's 15 targets contiguous and easy to reshape later.
    attack_indices = np.tile(np.repeat(np.arange(len(SKILL_NAMES)), len(DEFENSE_POSITIONS)), len(states))
    defense_indices = np.tile(np.tile(np.arange(len(DEFENSE_POSITIONS)), len(SKILL_NAMES)), len(states))
    state_indices = np.repeat(np.arange(len(states)), N_ACTION_PAIRS)
    pair_states = np.repeat(states, N_ACTION_PAIRS, axis=0)
    targets = phi_matrices.reshape(-1).astype(np.float32)

    payload = {
        "metadata": {
            "task": "v3_onehot_alpha_potential_supervised_targets",
            "source_labels": labels_path,
            "source_q_model": model_path,
            "state_encoder": STATE_ENCODER_NAME,
            "state_dim": STATE_DIM,
            "input_dim": MODEL_INPUT_DIM,
            "model_input_schema": MODEL_INPUT_SCHEMA_VERSION,
            "attack_actions": list(SKILL_NAMES),
            "defense_actions": list(DEFENSE_POSITIONS),
            "potential_anchor": "Phi[0,0] = 0",
            "n_states": len(states),
            "n_action_pairs_per_state": N_ACTION_PAIRS,
            "n_examples": len(pair_states),
            "pair_order": "state, attack, defense",
        },
        "examples": {
            "states": pair_states,
            "state_indices": state_indices,
            "attack_indices": attack_indices,
            "defense_indices": defense_indices,
            "potential_targets": targets,
        },
        "state_solutions": {
            "states": states,
            "q_matrices": q_matrices.astype(np.float32),
            "phi_matrices": phi_matrices,
            "lp_alphas": lp_alphas,
            "q_ranges": q_ranges,
        },
    }
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate all 15 V3 LP-potential targets per canonical state."
    )
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = generate_payload(args.labels, args.model, args.batch_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = payload["metadata"]
    print(f"Loaded canonical states: {metadata['n_states']}")
    print(f"Generated potential examples: {metadata['n_examples']}")
    print(f"Saved potential labels: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
