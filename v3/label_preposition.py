#!/usr/bin/env python3
"""Label V3 attacker-skill x defender-pre-position decisions for Q training."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle as pkl
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nash_skills.skills import SKILL_NAMES, skill_index  # noqa: E402
from nash_skills.v2.labeling import GAMMA  # noqa: E402
from v3.constants import DEFENSE_POSITIONS  # noqa: E402
from v3.state_encoder import STATE_DIM, STATE_ENCODER_NAME  # noqa: E402


def defense_index(position: str) -> int:
    return DEFENSE_POSITIONS.index(position)


def winner_label_from_value(value: Any) -> int:
    if value in (1, "1", "ego", "p1", "P1"):
        return 1
    if value in (2, "2", "opp", "p2", "P2"):
        return 2
    return 0


def discounted_returns_from_attackers(attackers: list[int], winner: int, gamma: float) -> list[float]:
    if winner not in (1, 2):
        return [0.0 for _ in attackers]
    last = len(attackers) - 1
    returns = []
    for index, attacker in enumerate(attackers):
        terminal = 1.0 if int(attacker) == winner else -1.0
        returns.append((gamma ** (last - index)) * terminal)
    return returns


def load_state(decision: dict[str, Any], fallback_state: Any) -> np.ndarray:
    value = decision.get("state", fallback_state)
    if isinstance(value, str):
        value = json.loads(value)
    state = np.asarray(value, dtype=np.float32).reshape(-1)
    if state.shape[0] != STATE_DIM:
        raise ValueError(f"Expected {STATE_DIM}-dim state, got {state.shape[0]}")
    return state


def examples_from_rally(rally: dict[str, Any], gamma: float) -> list[dict[str, Any]]:
    if rally.get("state_encoder") != STATE_ENCODER_NAME:
        raise ValueError(
            "Input data does not use the current canonical V3 state encoder: "
            f"expected {STATE_ENCODER_NAME!r}, got {rally.get('state_encoder')!r}. "
            "Recollect V3 data before labeling."
        )
    decisions = list(rally.get("decisions") or [])
    states = list(rally.get("states") or [])
    if not decisions and states:
        raise ValueError(
            "V3 labeling requires per-decision metadata. "
            "Use a pickle produced by v3/collect_random_preposition.py."
        )

    winner = winner_label_from_value(rally.get("winner", rally.get("final_rally_winner", 0)))
    attackers = [int(decision["attacker_id"]) for decision in decisions]
    fallback_returns = discounted_returns_from_attackers(attackers, winner, gamma)
    stored_returns = list(rally.get("attacker_returns") or [])

    examples = []
    for index, decision in enumerate(decisions):
        attack_skill = str(decision["attack_skill"])
        defense_position = str(decision["defense_preposition"])
        if decision.get("state_encoder") != STATE_ENCODER_NAME:
            raise ValueError(
                f"Rally {rally.get('rally_id', '')} decision {index + 1} has an incompatible state encoder"
            )
        if attack_skill not in SKILL_NAMES:
            raise ValueError(f"Unknown attack skill {attack_skill!r}")
        if defense_position not in DEFENSE_POSITIONS:
            raise ValueError(f"Unknown defense pre-position {defense_position!r}")

        fallback_state = states[index] if index < len(states) else None
        state = load_state(decision, fallback_state)
        if "discounted_return" in decision:
            attacker_return = float(decision["discounted_return"])
        elif index < len(stored_returns):
            attacker_return = float(stored_returns[index])
        else:
            attacker_return = float(fallback_returns[index])

        examples.append({
            "state": state,
            "attack_skill": attack_skill,
            "attack_skill_idx": skill_index(attack_skill),
            "defense_preposition": defense_position,
            "defense_preposition_idx": defense_index(defense_position),
            "attacker_id": int(decision["attacker_id"]),
            "defender_id": int(decision["defender_id"]),
            "attacker_return": attacker_return,
            "rally_id": int(decision.get("rally_id", rally.get("rally_id", -1))),
            "decision_index": int(decision.get("decision_index", index + 1)),
            "winner": winner,
            "final_rally_winner": decision.get("final_rally_winner", rally.get("final_rally_winner", "")),
            "termination_reason": decision.get("termination_reason", rally.get("termination_reason", "")),
            "truncated": bool(decision.get("truncated", rally.get("truncated", False))),
        })
    return examples


def label(input_pkl: str, output_pkl: str, output_csv: str, gamma: float) -> list[dict[str, Any]]:
    with open(input_pkl, "rb") as f:
        rallies = pkl.load(f)

    examples: list[dict[str, Any]] = []
    for rally in rallies:
        examples.extend(examples_from_rally(rally, gamma))

    Path(output_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, "wb") as f:
        pkl.dump({
            "examples": examples,
            "metadata": {
                "source": input_pkl,
                "gamma": gamma,
                "state_dim": STATE_DIM,
                "state_encoder": STATE_ENCODER_NAME,
                "attack_skills": list(SKILL_NAMES),
                "defense_prepositions": list(DEFENSE_POSITIONS),
            },
        }, f)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        fields = [
            "rally_id",
            "decision_index",
            "attacker_id",
            "defender_id",
            "attack_skill",
            "attack_skill_idx",
            "defense_preposition",
            "defense_preposition_idx",
            "attacker_return",
            "winner",
            "final_rally_winner",
            "termination_reason",
            "truncated",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for example in examples:
            writer.writerow({field: example[field] for field in fields})

    counts = Counter((ex["attack_skill"], ex["defense_preposition"]) for ex in examples)
    print(f"Loaded rallies: {len(rallies)}")
    print(f"Labeled examples: {len(examples)}")
    print(f"State dim: {STATE_DIM}, gamma={gamma}")
    print("Coverage:")
    for skill in SKILL_NAMES:
        for position in DEFENSE_POSITIONS:
            print(f"  {skill:12s} x {position:6s}: {counts[(skill, position)]}")
    print(f"Saved labels: {output_pkl}")
    print(f"Saved label summary CSV: {output_csv}")
    return examples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Label V3 pre-position decisions for attacker-perspective Q training.")
    parser.add_argument("--input-pkl", default="data/v3/random_preposition_canonical_decisions.pkl")
    parser.add_argument("--output-pkl", default="data/v3/labeled_preposition_canonical_examples.pkl")
    parser.add_argument("--output-csv", default="skill_eval/v3/labeled_preposition_canonical_examples.csv")
    parser.add_argument("--gamma", type=float, default=GAMMA)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    label(args.input_pkl, args.output_pkl, args.output_csv, args.gamma)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
