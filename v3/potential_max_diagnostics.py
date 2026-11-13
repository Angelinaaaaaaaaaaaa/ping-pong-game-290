"""Passive Q-regret sidecars for potential_max; no policy or scoring changes."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from v3.potential_policy import potential_max_distributions, potential_max_q_regrets

DEFAULT_REGRET_THRESHOLD = 0.05


def validate_regret_threshold(threshold: float) -> None:
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Potential-max regret threshold must be finite and nonnegative")


def potential_max_regret_rows(
    decisions: list[dict[str, Any]], threshold: float, **context: Any,
) -> list[dict[str, Any]]:
    """One row per decision involving max, even when both roles use it.

    Regrets concern max's proposed full pair. A different opponent policy
    may execute a different component; record the actual actions separately.
    """
    validate_regret_threshold(threshold)
    rows = []
    for decision in decisions:
        if "potential_max" not in (decision["attacker_policy"], decision["defender_policy"]):
            continue
        _, _, pair = potential_max_distributions(decision["potential_matrix"])
        regrets = potential_max_q_regrets(decision["q_matrix"], pair["attack_index"], pair["defense_index"])
        rows.append({
            **context,
            "episode_index": decision["episode_index"],
            "decision_index": decision["decision_index"],
            "attacker_id": decision["attacker_id"],
            "defender_id": decision["defender_id"],
            "attacker_policy": decision["attacker_policy"],
            "defender_policy": decision["defender_policy"],
            "potential_max_attack": pair["attack"],
            "potential_max_defense": pair["defense"],
            "executed_attack": decision["attack_skill"],
            "executed_defense": decision["defense_preposition"],
            **regrets,
            "regret_threshold": threshold,
            "both_regrets_below_threshold": all(value < threshold for value in regrets.values()),
        })
    return rows


def regret_summary(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    validate_regret_threshold(threshold)
    count = len(rows)
    both = sum(row["both_regrets_below_threshold"] for row in rows)
    return {
        "state_count": count,
        "mean_attacker_regret": float(np.mean([r["attacker_regret"] for r in rows])) if count else None,
        "mean_defender_regret": float(np.mean([r["defender_regret"] for r in rows])) if count else None,
        "regret_threshold": threshold,
        "threshold_comparison": "strictly less than; original Q units",
        "both_regrets_below_threshold_count": both,
        "both_regrets_below_threshold_frequency": both / count if count else None,
        "pair_scope": "potential_max proposed joint pair, which may differ from the executed pair against another policy",
    }


def write_regret_diagnostics(output_dir: Path, rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Write separate artifacts, preserving existing evaluator file schemas."""
    summary = regret_summary(rows, threshold)
    # Batch runs carry matchup context. Report each separately as well as pooled.
    group_keys = ["tested_policy", "opponent_type", "opponent_policy", "fixed_attack_skill",
                  "fixed_defense_position", "seed", "side_assignment"]
    if rows and "tested_policy" in rows[0]:
        groups: dict[tuple, list] = {}
        for row in rows:
            key = tuple(row[name] for name in group_keys)
            groups.setdefault(key, []).append(row)
        summary["matchups"] = [
            {**dict(zip(group_keys, key)), **regret_summary(group, threshold)}
            for key, group in groups.items()
        ]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "potential_max_regrets.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["episode_index", "decision_index"])
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "potential_max_regret_summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    return summary
