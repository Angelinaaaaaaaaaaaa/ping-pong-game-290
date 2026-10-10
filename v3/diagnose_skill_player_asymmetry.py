#!/usr/bin/env python3
"""Diagnose skill asymmetry separately from physical-player asymmetry.

This diagnostic uses the existing labeled V3 dataset only.  It summarizes
attacker-perspective returns by physical attacker and compares mirrored skill
pairs for P1 and P2, both overall and conditional on defender pre-position.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle as pkl
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nash_skills.skills import SKILL_NAMES  # noqa: E402
from v3.constants import DEFENSE_POSITIONS  # noqa: E402


MIRRORED_SKILL_PAIRS = (
    ("left", "right"),
    ("left_short", "right_short"),
)


def load_examples(path: str) -> list[dict[str, Any]]:
    """Load labeled examples from the existing V3 label-pickle formats."""
    with Path(path).open("rb") as f:
        payload = pkl.load(f)

    if isinstance(payload, dict) and "examples" in payload:
        examples = list(payload["examples"])
    elif isinstance(payload, list):
        examples = list(payload)
    else:
        raise ValueError(f"Unsupported labeled V3 pickle format: {type(payload)}")

    required = {"attack_skill", "defense_preposition", "attacker_id", "attacker_return"}
    missing = required.difference(examples[0]) if examples else required
    if missing:
        raise ValueError(f"Labeled examples are missing required fields: {sorted(missing)}")
    return examples


def validate_examples(examples: Iterable[dict[str, Any]]) -> None:
    for index, example in enumerate(examples):
        skill = str(example["attack_skill"])
        position = str(example["defense_preposition"])
        attacker_id = int(example["attacker_id"])
        if skill not in SKILL_NAMES:
            raise ValueError(f"Example {index} has unknown attack_skill {skill!r}")
        if position not in DEFENSE_POSITIONS:
            raise ValueError(f"Example {index} has unknown defense_preposition {position!r}")
        if attacker_id not in (1, 2):
            raise ValueError(f"Example {index} has unknown attacker_id {attacker_id!r}")


def metrics(values: list[float]) -> dict[str, float | int | str]:
    """Return the requested descriptive statistics for one group."""
    if not values:
        return {
            "count": 0,
            "mean_attacker_return": "",
            "median_attacker_return": "",
            "std_attacker_return": "",
            "attacker_win_rate": "",
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean_attacker_return": float(np.mean(arr)),
        "median_attacker_return": float(np.median(arr)),
        "std_attacker_return": float(np.std(arr)),
        "attacker_win_rate": float(np.mean(arr > 0.0)),
    }


def grouped_values(
    examples: Iterable[dict[str, Any]],
    *,
    include_defense_position: bool,
) -> dict[tuple[Any, ...], list[float]]:
    groups: dict[tuple[Any, ...], list[float]] = {}
    for example in examples:
        key = (
            str(example["attack_skill"]),
            *(  # Keep the grouping order explicit in the output schema.
                [str(example["defense_preposition"])]
                if include_defense_position
                else []
            ),
            int(example["attacker_id"]),
        )
        groups.setdefault(key, []).append(float(example["attacker_return"]))
    return groups


def write_group_summary(
    path: str,
    examples: list[dict[str, Any]],
    *,
    include_defense_position: bool,
) -> None:
    groups = grouped_values(examples, include_defense_position=include_defense_position)
    rows: list[dict[str, Any]] = []
    if include_defense_position:
        fields = ["attack_skill", "defense_preposition", "attacker_id"]
        ordered_keys = [
            (skill, position, attacker_id)
            for skill in SKILL_NAMES
            for position in DEFENSE_POSITIONS
            for attacker_id in (1, 2)
        ]
    else:
        fields = ["attack_skill", "attacker_id"]
        ordered_keys = [
            (skill, attacker_id)
            for skill in SKILL_NAMES
            for attacker_id in (1, 2)
        ]

    metric_fields = [
        "count",
        "mean_attacker_return",
        "median_attacker_return",
        "std_attacker_return",
        "attacker_win_rate",
    ]
    for key in ordered_keys:
        values = groups.get(key, [])
        row = dict(zip(fields, key))
        row.update(metrics(values))
        rows.append(row)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[*fields, *metric_fields])
        writer.writeheader()
        writer.writerows(rows)


def mirrored_comparison_rows(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build side-by-side P1/P2 comparisons for both mirrored skill pairs."""
    rows: list[dict[str, Any]] = []
    positions = ("all", *DEFENSE_POSITIONS)

    def select(skill: str, attacker_id: int, position: str) -> list[float]:
        return [
            float(example["attacker_return"])
            for example in examples
            if str(example["attack_skill"]) == skill
            and int(example["attacker_id"]) == attacker_id
            and (
                position == "all"
                or str(example["defense_preposition"]) == position
            )
        ]

    for left_skill, right_skill in MIRRORED_SKILL_PAIRS:
        for position in positions:
            row: dict[str, Any] = {
                "skill_pair": f"{left_skill}_vs_{right_skill}",
                "left_skill": left_skill,
                "right_skill": right_skill,
                "defense_preposition": position,
            }
            player_preferences: list[str] = []
            mean_differences: dict[int, float] = {}

            for attacker_id in (1, 2):
                left = metrics(select(left_skill, attacker_id, position))
                right = metrics(select(right_skill, attacker_id, position))
                prefix = f"p{attacker_id}"
                for side, summary in (("left", left), ("right", right)):
                    for field in (
                        "count",
                        "mean_attacker_return",
                        "median_attacker_return",
                        "std_attacker_return",
                        "attacker_win_rate",
                    ):
                        row[f"{prefix}_{side}_{field}"] = summary[field]

                left_mean = left["mean_attacker_return"]
                right_mean = right["mean_attacker_return"]
                if left_mean == "" or right_mean == "":
                    preference = "insufficient_data"
                    difference: float | str = ""
                else:
                    difference = float(left_mean) - float(right_mean)
                    mean_differences[attacker_id] = difference
                    if difference > 0.0:
                        preference = left_skill
                    elif difference < 0.0:
                        preference = right_skill
                    else:
                        preference = "tie"
                row[f"{prefix}_left_minus_right_mean"] = difference
                row[f"{prefix}_preferred_skill"] = preference
                player_preferences.append(preference)

            if len(player_preferences) == 2 and all(
                preference in {left_skill, right_skill} for preference in player_preferences
            ):
                same = player_preferences[0] == player_preferences[1]
                interpretation = (
                    "same_skill_preferred_for_both_players"
                    if same
                    else "preference_flips_between_players"
                )
            else:
                same = ""
                interpretation = "insufficient_data_or_tie"

            row["same_preference_for_p1_p2"] = same
            row["interpretation"] = interpretation
            if 1 in mean_differences and 2 in mean_differences:
                row["p1_minus_p2_difference_gap"] = (
                    mean_differences[1] - mean_differences[2]
                )
            else:
                row["p1_minus_p2_difference_gap"] = ""
            rows.append(row)
    return rows


def write_mirrored_comparison(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No mirrored-skill comparison rows were generated")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose V3 skill asymmetry by physical attacker and defender position."
    )
    parser.add_argument(
        "--labels",
        default="data/v3/labeled_preposition_5k.pkl",
        help="Existing labeled V3 dataset; no data is collected or modified.",
    )
    parser.add_argument(
        "--by-skill-player-csv",
        default="skill_eval/v3/skill_player_asymmetry.csv",
    )
    parser.add_argument(
        "--by-skill-defense-player-csv",
        default="skill_eval/v3/skill_defense_player_asymmetry.csv",
    )
    parser.add_argument(
        "--mirrored-comparison-csv",
        default="skill_eval/v3/mirrored_skill_player_comparison.csv",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    examples = load_examples(args.labels)
    validate_examples(examples)
    if not examples:
        raise ValueError(f"No labeled examples found in {args.labels}")

    write_group_summary(
        args.by_skill_player_csv,
        examples,
        include_defense_position=False,
    )
    write_group_summary(
        args.by_skill_defense_player_csv,
        examples,
        include_defense_position=True,
    )
    mirrored_rows = mirrored_comparison_rows(examples)
    write_mirrored_comparison(args.mirrored_comparison_csv, mirrored_rows)

    print(f"Loaded labeled examples: {len(examples)}")
    print(f"Wrote: {args.by_skill_player_csv}")
    print(f"Wrote: {args.by_skill_defense_player_csv}")
    print(f"Wrote: {args.mirrored_comparison_csv}")
    print("\nMirrored-skill interpretation:")
    for row in mirrored_rows:
        if row["defense_preposition"] == "all":
            print(
                f"  {row['skill_pair']}: {row['interpretation']} "
                f"(P1={row['p1_preferred_skill']}, P2={row['p2_preferred_skill']})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
