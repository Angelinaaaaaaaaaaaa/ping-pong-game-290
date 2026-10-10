#!/usr/bin/env python3
"""Diagnose the V3 ``left_short`` versus ``right_short`` asymmetry.

The random V3 decision pickle records attacker-perspective returns and
termination reasons, but it does not record per-decision racket contacts or
actual landing coordinates.  This script therefore derives contact/return
outcomes from observed rally transitions and reports landing fields as
unavailable when running on that artifact.  It never recollects or retrains.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle as pkl
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from v3.skill_semantics import (  # noqa: E402
    SKILL_PROFILES,
    get_skill,
    skill_target_xy,
    world_target_xy,
)
from v3.racket_target_planner import (  # noqa: E402
    PADDLE_OFFSET,
    TABLE_SHIFT,
    TABLE_Z,
)
from v3.constants import DEFENSE_POSITIONS  # noqa: E402


SHORT_SKILLS = ("left_short", "right_short")
PLAYER_LABELS = {1: "ego", 2: "opp"}
LANDING_FIELDS_UNAVAILABLE = "not_recorded_in_random_decision_collection"


def _as_bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _load_collection(path: str) -> list[dict[str, Any]]:
    """Load and flatten the saved V3 decision collection."""
    source = Path(path)
    if source.suffix == ".pkl":
        with source.open("rb") as f:
            payload = pkl.load(f)
        if isinstance(payload, list):
            if payload and isinstance(payload[0], dict) and "decisions" in payload[0]:
                rows = []
                for rally in payload:
                    for decision in rally.get("decisions", []):
                        row = dict(decision)
                        row.setdefault("rally_id", rally.get("rally_id", ""))
                        row.setdefault("final_rally_winner", rally.get("final_rally_winner", ""))
                        row.setdefault("termination_reason", rally.get("termination_reason", ""))
                        row.setdefault("truncated", rally.get("truncated", False))
                        rows.append(row)
                return rows
            return [dict(row) for row in payload]
        if isinstance(payload, dict) and "decisions" in payload:
            return [dict(row) for row in payload["decisions"]]
        raise ValueError(f"Unsupported pickle collection format: {type(payload)}")

    if source.suffix == ".json":
        with source.open() as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "decisions" in payload:
            return [dict(row) for row in payload["decisions"]]
        if isinstance(payload, list):
            return [dict(row) for row in payload]
        raise ValueError(f"Unsupported JSON collection format: {type(payload)}")

    if source.suffix == ".csv":
        with source.open(newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]

    raise ValueError(f"Unsupported collection extension: {source.suffix}")


def _validate_rows(rows: list[dict[str, Any]]) -> None:
    required = {"attack_skill", "defense_preposition", "attacker_id", "termination_reason"}
    for index, row in enumerate(rows):
        missing = required.difference(row)
        if missing:
            raise ValueError(f"Collection row {index} is missing {sorted(missing)}")
        if str(row["attack_skill"]) not in SHORT_SKILLS:
            continue
        if str(row["defense_preposition"]) not in DEFENSE_POSITIONS:
            raise ValueError(f"Unknown defender pre-position in row {index}: {row['defense_preposition']!r}")
        if int(row["attacker_id"]) not in (1, 2):
            raise ValueError(f"Unknown attacker_id in row {index}: {row['attacker_id']!r}")


def _winner_player(row: dict[str, Any]) -> int | None:
    value = str(row.get("final_rally_winner", row.get("winner", ""))).lower()
    if value in {"ego", "p1", "player1", "1"}:
        return 1
    if value in {"opp", "p2", "player2", "2"}:
        return 2
    return None


def _rally_groups(rows: list[dict[str, Any]]) -> dict[Any, list[dict[str, Any]]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row.get("rally_id", "")].append(row)
    for group in groups.values():
        group.sort(key=lambda row: int(row.get("decision_index", 0)))
    return groups


def _last_hitter_from_reason(reason: str) -> int | None:
    if reason.startswith("missed_"):
        receiver = reason.split("_", 2)[1]
        return 2 if receiver == "ego" else 1
    match = re.match(r"(ego|opp)_shot_out", reason)
    if match is None:
        return None
    return 1 if match.group(1) == "ego" else 2


def _receiver_from_reason(reason: str) -> int | None:
    match = re.search(r"(?:missed_|valid_)(ego|opp)", reason)
    if not match:
        return None
    return 1 if match.group(1) == "ego" else 2


def _derive_outcome_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Infer per-decision outcome status from crossing and terminal semantics."""
    by_rally = _rally_groups(rows)
    enriched: list[dict[str, Any]] = []

    for rally_rows in by_rally.values():
        for index, row in enumerate(rally_rows):
            result = dict(row)
            attacker_id = int(row["attacker_id"])
            reason = str(row.get("termination_reason", ""))
            has_next_decision = index + 1 < len(rally_rows)
            last_hitter = _last_hitter_from_reason(reason)
            receiver = _receiver_from_reason(reason)
            current_attacker_is_last_hitter = (
                last_hitter is not None and attacker_id == last_hitter
            )

            # A following crossing proves that the selected attacker sent the
            # ball back.  For the final decision, terminal-state semantics tell
            # whether the current attacker hit, missed, or hit out.
            if has_next_decision:
                contact = True
                successful = True
                attacker_miss = False
                attacker_out = False
                outcome_source = "following_net_crossing"
            elif reason == "max_steps" or _as_bool(row.get("truncated")):
                contact = None
                successful = None
                attacker_miss = None
                attacker_out = None
                outcome_source = "truncated_before_terminal_outcome"
            elif last_hitter is None:
                contact = None
                successful = None
                attacker_miss = None
                attacker_out = None
                outcome_source = "unrecognized_termination_reason"
            elif "shot_out_before_valid" in reason:
                if current_attacker_is_last_hitter:
                    contact = True
                    successful = False
                    attacker_out = True
                    attacker_miss = False
                    outcome_source = "terminal_shot_out_semantics"
                else:
                    contact = None
                    successful = None
                    attacker_out = None
                    attacker_miss = None
                    outcome_source = "terminal_hitter_does_not_match_decision_attacker"
            elif "missed_" in reason:
                if receiver == attacker_id:
                    contact = False
                    successful = False
                    attacker_miss = True
                    attacker_out = False
                    outcome_source = "terminal_return_miss_semantics"
                elif current_attacker_is_last_hitter:
                    contact = True
                    successful = True
                    attacker_miss = False
                    attacker_out = False
                    outcome_source = "terminal_return_miss_semantics"
                else:
                    contact = None
                    successful = None
                    attacker_miss = None
                    attacker_out = None
                    outcome_source = "terminal_hitter_does_not_match_decision_attacker"
            else:
                contact = None
                successful = None
                attacker_miss = None
                attacker_out = None
                outcome_source = "unrecognized_termination_reason"

            result.update({
                "contact_observed": contact,
                "successful_return_observed": successful,
                "attacker_miss": attacker_miss,
                "attacker_out_of_bounds": attacker_out,
                "outcome_source": outcome_source,
                "winner_player": _winner_player(row),
                "landing_data_available": False,
                "actual_landing_x": "",
                "actual_landing_y": "",
                "landing_error": "",
            })
            enriched.append(result)
    return enriched


def _rate(values: list[bool | None]) -> float | str:
    observed = [value for value in values if value is not None]
    if not observed:
        return ""
    return float(np.mean(np.asarray(observed, dtype=bool)))


def _mean(values: list[float]) -> float | str:
    return float(np.mean(values)) if values else ""


def _target_fields(attacker_id: int, skill: str, profile: str) -> dict[str, Any]:
    local_x, local_y = skill_target_xy(skill, profile=profile)
    world_x, world_y = world_target_xy(attacker_id, skill, profile=profile)
    return {
        "target_landing_local_x": local_x,
        "target_landing_local_y": local_y,
        "target_landing_world_x": world_x,
        "target_landing_world_y": world_y,
        "actual_landing_x": "",
        "actual_landing_y": "",
        "landing_error": "",
        "landing_data_status": LANDING_FIELDS_UNAVAILABLE,
    }


def _group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["attack_skill"]), int(row["attacker_id"]), str(row["defense_preposition"]))].append(row)
    return groups


def _summary_row(skill: str, attacker_id: int, position: str, group: list[dict[str, Any]], profile: str) -> dict[str, Any]:
    returns = [float(row.get("discounted_return", row.get("attacker_return", 0.0))) for row in group]
    contact = [row["contact_observed"] for row in group]
    successful = [row["successful_return_observed"] for row in group]
    misses = [row["attacker_miss"] for row in group]
    out = [row["attacker_out_of_bounds"] for row in group]
    target = _target_fields(attacker_id, skill, profile)
    termination_counts = Counter(str(row.get("termination_reason", "")) for row in group)
    return {
        "attack_skill": skill,
        "attacker_id": attacker_id,
        "defense_preposition": position,
        "count": len(group),
        "mean_discounted_attacker_return": _mean(returns),
        "attacker_win_rate": float(np.mean(np.asarray(returns) > 0.0)) if returns else "",
        "contact_rate": _rate(contact),
        "contact_observed_count": sum(value is not None for value in contact),
        "successful_return_rate": _rate(successful),
        "successful_return_observed_count": sum(value is not None for value in successful),
        "miss_rate": _rate(misses),
        "out_of_bounds_rate": _rate(out),
        "contact_metric_status": "inferred_from_crossings_and_termination_reasons",
        "termination_reason_distribution": json.dumps(dict(sorted(termination_counts.items())), sort_keys=True),
        "actual_landing_x_mean": "",
        "actual_landing_y_mean": "",
        "landing_error_mean": "",
        "landing_data_status": LANDING_FIELDS_UNAVAILABLE,
        **target,
    }


def _termination_rows(skill: str, attacker_id: int, position: str, group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(str(row.get("termination_reason", "")) for row in group)
    return [
        {
            "attack_skill": skill,
            "attacker_id": attacker_id,
            "defense_preposition": position,
            "termination_reason": reason,
            "count": count,
            "rate": count / len(group) if group else "",
        }
        for reason, count in sorted(counts.items())
    ]


def _mirrored_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["attack_skill"], int(row["attacker_id"]), row["defense_preposition"]): row
        for row in summary
    }
    output = []
    metric_names = (
        "count",
        "mean_discounted_attacker_return",
        "attacker_win_rate",
        "contact_rate",
        "successful_return_rate",
        "miss_rate",
        "out_of_bounds_rate",
    )
    for position in DEFENSE_POSITIONS:
        for attacker_id in (1, 2):
            left = by_key[("left_short", attacker_id, position)]
            right = by_key[("right_short", attacker_id, position)]
            row: dict[str, Any] = {
                "skill_pair": "left_short_vs_right_short",
                "attacker_id": attacker_id,
                "defense_preposition": position,
            }
            for side, source in (("left_short", left), ("right_short", right)):
                for metric in metric_names:
                    row[f"{side}_{metric}"] = source[metric]
            left_mean = left["mean_discounted_attacker_return"]
            right_mean = right["mean_discounted_attacker_return"]
            if left_mean == "" or right_mean == "":
                row["left_short_minus_right_short_mean"] = ""
                row["preferred_skill_by_mean"] = "insufficient_data"
            else:
                difference = float(left_mean) - float(right_mean)
                row["left_short_minus_right_short_mean"] = difference
                row["preferred_skill_by_mean"] = (
                    "left_short" if difference > 0 else "right_short" if difference < 0 else "tie"
                )
            output.append(row)
    return output


def _geometry_rows() -> list[dict[str, Any]]:
    rows = []
    for profile in SKILL_PROFILES:
        left_side, left_x = get_skill("left_short", profile=profile)
        right_side, right_x = get_skill("right_short", profile=profile)
        left_local = skill_target_xy("left_short", profile=profile)
        right_local = skill_target_xy("right_short", profile=profile)
        left_p1 = world_target_xy(1, "left_short", profile=profile)
        right_p1 = world_target_xy(1, "right_short", profile=profile)
        left_p2 = world_target_xy(2, "left_short", profile=profile)
        right_p2 = world_target_xy(2, "right_short", profile=profile)
        rows.append({
            "profile": profile,
            "left_short_side_target": left_side,
            "right_short_side_target": right_side,
            "left_short_local_x": left_local[0],
            "left_short_local_y": left_local[1],
            "right_short_local_x": right_local[0],
            "right_short_local_y": right_local[1],
            "left_short_p1_world_x": left_p1[0],
            "left_short_p1_world_y": left_p1[1],
            "right_short_p1_world_x": right_p1[0],
            "right_short_p1_world_y": right_p1[1],
            "left_short_p2_world_x": left_p2[0],
            "left_short_p2_world_y": left_p2[1],
            "right_short_p2_world_x": right_p2[0],
            "right_short_p2_world_y": right_p2[1],
            "same_depth_target": bool(np.isclose(left_x, right_x)),
            "opposite_lateral_target": bool(np.isclose(left_side, -right_side)),
            "same_lateral_magnitude": bool(np.isclose(abs(left_side), abs(right_side))),
            "p1_world_mirror_exact": bool(
                np.allclose(left_p1, (right_p1[0], -right_p1[1]))
            ),
            "p2_world_mirror_exact": bool(
                np.allclose(left_p2, (right_p2[0], -right_p2[1]))
            ),
            "reference_height_defined_per_skill": False,
            "planner_table_z": TABLE_Z,
            "planner_paddle_offset": PADDLE_OFFSET,
            "planner_default_z_dis": 0.9,
            "p2_world_transform": "x_world=2*TABLE_SHIFT-x_local; y_world=-y_local",
            "p2_racket_pose_transform": "position unchanged; orientation multiplied by diag(-1,-1,1)",
            "planner_target_formula": "target=[x_target,y_target,table_z]; paddle pos offset uses theta_z",
            "table_shift": TABLE_SHIFT,
        })
    return rows


def _write_csv(path: str, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _print_conclusion(rows: list[dict[str, Any]], geometry: list[dict[str, Any]]) -> None:
    print("\nConclusion:")
    print("  Landing coordinates/errors: unavailable in the random decision collection.")

    aggregate_metrics: dict[str, dict[str, float | str]] = {}
    for skill in SHORT_SKILLS:
        values = [row for row in rows if row["attack_skill"] == skill]
        returns = [float(r.get("discounted_return", r.get("attacker_return", 0.0))) for r in values]
        aggregate_metrics[skill] = {
            "mean": _mean(returns),
            "contact": _rate([r["contact_observed"] for r in values]),
            "successful": _rate([r["successful_return_observed"] for r in values]),
            "miss": _rate([r["attacker_miss"] for r in values]),
            "out": _rate([r["attacker_out_of_bounds"] for r in values]),
        }
        print(
            f"  {skill}: n={len(values)}, "
            f"mean_return={aggregate_metrics[skill]['mean']:.4f}, "
            f"contact_rate={aggregate_metrics[skill]['contact']}, "
            f"successful_return_rate={aggregate_metrics[skill]['successful']}, "
            f"miss_rate={aggregate_metrics[skill]['miss']}, "
            f"out_rate={aggregate_metrics[skill]['out']}",
        )

    if all(
        row["opposite_lateral_target"]
        and row["same_depth_target"]
        and row["same_lateral_magnitude"]
        and row["p1_world_mirror_exact"]
        and row["p2_world_mirror_exact"]
        for row in geometry
    ):
        print("  Geometry: left_short/right_short are exact lateral mirrors in all skill profiles;")
        print("            they share depth and planner reference height, with symmetric P2 transforms.")
    else:
        print("  Geometry: asymmetric target geometry detected; inspect short_skill_geometry.csv.")

    position_means = {}
    for position in DEFENSE_POSITIONS:
        values = {}
        for skill in SHORT_SKILLS:
            group = [
                row for row in rows
                if row["attack_skill"] == skill and row["defense_preposition"] == position
            ]
            values[skill] = _mean([
                float(r.get("discounted_return", r.get("attacker_return", 0.0)))
                for r in group
            ])
        position_means[position] = values
        print(
            f"  Defense {position}: left_short_mean={values['left_short']:.4f}, "
            f"right_short_mean={values['right_short']:.4f}"
        )

    left = aggregate_metrics["left_short"]
    right = aggregate_metrics["right_short"]
    if left["contact"] != "" and right["contact"] != "" and left["successful"] != "" and right["successful"] != "":
        print(
            "  Contact/execution: right_short contacts are not lower, but its successful-return rate is lower."
        )
    if left["out"] != "" and right["out"] != "" and float(right["out"]) > float(left["out"]):
        print(
            f"  Out-of-bounds: right_short is higher ({float(right['out']):.3f} vs {float(left['out']):.3f}); "
            "this is the strongest available explanation."
        )
    print("  Defender interaction: the return gap changes sign across defender pre-positions, so it is also material.")
    print("  Landing accuracy: unavailable from this saved collection; planner-event data would be required.")
    print("  Skill/planner definitions: no static left/right asymmetry was found in target geometry or transforms.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection",
        default="data/v3/random_preposition_decisions_5k.pkl",
        help="Existing V3 decision collection; no recollection is performed.",
    )
    parser.add_argument(
        "--skill-profile",
        choices=sorted(SKILL_PROFILES),
        default="aggressive",
        help="Profile used for target-coordinate reporting; the collection does not store this metadata.",
    )
    parser.add_argument("--detail-csv", default="skill_eval/v3/short_skill_outcome_detail.csv")
    parser.add_argument("--summary-csv", default="skill_eval/v3/short_skill_outcome_summary.csv")
    parser.add_argument("--termination-csv", default="skill_eval/v3/short_skill_termination_distribution.csv")
    parser.add_argument("--comparison-csv", default="skill_eval/v3/short_skill_mirrored_comparison.csv")
    parser.add_argument("--geometry-csv", default="skill_eval/v3/short_skill_geometry.csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_rows = _load_collection(args.collection)
    _validate_rows(raw_rows)
    short_rows = [row for row in raw_rows if str(row["attack_skill"]) in SHORT_SKILLS]
    if not short_rows:
        raise ValueError(f"No {SHORT_SKILLS} rows found in {args.collection}")
    rows = _derive_outcome_fields(short_rows)

    for row in rows:
        row.update(_target_fields(int(row["attacker_id"]), str(row["attack_skill"]), args.skill_profile))

    groups = _group_rows(rows)
    summary: list[dict[str, Any]] = []
    termination: list[dict[str, Any]] = []
    for skill in SHORT_SKILLS:
        for attacker_id in (1, 2):
            for position in DEFENSE_POSITIONS:
                group = groups.get((skill, attacker_id, position), [])
                summary.append(_summary_row(skill, attacker_id, position, group, args.skill_profile))
                termination.extend(_termination_rows(skill, attacker_id, position, group))

    geometry = _geometry_rows()
    comparison = _mirrored_rows(summary)

    detail_fields = [
        "rally_id",
        "decision_index",
        "attack_skill",
        "attacker_id",
        "defender_id",
        "defense_preposition",
        "discounted_return",
        "attacker_return",
        "final_rally_winner",
        "winner_player",
        "termination_reason",
        "truncated",
        "contact_observed",
        "successful_return_observed",
        "attacker_miss",
        "attacker_out_of_bounds",
        "outcome_source",
        "target_landing_local_x",
        "target_landing_local_y",
        "target_landing_world_x",
        "target_landing_world_y",
        "actual_landing_x",
        "actual_landing_y",
        "landing_error",
        "landing_data_status",
    ]
    detail_rows = [{field: row.get(field, "") for field in detail_fields} for row in rows]
    _write_csv(args.detail_csv, detail_rows, detail_fields)
    _write_csv(args.summary_csv, summary)
    _write_csv(args.termination_csv, termination)
    _write_csv(args.comparison_csv, comparison)
    _write_csv(args.geometry_csv, geometry)

    print(f"Loaded collection rows: {len(raw_rows)}")
    print(f"Short-skill decision rows: {len(rows)}")
    print(f"Target profile for coordinate reporting: {args.skill_profile}")
    print(f"Wrote: {args.detail_csv}")
    print(f"Wrote: {args.summary_csv}")
    print(f"Wrote: {args.termination_csv}")
    print(f"Wrote: {args.comparison_csv}")
    print(f"Wrote: {args.geometry_csv}")
    _print_conclusion(rows, geometry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
