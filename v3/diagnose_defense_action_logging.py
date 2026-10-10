#!/usr/bin/env python3
"""Sanity-check V3 semantic defense labels against mirrored coordinates.

This is intentionally a coordinate-only test.  It exercises the same
semantic-side and world/raw-gantry conversions used by the runtime logger,
without starting MuJoCo or changing controller behavior.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from v3.constants import (  # noqa: E402
    gantry_to_world_position,
    target_for_defender,
    world_to_gantry_target,
)
from v3.eval_fixed_preposition import player_relative_side  # noqa: E402


def run(output_csv: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for player in (1, 2):
        for semantic_action in ("left", "right"):
            world_target = target_for_defender(player, semantic_action)
            raw_target = world_to_gantry_target(player, world_target)
            round_trip_world = gantry_to_world_position(player, raw_target)
            logged_side = player_relative_side(player, round_trip_world.y)
            matches = logged_side == semantic_action
            row = {
                "player": player,
                "defense_semantic_action": semantic_action,
                "defense_world_y": world_target.y,
                "defender_raw_qpos_y": raw_target.y,
                "defender_actual_world_y": round_trip_world.y,
                "defense_player_relative_side": logged_side,
                "matches_intended_side": matches,
            }
            rows.append(row)
            print(
                f"P{player} {semantic_action}: "
                f"world_y={world_target.y:+.3f} "
                f"raw_qpos_y={raw_target.y:+.3f} "
                f"actual_world_y={round_trip_world.y:+.3f} "
                f"logged_side={logged_side} "
                f"match={matches}"
            )

    if not all(bool(row["matches_intended_side"]) for row in rows):
        raise AssertionError("semantic defense logging does not match player-relative side")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"PASS: wrote {output_csv}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("skill_eval/v3/defense_action_logging_sanity.csv"),
    )
    args = parser.parse_args()
    run(args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
