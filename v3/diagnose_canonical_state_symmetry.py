#!/usr/bin/env python3
"""Verify P1/P2 role symmetry of the canonical V3 Q-state encoder."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from v3.racket_target_planner import mirror_position_to_world, mirror_velocity_to_local  # noqa: E402
from v3.state_encoder import (  # noqa: E402
    STATE_ENCODER_NAME,
    STATE_FIELD_NAMES,
    encode_canonical_state,
)


def _role_qpos(seed: float) -> np.ndarray:
    return np.asarray(
        [-0.7, seed, 0.11 + seed, -0.22, 0.31, -0.14, 0.08, -0.27, 0.19],
        dtype=np.float32,
    )


def _role_qvel(seed: float) -> np.ndarray:
    return np.asarray(
        [0.03, -0.04 * seed, 0.12, -0.08, 0.05, 0.02, -0.07, 0.09, -0.11],
        dtype=np.float32,
    )


def _role_history(seed: float) -> np.ndarray:
    frames = []
    for index in range(4):
        frame = _role_qpos(seed + 0.02 * index)
        frames.append(frame)
    return np.asarray(frames, dtype=np.float32).reshape(-1)


def _target_pose(seed: float) -> np.ndarray:
    rotation = R.from_euler("xyz", [0.08 + seed, -0.12, 0.22]).as_quat()
    return np.asarray([-0.35, 0.16 + seed, 1.05, *rotation], dtype=np.float32)


def _scene(
    *,
    p1_qpos: np.ndarray,
    p2_qpos: np.ndarray,
    p1_qvel: np.ndarray,
    p2_qvel: np.ndarray,
    p1_history: np.ndarray,
    p2_history: np.ndarray,
    p1_target: np.ndarray,
    p2_target: np.ndarray,
    p1_diff: np.ndarray,
    p2_diff: np.ndarray,
    ball_position: np.ndarray,
    ball_velocity: np.ndarray,
    p1_skill: float,
    p2_skill: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    obs = np.zeros(116, dtype=np.float32)
    obs[0:9] = p1_qpos
    obs[9:18] = p1_qvel
    obs[18:27] = p2_qpos
    obs[27:36] = p2_qvel
    obs[36:39] = ball_position
    obs[39:42] = ball_velocity
    obs[42:78] = p1_history
    obs[78:114] = p2_history
    obs[-2] = p1_skill
    obs[-1] = p2_skill
    info = {
        "target": p1_target,
        "target_opp": p2_target,
        "diff_pos": p1_diff,
        "diff_pos_opp": p2_diff,
    }
    return obs, info


def _mirrored_scene() -> tuple[tuple[np.ndarray, dict], tuple[np.ndarray, dict]]:
    attacker_qpos = _role_qpos(0.21)
    defender_qpos = _role_qpos(-0.17)
    attacker_qvel = _role_qvel(0.21)
    defender_qvel = _role_qvel(-0.17)
    attacker_history = _role_history(0.21)
    defender_history = _role_history(-0.17)
    attacker_target = _target_pose(0.21)
    defender_target = _target_pose(-0.17)
    attacker_diff = np.asarray([0.04, -0.07, 0.02], dtype=np.float32)
    defender_diff = np.asarray([-0.06, 0.03, -0.01], dtype=np.float32)
    ball_position = np.asarray([0.42, -0.19, 0.86], dtype=np.float32)
    ball_velocity = np.asarray([-0.63, 0.28, 0.91], dtype=np.float32)

    # Scene A: physical P1 is attacker and physical P2 is defender.
    scene_p1_attacker = _scene(
        p1_qpos=attacker_qpos,
        p2_qpos=defender_qpos,
        p1_qvel=attacker_qvel,
        p2_qvel=defender_qvel,
        p1_history=attacker_history,
        p2_history=defender_history,
        p1_target=attacker_target,
        p2_target=defender_target,
        p1_diff=attacker_diff,
        p2_diff=defender_diff,
        ball_position=ball_position,
        ball_velocity=ball_velocity,
        p1_skill=0.75,
        p2_skill=0.25,
    )

    # Scene B is the exact physical mirror with roles swapped.  P2 now has
    # the same local attacker state and P1 has the same local defender state.
    scene_p2_attacker = _scene(
        p1_qpos=defender_qpos,
        p2_qpos=attacker_qpos,
        p1_qvel=defender_qvel,
        p2_qvel=attacker_qvel,
        p1_history=defender_history,
        p2_history=attacker_history,
        p1_target=defender_target,
        p2_target=attacker_target,
        p1_diff=defender_diff,
        p2_diff=attacker_diff,
        ball_position=mirror_position_to_world(ball_position),
        ball_velocity=mirror_velocity_to_local(ball_velocity),
        p1_skill=0.25,
        p2_skill=0.75,
    )
    return scene_p1_attacker, scene_p2_attacker


def run(output_csv: str, tolerance: float) -> int:
    scene_p1, scene_p2 = _mirrored_scene()
    state_p1 = encode_canonical_state(scene_p1[0], scene_p1[1], 1, attack_skill="right")
    state_p2 = encode_canonical_state(scene_p2[0], scene_p2[1], 2, attack_skill="right")
    differences = np.abs(state_p1 - state_p2)
    order = np.argsort(-differences)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_csv).open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["index", "field", "p1_attacker_value", "p2_attacker_value", "abs_difference"],
        )
        writer.writeheader()
        for index, field in enumerate(STATE_FIELD_NAMES):
            writer.writerow({
                "index": index,
                "field": field,
                "p1_attacker_value": float(state_p1[index]),
                "p2_attacker_value": float(state_p2[index]),
                "abs_difference": float(differences[index]),
            })

    max_difference = float(np.max(differences))
    mean_difference = float(np.mean(differences))
    print(f"State encoder: {STATE_ENCODER_NAME}")
    print(f"State dimension: {state_p1.shape[0]}")
    print(f"Max absolute difference: {max_difference:.12g}")
    print(f"Mean absolute difference: {mean_difference:.12g}")
    print("Largest differences:")
    for index in order[:10]:
        print(
            f"  {int(index):2d} {STATE_FIELD_NAMES[int(index)]:32s} "
            f"diff={differences[index]:.12g} "
            f"p1={state_p1[index]:.12g} p2={state_p2[index]:.12g}"
        )
    print(f"CSV saved to: {output_csv}")
    if max_difference > tolerance:
        print(f"FAIL: max difference exceeds tolerance {tolerance:g}")
        return 1
    print(f"PASS: max difference is within tolerance {tolerance:g}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-csv",
        default="skill_eval/v3/canonical_state_symmetry.csv",
    )
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    raise SystemExit(run(args.output_csv, args.tolerance))
