#!/usr/bin/env python3
"""Check physical left/right semantics of V3 defensive gantry targets.

This diagnostic deliberately bypasses ``set_defense_position`` and the V3
pre-position controller. It sends raw gantry-y commands to each robot after a
centered reset, so the result answers which world-y sign corresponds to each
robot's physical left/right direction.

The MuJoCo body frame is right-handed. For each robot, local +x is verified to
point from the base toward the racket/table and local +z is up. Therefore:

    physical left = local +y = local +z cross local +x
    physical right = local -y
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from v3.constants import (  # noqa: E402
    DEFENSE_POSITIONS,
    GantryTarget,
    gantry_to_world_position,
    target_for_defender,
    world_to_gantry_target,
)
from v3.env_wrapper import V3PrePositionEnv  # noqa: E402


GANTRY_ACTION_INDEX = {1: 1, 2: 10}
BASE_BODY = {1: "base", 2: "base_opp"}
RACKET_BODY = {1: "tennis_racket", 2: "tennis_racket_opp"}
RACKET_GEOM = {1: "racket", 2: "racket_opp"}


def _world_axes(env: V3PrePositionEnv, player: int) -> dict[str, Any]:
    """Return physical body-frame axes and verify the forward direction."""
    base = env._env.data.body(BASE_BODY[player])
    # Use the rendered racket geom position rather than the raw P2 racket body
    # xpos. The legacy MuJoCo model exposes a stale/mirrored P2 body xpos while
    # its rendered geom and link chain are in the correct world position.
    import mujoco

    racket_geom_id = mujoco.mj_name2id(
        env._env.model, mujoco.mjtObj.mjOBJ_GEOM, RACKET_GEOM[player]
    )
    racket_position = np.asarray(env._env.data.geom_xpos[racket_geom_id], dtype=float)
    rotation = np.asarray(base.xmat, dtype=float).reshape(3, 3)
    forward = rotation[:, 0]
    local_left = rotation[:, 1]
    up = rotation[:, 2]
    derived_left = np.cross(up, forward)
    racket_from_base = racket_position - np.asarray(base.xpos, dtype=float)
    forward_alignment = float(np.dot(forward, racket_from_base))
    if not np.allclose(local_left, derived_left, atol=1e-6):
        raise AssertionError(f"P{player} body frame is not right-handed as expected")
    return {
        "player": player,
        "base_body": BASE_BODY[player],
        "racket_body": RACKET_BODY[player],
        "base_world_position": np.asarray(base.xpos, dtype=float).tolist(),
        "racket_world_position": racket_position.tolist(),
        "forward_world": forward.tolist(),
        "left_world": local_left.tolist(),
        "right_world": (-local_left).tolist(),
        "up_world": up.tolist(),
        "racket_forward_alignment": forward_alignment,
        "world_y_for_own_left": float(local_left[1]),
        "world_y_for_own_right": float(-local_left[1]),
    }


def _command_world_y(
    env: V3PrePositionEnv,
    player: int,
    target_y: float,
    max_steps: int,
    semantic_side: str,
) -> dict[str, Any]:
    """Move one gantry to a physical/world-y target using raw qpos actions."""
    action_index = GANTRY_ACTION_INDEX[player]
    start = float(env.defender_gantry(player)[1])
    qpos_target = world_to_gantry_target(
        player,
        GantryTarget(float(env.defender_gantry(player)[0]), target_y),
    )
    steps = 0
    for steps in range(1, max_steps + 1):
        current = float(env.defender_gantry(player)[1])
        if abs(qpos_target.y - current) <= 0.01:
            break
        action = np.zeros(18, dtype=np.float32)
        action[action_index] = np.sign(qpos_target.y - current)
        env.step(action)
    final = float(env.defender_gantry(player)[1])
    start_world = gantry_to_world_position(
        player,
        GantryTarget(float(env.defender_gantry(player)[0]), start),
    )
    final_world = gantry_to_world_position(
        player,
        GantryTarget(float(env.defender_gantry(player)[0]), final),
    )
    return {
        "player": player,
        "semantic_side": semantic_side,
        "command_target_world_y": target_y,
        "command_target_qpos_y": qpos_target.y,
        "start_qpos_y": start,
        "final_qpos_y": final,
        "start_actual_world_y": start_world.y,
        "final_actual_world_y": final_world.y,
        "steps": steps,
        "error_world_y": final_world.y - target_y,
    }


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    env = V3PrePositionEnv(
        proc_id=1,
        reset_mode="ready",
        skill_profile=args.skill_profile,
        planner="legacy",
    )
    frames: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    try:
        for player in (1, 2):
            env.reset(seed=args.seed)
            frames.append(_world_axes(env, player))
            for position in DEFENSE_POSITIONS:
                env.reset(seed=args.seed)
                semantic_target = target_for_defender(player, position)
                command = _command_world_y(
                    env,
                    player,
                    semantic_target.y,
                    args.max_steps,
                    position,
                )
                command["semantic_target_world_x"] = semantic_target.x
                command["semantic_target_qpos_y"] = world_to_gantry_target(player, semantic_target).y
                commands.append(command)
                if args.render_dir:
                    _save_labeled_render(
                        env,
                        frame=frames[-1],
                        command=command,
                        output_dir=Path(args.render_dir),
                    )
    finally:
        env.close()

    for frame in frames:
        player = int(frame["player"])
        left_target = target_for_defender(player, "left")
        right_target = target_for_defender(player, "right")
        frame["target_for_defender_left_world_y"] = left_target.y
        frame["target_for_defender_right_world_y"] = right_target.y
        frame["target_for_defender_left_qpos_y"] = world_to_gantry_target(player, left_target).y
        frame["target_for_defender_right_qpos_y"] = world_to_gantry_target(player, right_target).y
        frame["target_left_matches_physical_left"] = bool(
            np.sign(left_target.y) == np.sign(frame["world_y_for_own_left"])
        )
        frame["target_right_matches_physical_right"] = bool(
            np.sign(right_target.y) == np.sign(frame["world_y_for_own_right"])
        )
    return frames, commands


def _save_labeled_render(
    env: V3PrePositionEnv,
    *,
    frame: dict[str, Any],
    command: dict[str, Any],
    output_dir: Path,
) -> None:
    """Save the fixed overview render with explicit semantic labels."""
    from PIL import Image, ImageDraw

    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(env.render(mode="rgb_array"))
    draw = ImageDraw.Draw(image)
    player = int(command["player"])
    text = (
        f"P{player} semantic={command['semantic_side']} "
        f"world y={command['command_target_world_y']:+.2f}, "
        f"raw qpos y={command['command_target_qpos_y']:+.2f}\n"
        f"forward world={np.round(frame['forward_world'], 2).tolist()}\n"
        f"own LEFT world-y={frame['world_y_for_own_left']:+.0f}; "
        f"own RIGHT world-y={frame['world_y_for_own_right']:+.0f}\n"
        f"actual world y={command['final_actual_world_y']:+.3f}"
    )
    draw.rectangle((8, 8, 430, 105), fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    draw.multiline_text((16, 14), text, fill=(0, 0, 0), spacing=3)
    sign = "neg" if command["command_target_world_y"] < 0 else "pos"
    output_path = output_dir / f"p{player}_world_y_{sign}.png"
    image.save(output_path)

    # A top-oblique camera makes the robots' inward-facing directions and
    # lateral motion easier to compare than the default overview camera.
    import mujoco

    renderer = mujoco.Renderer(env._env.model, height=480, width=640)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env._env.model, camera)
    camera.lookat[:] = [1.5, 0.0, 0.8]
    camera.distance = 4.5
    camera.azimuth = 90.0
    camera.elevation = -75.0
    renderer.update_scene(env._env.data, camera=camera)
    semantic_image = Image.fromarray(renderer.render())
    semantic_draw = ImageDraw.Draw(semantic_image)
    semantic_text = (
        f"P{player} base world x={frame['base_world_position'][0]:+.2f}; "
        f"racket world x={frame['racket_world_position'][0]:+.2f}\n"
        f"faces world {np.round(frame['forward_world'], 2).tolist()}\n"
        f"own LEFT is world-y {frame['world_y_for_own_left']:+.0f}; "
        f"own RIGHT is world-y {frame['world_y_for_own_right']:+.0f}\n"
        f"semantic {command['semantic_side']}; "
        f"world y={command['command_target_world_y']:+.2f}; "
        f"raw qpos y={command['command_target_qpos_y']:+.2f}; "
        f"actual world y={command['final_actual_world_y']:+.3f}"
    )
    semantic_draw.rectangle((5, 5, 470, 105), fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    semantic_draw.multiline_text((10, 10), semantic_text, fill=(0, 0, 0), spacing=2)
    semantic_image.save(output_dir / f"p{player}_world_y_{sign}_top_oblique.png")
    renderer.close()


def _print_report(frames: list[dict[str, Any]], commands: list[dict[str, Any]]) -> None:
    print("Physical frame")
    print("player | forward_world | own_left_world_y | own_right_world_y | target_left_y | target_right_y | left_semantics_ok | right_semantics_ok")
    for row in frames:
        print(
            f"P{row['player']} | {np.round(row['forward_world'], 3).tolist()} | "
            f"{row['world_y_for_own_left']:+.0f} | {row['world_y_for_own_right']:+.0f} | "
            f"{row['target_for_defender_left_world_y']:+.2f} | "
            f"{row['target_for_defender_right_world_y']:+.2f} | "
            f"{row['target_left_matches_physical_left']} | "
            f"{row['target_right_matches_physical_right']}"
        )

    print("\nRaw runtime commands")
    print("player | semantic_side | target_world_y | target_qpos_y | start_qpos_y | final_qpos_y | start_actual_world_y | final_actual_world_y | error_world_y | steps")
    for row in commands:
        print(
            f"P{row['player']} | {row['semantic_side']} | "
            f"{row['command_target_world_y']:+.2f} | {row['command_target_qpos_y']:+.2f} | "
            f"{row['start_qpos_y']:+.3f} | {row['final_qpos_y']:+.3f} | "
            f"{row['start_actual_world_y']:+.3f} | {row['final_actual_world_y']:+.3f} | "
            f"{row['error_world_y']:+.3f} | {row['steps']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--skill-profile", default="aggressive")
    parser.add_argument("--output-dir", default="v3/diagnostics/defense_semantics")
    parser.add_argument("--render-dir", default="v3/diagnostics/defense_semantics/renders")
    args = parser.parse_args()
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")

    frames, commands = run(args)
    _print_report(frames, commands)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "frames.json").write_text(json.dumps(frames, indent=2))
    (output_dir / "commands.json").write_text(json.dumps(commands, indent=2))
    with (output_dir / "commands.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(commands[0].keys()))
        writer.writeheader()
        writer.writerows(commands)
    print(f"\nWrote {output_dir / 'frames.json'}")
    print(f"Wrote {output_dir / 'commands.json'}")
    print(f"Wrote {output_dir / 'commands.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
