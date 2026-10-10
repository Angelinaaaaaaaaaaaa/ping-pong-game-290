#!/usr/bin/env python3
"""Random V3.1 decision collector for attacker skill x defender pre-position."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle as pkl
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nash_skills.skills import N_SKILLS, SKILL_NAMES, SKILL_PROFILE_NAMES, skill_index  # noqa: E402
from nash_skills.v2.labeling import GAMMA  # noqa: E402
from v3.constants import DEFENSE_POSITIONS  # noqa: E402
from v3.eval_fixed_preposition import (  # noqa: E402
    HISTORY,
    PPO_MODEL_PATH,
    TABLE_SHIFT,
    attack_action_diagnostics,
    build_ppo_action,
    defense_action_diagnostics,
    maybe_render,
    opponent,
    position_snapshot,
    receiver_from_crossing,
)
from v3.state_encoder import (  # noqa: E402
    STATE_ENCODER_NAME,
    state_for_attacker as encode_state_for_attacker,
)


def skill_value(skill: str) -> float:
    return skill_index(skill) / (N_SKILLS - 1)


def state_for_attacker(obs: np.ndarray, info: dict[str, Any], attacker: int, attack_skill: str) -> list[float]:
    state = encode_state_for_attacker(obs, info, attacker, attack_skill=attack_skill)
    return [float(x) for x in state]


def raw_obs_for_decision(obs: np.ndarray, attacker: int, attack_skill: str) -> list[float]:
    raw = obs.copy()
    if attacker == 1:
        raw[-2] = skill_value(attack_skill)
    elif attacker == 2:
        raw[-1] = skill_value(attack_skill)
    else:
        raise ValueError("attacker must be 1 or 2")
    return [float(x) for x in raw]


def winner_label(winner: str) -> int:
    if winner == "ego":
        return 1
    if winner == "opp":
        return 2
    return 0


def discounted_attacker_returns(rows: list[dict[str, Any]], winner: str, gamma: float) -> list[float]:
    final_winner = winner_label(winner)
    if final_winner == 0:
        return [0.0 for _row in rows]
    last = len(rows) - 1
    returns = []
    for index, row in enumerate(rows):
        attacker = int(row["attacker_id"])
        terminal = 1.0 if attacker == final_winner else -1.0
        returns.append((gamma ** (last - index)) * terminal)
    return returns


def commit_random_decision(
    *,
    env,
    obs: np.ndarray,
    info: dict[str, Any],
    rng: np.random.Generator,
    rally_id: int,
    decision_index: int,
    attacker: int,
    gamma: float,
    debug: bool,
) -> dict[str, Any]:
    attack_skill = str(rng.choice(SKILL_NAMES))
    defense_position = str(rng.choice(DEFENSE_POSITIONS))
    defender = opponent(attacker)
    state = state_for_attacker(obs, info, attacker, attack_skill)
    raw_obs = raw_obs_for_decision(obs, attacker, attack_skill)

    if attacker == 1:
        env.set_skills(attack_skill, env._skill2)
    else:
        env.set_skills(env._skill1, attack_skill)
    commitment = env.set_defense_position(defender, defense_position)
    actual = position_snapshot(env, defender)
    semantic_diagnostics = {
        **attack_action_diagnostics(attacker, attack_skill, env.skill_profile),
        **defense_action_diagnostics(defender, defense_position, commitment, actual),
    }
    row = {
        "rally_id": rally_id,
        "decision_index": decision_index,
        "attacker_id": attacker,
        "attack_skill": attack_skill,
        **semantic_diagnostics,
        "defender_id": defender,
        "defense_preposition": defense_position,
        "defender_target_world_x": commitment.target.x,
        "defender_target_world_y": commitment.target.y,
        "defender_target_qpos_x": commitment.qpos_target.x,
        "defender_target_qpos_y": commitment.qpos_target.y,
        "defender_actual_qpos_x_at_commit": actual["qpos_x"],
        "defender_actual_qpos_y_at_commit": actual["qpos_y"],
        "defender_actual_world_x_at_commit": actual["world_x"],
        "defender_actual_world_y_at_commit": actual["world_y"],
        # Backward-compatible aliases: target_* are world and actual_* are qpos.
        "defender_target_x": commitment.target.x,
        "defender_target_y": commitment.target.y,
        "defender_actual_x_at_commit": actual["qpos_x"],
        "defender_actual_y_at_commit": actual["qpos_y"],
        "defender_actual_qpos_x_at_receive": "",
        "defender_actual_qpos_y_at_receive": "",
        "defender_actual_world_x_at_receive": "",
        "defender_actual_world_y_at_receive": "",
        "defender_actual_x_at_receive": "",
        "defender_actual_y_at_receive": "",
        "defender_actual_qpos_x_at_contact": "",
        "defender_actual_qpos_y_at_contact": "",
        "defender_actual_world_x_at_contact": "",
        "defender_actual_world_y_at_contact": "",
        "defender_actual_x_at_contact": "",
        "defender_actual_y_at_contact": "",
        "state_before_action": json.dumps(state),
        "raw_obs_before_action": json.dumps(raw_obs),
        "final_rally_winner": "",
        "termination_reason": "",
        "truncated": False,
        "rally_length": "",
        "physics_steps": "",
        "discounted_return": "",
        "gamma": gamma,
        "state_encoder": STATE_ENCODER_NAME,
    }
    if debug:
        print(
            f"rally={rally_id} decision={decision_index}: "
            f"attacker=P{attacker} skill={attack_skill} "
            f"defender=P{defender} pre_position={defense_position} "
            f"target_world=({commitment.target.x:.3f}, {commitment.target.y:.3f}) "
            f"target_qpos=({commitment.qpos_target.x:.3f}, {commitment.qpos_target.y:.3f})",
            flush=True,
        )
    return row


def update_previous_receive(env, pending_rows: list[dict[str, Any]]) -> None:
    if not pending_rows:
        return
    previous = pending_rows[-1]
    defender = int(previous["defender_id"])
    actual = position_snapshot(env, defender)
    previous["defender_actual_qpos_x_at_receive"] = actual["qpos_x"]
    previous["defender_actual_qpos_y_at_receive"] = actual["qpos_y"]
    previous["defender_actual_world_x_at_receive"] = actual["world_x"]
    previous["defender_actual_world_y_at_receive"] = actual["world_y"]
    previous["defender_actual_x_at_receive"] = actual["qpos_x"]
    previous["defender_actual_y_at_receive"] = actual["qpos_y"]


def update_previous_contact(env, pending_rows: list[dict[str, Any]], lines: list[str]) -> None:
    if not pending_rows:
        return
    previous = pending_rows[-1]
    defender = int(previous["defender_id"])
    defender_label = "ego" if defender == 1 else "opp"
    if any(f"by {defender_label}" in line for line in lines):
        actual = position_snapshot(env, defender)
        previous["defender_actual_qpos_x_at_contact"] = actual["qpos_x"]
        previous["defender_actual_qpos_y_at_contact"] = actual["qpos_y"]
        previous["defender_actual_world_x_at_contact"] = actual["world_x"]
        previous["defender_actual_world_y_at_contact"] = actual["world_y"]
        previous["defender_actual_x_at_contact"] = actual["qpos_x"]
        previous["defender_actual_y_at_contact"] = actual["qpos_y"]


def finalize_rally_rows(
    pending_rows: list[dict[str, Any]],
    *,
    winner: str,
    termination_reason: str,
    truncated: bool,
    rally_length: int,
    physics_steps: int,
    gamma: float,
) -> None:
    returns = discounted_attacker_returns(pending_rows, winner, gamma)
    for row, value in zip(pending_rows, returns):
        row["final_rally_winner"] = winner
        row["termination_reason"] = termination_reason
        row["truncated"] = truncated
        row["rally_length"] = rally_length
        row["physics_steps"] = physics_steps
        row["discounted_return"] = value


def append_until_target(rows: list[dict[str, Any]], pending_rows: list[dict[str, Any]], target: int) -> None:
    remaining = max(0, target - len(rows))
    if remaining:
        rows.extend(pending_rows[:remaining])


def all_combo_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter((row["attack_skill"], row["defense_preposition"]) for row in rows)
    output = []
    values = []
    for skill in SKILL_NAMES:
        for position in DEFENSE_POSITIONS:
            count = counts[(skill, position)]
            values.append(count)
            output.append({
                "attack_skill": skill,
                "defense_preposition": position,
                "count": count,
            })
    nonzero = [value for value in values if value > 0]
    ratio = (max(nonzero) / min(nonzero)) if nonzero and len(nonzero) == len(values) else ""
    print("\nV3.1 random collection coverage:")
    for row in output:
        print(f"  {row['attack_skill']:12s} x {row['defense_preposition']:6s}: {row['count']}")
    if ratio == "":
        print("  balance_ratio: incomplete coverage")
    else:
        print(f"  balance_ratio max/min: {ratio:.3f}")
    return output


def write_csv(path: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_pickle_rallies(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_rally: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_rally.setdefault(int(row["rally_id"]), []).append(row)

    rallies: list[dict[str, Any]] = []
    for rally_id, rally_rows in sorted(by_rally.items()):
        ordered = sorted(rally_rows, key=lambda row: int(row["decision_index"]))
        states = [
            np.asarray(json.loads(row["state_before_action"]), dtype=np.float32)
            for row in ordered
        ]
        raw_obs = [
            np.asarray(json.loads(row["raw_obs_before_action"]), dtype=np.float32)
            for row in ordered
        ]
        decisions = []
        for row in ordered:
            decisions.append({
                "rally_id": int(row["rally_id"]),
                "decision_index": int(row["decision_index"]),
                "state": np.asarray(json.loads(row["state_before_action"]), dtype=np.float32),
                "raw_obs": np.asarray(json.loads(row["raw_obs_before_action"]), dtype=np.float32),
                "attacker_id": int(row["attacker_id"]),
                "attack_skill": str(row["attack_skill"]),
                "defender_id": int(row["defender_id"]),
                "defense_preposition": str(row["defense_preposition"]),
                "defender_target_world_position": (
                    float(row["defender_target_world_x"]),
                    float(row["defender_target_world_y"]),
                ),
                "defender_target_qpos_position": (
                    float(row["defender_target_qpos_x"]),
                    float(row["defender_target_qpos_y"]),
                ),
                "defender_target_position": (
                    float(row["defender_target_x"]),
                    float(row["defender_target_y"]),
                ),
                "defender_actual_world_position_at_commit": _optional_position(
                    row["defender_actual_world_x_at_commit"],
                    row["defender_actual_world_y_at_commit"],
                ),
                "defender_actual_qpos_position_at_commit": _optional_position(
                    row["defender_actual_qpos_x_at_commit"],
                    row["defender_actual_qpos_y_at_commit"],
                ),
                "defender_actual_position_at_commit": (
                    float(row["defender_actual_x_at_commit"]),
                    float(row["defender_actual_y_at_commit"]),
                ),
                "defender_actual_world_position_at_receive": _optional_position(
                    row["defender_actual_world_x_at_receive"],
                    row["defender_actual_world_y_at_receive"],
                ),
                "defender_actual_qpos_position_at_receive": _optional_position(
                    row["defender_actual_qpos_x_at_receive"],
                    row["defender_actual_qpos_y_at_receive"],
                ),
                "defender_actual_position_at_receive": _optional_position(
                    row["defender_actual_x_at_receive"],
                    row["defender_actual_y_at_receive"],
                ),
                "defender_actual_world_position_at_contact": _optional_position(
                    row["defender_actual_world_x_at_contact"],
                    row["defender_actual_world_y_at_contact"],
                ),
                "defender_actual_qpos_position_at_contact": _optional_position(
                    row["defender_actual_qpos_x_at_contact"],
                    row["defender_actual_qpos_y_at_contact"],
                ),
                "defender_actual_position_at_contact": _optional_position(
                    row["defender_actual_x_at_contact"],
                    row["defender_actual_y_at_contact"],
                ),
                "final_rally_winner": str(row["final_rally_winner"]),
                "winner": winner_label(str(row["final_rally_winner"])),
                "termination_reason": str(row["termination_reason"]),
                "truncated": bool(row["truncated"]),
                "rally_length": int(row["rally_length"]),
                "physics_steps": int(row["physics_steps"]),
                "discounted_return": float(row["discounted_return"]),
                "return_perspective": "attacker",
                "state_encoder": str(row["state_encoder"]),
            })

        first = ordered[0]
        rallies.append({
            "rally_id": rally_id,
            "skill1": str(first["attack_skill"]),
            "skill2": f"defense_{first['defense_preposition']}",
            "states": states,
            "raw_obs": raw_obs,
            "skill_pairs": [
                (str(row["attack_skill"]), f"defense_{row['defense_preposition']}")
                for row in ordered
            ],
            "attacker_returns": [
                float(row["discounted_return"])
                for row in ordered
            ],
            "winner": winner_label(str(first["final_rally_winner"])),
            "final_rally_winner": str(first["final_rally_winner"]),
            "termination_reason": str(first["termination_reason"]),
            "truncated": bool(first["truncated"]),
            "decisions": decisions,
            "state_encoder": STATE_ENCODER_NAME,
        })
    return rallies


def _optional_position(x_value: Any, y_value: Any) -> tuple[float, float] | None:
    if x_value in ("", None) or y_value in ("", None):
        return None
    return (float(x_value), float(y_value))


def write_pickle(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rallies = build_pickle_rallies(rows)
    with Path(path).open("wb") as f:
        pkl.dump(rallies, f)
    print(f"Pickle saved to: {path}")


def planner_output_path(path: str, planner: str, default_path: str) -> str:
    """Avoid replacing the default legacy artifact for optimized collection."""
    if planner == "legacy" or path != default_path:
        return path
    output = Path(path)
    return str(output.with_name(f"{output.stem}_optimized{output.suffix}"))


def collect(args: argparse.Namespace) -> list[dict[str, Any]]:
    from stable_baselines3 import PPO
    from nash_skills.eval_matchup import _capture_env_step, _infer_winner
    from v3.env_wrapper import V3PrePositionEnv

    print(f"Planner: {args.planner}", flush=True)
    rng = np.random.default_rng(args.seed)
    ppo = PPO.load(args.ppo)
    env = V3PrePositionEnv(
        proc_id=1,
        history=HISTORY,
        reset_mode=args.reset_mode,
        skill_profile=args.skill_profile,
        gantry_speed_scale=args.gantry_speed_scale,
        planner=args.planner,
    )

    rows: list[dict[str, Any]] = []
    rally_id = 0
    start = time.monotonic()

    try:
        while len(rows) < args.decisions and rally_id < args.max_rallies:
            env.set_skills("left", "right")
            env.clear_defense_position()
            obs, info = env.reset()
            prev_ball_x = float(obs[36])
            pending_rows: list[dict[str, Any]] = []
            steps = 0
            rally_length = 0

            while steps < args.steps and len(rows) < args.decisions:
                action = build_ppo_action(ppo, obs, info)
                (obs, _, done, _, info), lines = _capture_env_step(env, action)
                maybe_render(env, args.render)
                steps += 1
                update_previous_contact(env, pending_rows, lines)

                curr_ball_x = float(obs[36])
                attacker = receiver_from_crossing(prev_ball_x, curr_ball_x)
                if attacker is not None:
                    update_previous_receive(env, pending_rows)
                    rally_length += 1
                    pending_rows.append(
                        commit_random_decision(
                            env=env,
                            obs=obs,
                            info=info,
                            rng=rng,
                            rally_id=rally_id,
                            decision_index=rally_length,
                            attacker=attacker,
                            gamma=args.gamma,
                            debug=args.debug_decisions or args.render,
                        )
                    )

                prev_ball_x = curr_ball_x

                if done:
                    winner = _infer_winner(obs, info)
                    finalize_rally_rows(
                        pending_rows,
                        winner=winner,
                        termination_reason=str(info.get("termination_reason", "done")),
                        truncated=False,
                        rally_length=rally_length,
                        physics_steps=steps,
                        gamma=args.gamma,
                    )
                    append_until_target(rows, pending_rows, args.decisions)
                    break
            else:
                finalize_rally_rows(
                    pending_rows,
                    winner="truncated",
                    termination_reason="max_steps",
                    truncated=True,
                    rally_length=rally_length,
                    physics_steps=steps,
                    gamma=args.gamma,
                )
                append_until_target(rows, pending_rows, args.decisions)

            rally_id += 1
            if args.progress_every and rally_id % args.progress_every == 0:
                elapsed = time.monotonic() - start
                print(f"rallies={rally_id} decisions={len(rows)}/{args.decisions} elapsed={elapsed:.1f}s", flush=True)
    finally:
        env.close()

    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V3.1 random attacker-skill x defender-preposition collector")
    parser.add_argument("--decisions", type=int, default=1000)
    parser.add_argument("--max-rallies", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--ppo", default=PPO_MODEL_PATH)
    parser.add_argument("--reset-mode", choices=["clean", "ready", "carryover"], default="ready")
    parser.add_argument("--skill-profile", choices=SKILL_PROFILE_NAMES, default="aggressive")
    parser.add_argument("--gantry-speed-scale", type=float, default=1.0)
    parser.add_argument("--planner", choices=["legacy", "optimized"], default="legacy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--debug-decisions", action="store_true")
    parser.add_argument("--render", action="store_true", help="Use the existing live MuJoCo viewer via env.render().")
    parser.add_argument("--output-csv", default="skill_eval/v3/random_preposition_canonical_decisions.csv")
    parser.add_argument("--coverage-csv", default="skill_eval/v3/random_preposition_canonical_coverage.csv")
    parser.add_argument("--output-json", default="skill_eval/v3/random_preposition_canonical_decisions.json")
    parser.add_argument("--output-pkl", default="data/v3/random_preposition_canonical_decisions.pkl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    default_outputs = {
        "output_csv": "skill_eval/v3/random_preposition_canonical_decisions.csv",
        "coverage_csv": "skill_eval/v3/random_preposition_canonical_coverage.csv",
        "output_json": "skill_eval/v3/random_preposition_canonical_decisions.json",
        "output_pkl": "data/v3/random_preposition_canonical_decisions.pkl",
    }
    for field, default_path in default_outputs.items():
        setattr(args, field, planner_output_path(getattr(args, field), args.planner, default_path))
    rows = collect(args)
    coverage = all_combo_counts(rows)
    fields = [
        "rally_id",
        "decision_index",
        "attacker_id",
        "attack_skill",
        "attack_semantic_action",
        "attack_player_relative_side",
        "attack_target_local_y",
        "attack_target_world_x",
        "attack_target_world_y",
        "defender_id",
        "defense_preposition",
        "defense_semantic_action",
        "defense_world_y",
        "defender_raw_qpos_y",
        "defender_actual_world_y",
        "defender_actual_raw_qpos_y",
        "defense_target_player_relative_side",
        "defense_player_relative_side",
        "state_encoder",
        "defender_target_world_x",
        "defender_target_world_y",
        "defender_target_qpos_x",
        "defender_target_qpos_y",
        "defender_target_x",
        "defender_target_y",
        "defender_actual_qpos_x_at_commit",
        "defender_actual_qpos_y_at_commit",
        "defender_actual_world_x_at_commit",
        "defender_actual_world_y_at_commit",
        "defender_actual_x_at_commit",
        "defender_actual_y_at_commit",
        "defender_actual_qpos_x_at_receive",
        "defender_actual_qpos_y_at_receive",
        "defender_actual_world_x_at_receive",
        "defender_actual_world_y_at_receive",
        "defender_actual_x_at_receive",
        "defender_actual_y_at_receive",
        "defender_actual_qpos_x_at_contact",
        "defender_actual_qpos_y_at_contact",
        "defender_actual_world_x_at_contact",
        "defender_actual_world_y_at_contact",
        "defender_actual_x_at_contact",
        "defender_actual_y_at_contact",
        "state_before_action",
        "raw_obs_before_action",
        "final_rally_winner",
        "termination_reason",
        "truncated",
        "rally_length",
        "physics_steps",
        "discounted_return",
        "gamma",
    ]
    write_csv(args.output_csv, rows, fields)
    write_csv(args.coverage_csv, coverage, ["attack_skill", "defense_preposition", "count"])
    write_pickle(args.output_pkl, rows)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_json).open("w") as f:
        json.dump(
            {
                "metadata": {
                    "collector": "v3.collect_random_preposition",
                    "planner": args.planner,
                    "seed": args.seed,
                    "gamma": args.gamma,
                },
                "decisions": rows,
                "coverage": coverage,
            },
            f,
            indent=2,
        )
    print(f"\nDecision CSV saved to: {args.output_csv}")
    print(f"Coverage CSV saved to: {args.coverage_csv}")
    print(f"JSON saved to: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
