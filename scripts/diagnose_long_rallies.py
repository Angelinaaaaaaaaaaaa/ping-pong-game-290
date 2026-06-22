#!/usr/bin/env python3
"""Diagnose 5-skill rally attempts that hit the step cap."""

import argparse
import io
import json
import os
import sys
import time
from contextlib import redirect_stdout
from typing import Any

import numpy as np
from stable_baselines3 import PPO
import mujoco as mj

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nash_skills.env_wrapper import SkillEnv
from nash_skills.v2.collect_data import _build_ppo_obs, PPO_MODEL_PATH


def _capture_step(env: SkillEnv, action: np.ndarray):
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = env.step(action)
    return result, buf.getvalue().splitlines()


def _contacts(lines: list[str], step: int) -> list[dict[str, Any]]:
    out = []
    for line in lines:
        if "Returned successfully by ego" in line:
            player = "ego"
        elif "Returned successfully by opp" in line:
            player = "opp"
        else:
            continue
        parts = line.split()
        try:
            x_land = float(parts[-2])
            y_land = float(parts[-1])
        except (ValueError, IndexError):
            x_land = None
            y_land = None
        out.append({
            "step": step,
            "player": player,
            "x_land": x_land,
            "y_land": y_land,
            "line": line,
        })
    return out


def _active_ball_contacts(env: SkillEnv, step: int) -> list[dict[str, Any]]:
    inner = env._env
    contacts = []
    for idx in range(inner.data.ncon):
        contact = inner.data.contact[idx]
        geom1 = mj.mj_id2name(inner.model, mj.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2 = mj.mj_id2name(inner.model, mj.mjtObj.mjOBJ_GEOM, contact.geom2)
        if geom1 != "ball_geom" and geom2 != "ball_geom":
            continue
        other = geom2 if geom1 == "ball_geom" else geom1
        contacts.append({
            "step": step,
            "player": (
                "ego" if other == "racket"
                else "opp" if other == "racket_opp"
                else "other"
            ),
            "geom1": geom1,
            "geom2": geom2,
        })
    return contacts


def _termination_gap(env: SkillEnv, obs: np.ndarray) -> dict[str, float]:
    inner = env._env
    ball_x = float(obs[36])
    ego_racket_x = float(inner.data.body("tennis_racket").xpos[0])
    opp_racket_x = float(inner.data.body("tennis_racket_opp").xpos[0])
    return {
        "ball_x": ball_x,
        "ego_racket_x": ego_racket_x,
        "opp_racket_x": opp_racket_x,
        "ego_boundary": ego_racket_x - 0.3,
        "opp_boundary": opp_racket_x + 0.3,
        "past_ego_margin": ball_x - (ego_racket_x - 0.3),
        "past_opp_margin": ball_x - (opp_racket_x + 0.3),
    }


def _classify_no_done(env: SkillEnv, obs: np.ndarray, crossings: int, samples: list[dict[str, Any]]) -> str:
    gap = _termination_gap(env, obs)
    ball_pos = obs[36:39]
    ball_vel = obs[39:42]
    speed = float(np.linalg.norm(ball_vel))
    if gap["ball_x"] < gap["ego_boundary"]:
        return "would_end_past_ego_but_done_false"
    if gap["ball_x"] > gap["opp_boundary"]:
        return "would_end_past_opp_but_done_false"
    if speed < 0.05:
        return "dead_or_nearly_stationary_ball_inside_racket_bounds"
    if float(ball_pos[2]) < 0.1:
        return "ball_low_inside_racket_bounds"
    if crossings > 0:
        return "active_rally_inside_racket_bounds"
    if samples and all(abs(float(s["ball_pos"][0]) - 1.5) > 0.5 for s in samples):
        return "no_net_crossing_inside_racket_bounds"
    return "inside_racket_bounds_no_env_done_condition"


def _attempt_signature(samples: list[dict[str, Any]]) -> tuple:
    return tuple(
        (
            s["step"],
            tuple(round(float(v), 4) for v in s["ball_pos"]),
            tuple(round(float(v), 4) for v in s["ball_vel"]),
        )
        for s in samples
    )


def run_attempt(
    env: SkillEnv,
    model: PPO,
    skill1: str,
    skill2: str,
    max_steps: int,
    deterministic: bool,
) -> dict[str, Any]:
    env.set_skills(skill1, skill2)
    obs, info = env.reset()
    prev_ball_x = float(obs[36])
    initial = {
        "ball_pos": obs[36:39].astype(float).tolist(),
        "ball_vel": obs[39:42].astype(float).tolist(),
        "ego_gantry": obs[0:2].astype(float).tolist(),
        "opp_gantry": obs[18:20].astype(float).tolist(),
        "skill_fields": obs[-2:].astype(float).tolist(),
    }

    contacts = []
    raw_contacts = []
    samples = [{
        "step": 0,
        "ball_pos": obs[36:39].astype(float).tolist(),
        "ball_vel": obs[39:42].astype(float).tolist(),
    }]
    crossings = 0
    done = False
    final_info = {}
    final_step = 0
    start = time.monotonic()

    for step in range(1, max_steps + 1):
        raw_contacts.extend(_active_ball_contacts(env, step))
        ppo1 = _build_ppo_obs(obs, info, player=1)
        ppo2 = _build_ppo_obs(obs, info, player=2)
        a1, _ = model.predict(ppo1, deterministic=deterministic)
        a2, _ = model.predict(ppo2, deterministic=deterministic)
        action = np.zeros(18, dtype=np.float32)
        action[:9] = a1[:9]
        action[9:] = a2[:9]

        (obs, _reward, done, _truncated, info), lines = _capture_step(env, action)
        contacts.extend(_contacts(lines, step))
        curr_ball_x = float(obs[36])
        if (prev_ball_x - 1.5) * (curr_ball_x - 1.5) < 0:
            crossings += 1
        prev_ball_x = curr_ball_x

        if step % 50 == 0 or done:
            samples.append({
                "step": step,
                "ball_pos": obs[36:39].astype(float).tolist(),
                "ball_vel": obs[39:42].astype(float).tolist(),
            })

        final_step = step
        final_info = dict(info)
        if done:
            break

    final = {
        "ball_pos": obs[36:39].astype(float).tolist(),
        "ball_vel": obs[39:42].astype(float).tolist(),
        "speed": float(np.linalg.norm(obs[39:42])),
        "skill_fields": obs[-2:].astype(float).tolist(),
        "termination_gap": _termination_gap(env, obs),
    }
    reason = (
        final_info.get("termination_reason")
        if done
        else _classify_no_done(env, obs, crossings, samples)
    )
    return {
        "skill1": skill1,
        "skill2": skill2,
        "max_steps": max_steps,
        "steps": final_step,
        "done": bool(done),
        "winner": final_info.get("winner"),
        "reason": reason,
        "elapsed": time.monotonic() - start,
        "initial": initial,
        "contacts": contacts,
        "raw_contacts": raw_contacts,
        "contact_counts": {
            "ego": sum(c["player"] == "ego" for c in contacts),
            "opp": sum(c["player"] == "opp" for c in contacts),
        },
        "raw_contact_counts": {
            "ego": sum(c["player"] == "ego" for c in raw_contacts),
            "opp": sum(c["player"] == "opp" for c in raw_contacts),
            "other": sum(c["player"] == "other" for c in raw_contacts),
        },
        "crossings": crossings,
        "samples": samples,
        "final": final,
        "still_moving": final["speed"] > 0.05,
        "trajectory_signature": _attempt_signature(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", default=["center_safe:center_safe", "right:left"])
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--long-max-steps", type=int, default=1200)
    parser.add_argument("--long-pair", default="center_safe:center_safe")
    parser.add_argument("--ppo", default=PPO_MODEL_PATH)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    deterministic = not args.stochastic
    print(f"Loading PPO model: {args.ppo}", flush=True)
    print(f"PPO deterministic={deterministic}", flush=True)
    model = PPO.load(args.ppo)
    env = SkillEnv(proc_id=1, history=4)

    results = {"deterministic": deterministic, "pairs": {}, "long_attempt": None}
    try:
        for pair in args.pairs:
            skill1, skill2 = pair.split(":", 1)
            pair_results = []
            print(f"\nPAIR {skill1} vs {skill2}", flush=True)
            for idx in range(args.attempts):
                result = run_attempt(env, model, skill1, skill2, args.max_steps, deterministic)
                pair_results.append(result)
                print(
                    f"  attempt={idx + 1} steps={result['steps']} done={result['done']} "
                    f"winner={result['winner']} reason={result['reason']} "
                    f"successful_contacts={result['contact_counts']} "
                    f"raw_contacts={result['raw_contact_counts']} "
                    f"crossings={result['crossings']} "
                    f"elapsed={result['elapsed']:.2f}s",
                    flush=True,
                )
                print(
                    f"    initial ball={result['initial']['ball_pos']} "
                    f"vel={result['initial']['ball_vel']} gantry1={result['initial']['ego_gantry']} "
                    f"gantry2={result['initial']['opp_gantry']}",
                    flush=True,
                )
                print(
                    f"    final ball={result['final']['ball_pos']} "
                    f"vel={result['final']['ball_vel']} speed={result['final']['speed']:.3f}",
                    flush=True,
                )
            signatures = [r["trajectory_signature"] for r in pair_results]
            identical = len(set(signatures)) == 1 if signatures else False
            initial_states = [
                (
                    tuple(round(v, 5) for v in r["initial"]["ball_pos"]),
                    tuple(round(v, 5) for v in r["initial"]["ball_vel"]),
                    tuple(round(v, 5) for v in r["initial"]["ego_gantry"]),
                    tuple(round(v, 5) for v in r["initial"]["opp_gantry"]),
                )
                for r in pair_results
            ]
            reset_differs = len(set(initial_states)) > 1
            print(f"  reset_states_differ={reset_differs} trajectories_identical={identical}", flush=True)
            results["pairs"][pair] = {
                "reset_states_differ": reset_differs,
                "trajectories_identical": identical,
                "attempts": pair_results,
            }

        long_skill1, long_skill2 = args.long_pair.split(":", 1)
        print(f"\nLONG {long_skill1} vs {long_skill2} max_steps={args.long_max_steps}", flush=True)
        long_result = run_attempt(env, model, long_skill1, long_skill2, args.long_max_steps, deterministic)
        results["long_attempt"] = long_result
        print(
            f"  steps={long_result['steps']} done={long_result['done']} "
            f"winner={long_result['winner']} reason={long_result['reason']} "
            f"successful_contacts={long_result['contact_counts']} "
            f"raw_contacts={long_result['raw_contact_counts']} "
            f"crossings={long_result['crossings']} "
            f"elapsed={long_result['elapsed']:.2f}s",
            flush=True,
        )
        print(
            f"  final ball={long_result['final']['ball_pos']} "
            f"vel={long_result['final']['ball_vel']} speed={long_result['final']['speed']:.3f}",
            flush=True,
        )
    finally:
        env.close()

    if args.json_out:
        serializable = json.loads(json.dumps(results, default=str))
        with open(args.json_out, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nWrote {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
