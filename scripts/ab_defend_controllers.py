"""
Strict A/B of 4 defend-steering controller designs (nash_skills.defend_
controllers), instrumented with gantry velocity at the moment of release/
freeze, to find which design axis -- approach smoothness, or
release-vs-freeze after reaching -- explains why this project's own
defend-steering override suppressed return ability (ego_wins 9-10/10 in
every combo of the project's own 6-combination mechanistic-validation
experiment, even where the target was reached 100% of the time) while a
teammate's structurally similar "V3" pipeline did not (sustained rallies
averaging 5.6 decisions/rally, 100% return-success-per-attempt in a
smoke test) -- despite both overriding the exact same two gantry action
components.

Runs the SAME 6-combination experiment (attack={left,right} x defend=
{left,center,right}, 10 rallies each) for each of the 4 controllers in
nash_skills.defend_controllers.CONTROLLER_NAMES, real PPO-driven hitting
throughout (the established _build_ppo_obs / ppo.predict / action-
construction pattern from nash_skills.v2.collect_data). Also uses
skill_profile="aggressive" and reset_mode="ready" this time -- this
project's own collect_data.py defaults, and what the teammate's V3 smoke
test used. (reset_mode="ready" is aliased to "clean" in
mujoco_env_comp.py, so this specific change is a no-op either way --
included anyway to remove any doubt, since skill_profile is NOT a no-op
and matching it matters.)

Does NOT touch the production nash_skills.env_wrapper / defend_steering /
defend_engagement classes -- this script disables SkillEnv's own
set_defend_positions() override path (by never calling it / forcing
env._defend1 = env._defend2 = None) and implements its own per-step
override inline, so all 4 controllers can be tested through one harness
without 4 separate production code paths. This is a diagnostic tool to
decide whether/how to change the production design, not a replacement
for it.

Role determination (who is currently defending) still uses
nash_skills.defend_role.DefenderRoleTracker, completely UNCHANGED -- this
is deliberately not part of the A/B.

Run:
    MUJOCO_GL=egl PYTHONPATH=. venv/bin/python scripts/ab_defend_controllers.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import io
import statistics
from contextlib import redirect_stdout

import numpy as np
from stable_baselines3 import PPO

from nash_skills.env_wrapper import SkillEnv
from nash_skills.defend_positions import DEFEND_POSITION_NAMES, world_defend_qpos
from nash_skills.defend_role import DefenderRoleTracker
from nash_skills.defend_controllers import compute_action, CONTROLLER_NAMES
from nash_skills.winner_inference import infer_terminal_winner
from nash_skills.v2.collect_data import _build_ppo_obs

PPO_MODEL_PATH = "logs/best_model_tracker1/best_model"
N_RALLIES_PER_COMBO = 10
ATTACK_SKILLS = ["left", "right"]
MAX_STEPS = 1000
TOLERANCE = 0.01


def _swallow_step(env, action):
    buf = io.StringIO()
    with redirect_stdout(buf):
        return env.step(action)


def run_one_rally(env, ppo, controller: str, ego_attack: str, opp_defend: str,
                   role_tracker: DefenderRoleTracker):
    """
    Ego (player 1) always attacks with `ego_attack`; opp (player 2)
    always defends toward `opp_defend` under the given controller design
    whenever the ball is heading their way. Only player 2's gantry is
    ever overridden -- player 1 plays entirely under its own PPO policy
    the whole rally, exactly like the project's existing fixed-role
    mechanistic-validation harness (test_defend_physics.py), so results
    are directly comparable to the earlier (not-clean) run.
    """
    env.set_skills(ego_attack, ego_attack)
    env._defend1, env._defend2 = None, None  # disable SkillEnv's own
                                              # override path entirely --
                                              # this script drives P2's
                                              # gantry itself, below
    role_tracker.reset()
    obs, info = env.reset()

    target = world_defend_qpos(2, opp_defend)
    reached_before = False
    release_velocity = None  # |qvel| of P2's gantry at the exact step
                              # `reached` first flips True this rally
    steps = 0
    done = False

    while True:
        ppo1 = _build_ppo_obs(obs, info, player=1)
        ppo2 = _build_ppo_obs(obs, info, player=2)
        a1, _ = ppo.predict(ppo1, deterministic=True)
        a2, _ = ppo.predict(ppo2, deterministic=True)
        action = np.zeros(18, dtype=np.float32)
        action[:9] = a1[:9]
        action[9:] = a2[:9]

        ball_x = float(env._env.data.body('ball').xpos[0])
        ball_vel_x = float(env._env.data.qvel[-6])
        defending_player = role_tracker.update(ball_x, ball_vel_x)

        if defending_player == 2:
            qpos2 = (float(env._env.data.qpos[9]), float(env._env.data.qpos[10]))
            gantry_step = env._env.gantry_speed_scale * 10.0 / 1000.0
            override, reached_now = compute_action(
                controller, qpos2, target, gantry_step, TOLERANCE, reached_before,
            )
            if reached_now and not reached_before:
                vx = float(env._env.data.qvel[9])
                vy = float(env._env.data.qvel[10])
                release_velocity = (vx * vx + vy * vy) ** 0.5
            reached_before = reached_now
            if override is not None:
                action[9], action[10] = override

        obs, _r, done, _t, info = _swallow_step(env, action)
        steps += 1
        if done or steps >= MAX_STEPS:
            break

    winner = infer_terminal_winner(obs, info, fallback="position") if done else None

    return {
        "winner": winner,
        "steps": steps,
        "target_reached": reached_before,
        "release_velocity": release_velocity,
    }


def _summarize(rows):
    n = len(rows)
    ego_wins = sum(1 for r in rows if r["winner"] == "ego")
    opp_wins = sum(1 for r in rows if r["winner"] == "opp")
    trunc = sum(1 for r in rows if r["winner"] is None)
    reached = [r for r in rows if r["target_reached"]]
    vels = [r["release_velocity"] for r in reached if r["release_velocity"] is not None]
    return {
        "n": n,
        "ego_wins": ego_wins,
        "opp_wins": opp_wins,
        "trunc": trunc,
        "reached_rate": len(reached) / n,
        "avg_len": statistics.mean(r["steps"] for r in rows),
        "avg_release_vel": statistics.mean(vels) if vels else None,
        "max_release_vel": max(vels) if vels else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rallies", type=int, default=N_RALLIES_PER_COMBO)
    ap.add_argument("--controllers", nargs="+", default=list(CONTROLLER_NAMES),
                     choices=list(CONTROLLER_NAMES))
    args = ap.parse_args()

    env = SkillEnv(proc_id=1, history=4, reset_mode="ready", skill_profile="aggressive")
    print(f"Loading PPO from {PPO_MODEL_PATH} ...")
    ppo = PPO.load(PPO_MODEL_PATH, device="cpu")
    role_tracker = DefenderRoleTracker()

    all_results = {}
    for controller in args.controllers:
        print(f"\n{'='*70}\ncontroller = {controller}\n{'='*70}")
        results = []
        for attack in ATTACK_SKILLS:
            for defend in DEFEND_POSITION_NAMES:
                rows = [
                    run_one_rally(env, ppo, controller, attack, defend, role_tracker)
                    for _ in range(args.rallies)
                ]
                results.extend((attack, defend, r) for r in rows)
                s = _summarize(rows)
                vel_str = (f"avg={s['avg_release_vel']:.4f} max={s['max_release_vel']:.4f}"
                           if s['avg_release_vel'] is not None else "n/a")
                print(f"  attack={attack:<6} defend={defend:<7} "
                      f"ego_wins={s['ego_wins']}/{s['n']} opp_wins={s['opp_wins']}/{s['n']} "
                      f"trunc={s['trunc']}/{s['n']} avg_len={s['avg_len']:.1f} "
                      f"reached={s['reached_rate']:.2f} release_vel[{vel_str}]")
        all_results[controller] = results

    env.close()

    print(f"\n{'='*70}\ncross-controller summary (aggregated over all 6 combos)\n{'='*70}")
    print(f"{'controller':<22}{'ego_wr':<9}{'opp_wr':<9}{'trunc':<9}{'reached':<10}"
          f"{'avg_rel_vel':<13}{'max_rel_vel':<13}")
    for controller in args.controllers:
        rows = [r for _a, _d, r in all_results[controller]]
        s = _summarize(rows)
        avg_v = f"{s['avg_release_vel']:.4f}" if s['avg_release_vel'] is not None else "n/a"
        max_v = f"{s['max_release_vel']:.4f}" if s['max_release_vel'] is not None else "n/a"
        print(f"{controller:<22}{s['ego_wins']/s['n']:<9.2f}{s['opp_wins']/s['n']:<9.2f}"
              f"{s['trunc']/s['n']:<9.2f}{s['reached_rate']:<10.2f}{avg_v:<13}{max_v:<13}")


if __name__ == "__main__":
    main()
