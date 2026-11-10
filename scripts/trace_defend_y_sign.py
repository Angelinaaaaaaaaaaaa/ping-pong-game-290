"""
Diagnostic: does defend_positions.py's y convention (left=+0.35,
right=-0.35 -- designed for world_defend_qpos / the gantry-qpos consumer)
actually match the RACKET-POSE-TARGET y sign convention that
defend_target_pose() silently repurposes it for (fed straight into
set_target_pose/set_target_pose_opp)?

skills.py is explicit and unambiguous about the racket-pose-target sign:
    "side_target: lateral aim, multiplied by 0.38m (negative=left, 0=center,
    positive=right)"
i.e. commanding attack skill "left" sends the ball toward NEGATIVE y on the
receiving court; "right" sends it toward POSITIVE y.

This script never calls set_defend_target_positions (pure "off" baseline,
identical in spirit to trace_real_defend_target.py) and instead reads
env._env.curr_target_opp -- the REAL, already-correct racket-pose target
update_target_racket_pose_opp computes reactively from the ball -- during
player-2-defending phases, once under attack="left" and once under
attack="right". If player 2's real racket-pose target goes to NEGATIVE y
when receiving a "left"-aimed shot and POSITIVE y when receiving "right",
that confirms defend_positions.py's left=+0.35/right=-0.35 is the OPPOSITE
sign of what Option B needs -- i.e. defend_target_pose("left") currently
steers the racket to where "right" should go, and vice versa.

Run (server only, MuJoCo required):
    MUJOCO_GL=egl PYTHONPATH=. venv/bin/python scripts/trace_defend_y_sign.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import io
from contextlib import redirect_stdout

import numpy as np
from stable_baselines3 import PPO

from nash_skills.env_wrapper import SkillEnv
from nash_skills.v2.collect_data import _build_ppo_obs
from nash_skills.defend_positions import DEFEND_POSITIONS

PPO_MODEL_PATH = "logs/best_model_tracker1/best_model"
N_RALLIES_PER_ATTACK = 20
N_STEPS_TO_TRACE = 300


def _swallow_step(env, action):
    buf = io.StringIO()
    with redirect_stdout(buf):
        return env.step(action)


def trace_one_attack(env, ppo, attack: str):
    """Returns list of (diff_pos_opp_norm, target_opp_xyz) at closest-approach
    for each rally where player 2 had a defending phase."""
    results = []
    for rally_idx in range(N_RALLIES_PER_ATTACK):
        env.set_skills(attack, attack)
        obs, info = env.reset()

        min_seen = None
        min_seen_xyz = None
        for step in range(N_STEPS_TO_TRACE):
            ppo1 = _build_ppo_obs(obs, info, player=1)
            ppo2 = _build_ppo_obs(obs, info, player=2)
            a1, _ = ppo.predict(ppo1, deterministic=True)
            a2, _ = ppo.predict(ppo2, deterministic=True)
            action = np.zeros(18, dtype=np.float32)
            action[:9] = a1[:9]
            action[9:] = a2[:9]
            obs, _r, done, _t, info = _swallow_step(env, action)

            ball_x = float(env._env.data.body('ball').xpos[0])
            ball_vel_x = float(env._env.data.qvel[-6])
            role = env._role_tracker.update(ball_x, ball_vel_x)
            diff_pos_opp_norm = float(np.linalg.norm(info['diff_pos_opp']))
            target_opp_xyz = env._env.curr_target_opp[:3].copy()

            if role == 2 and (min_seen is None or diff_pos_opp_norm < min_seen):
                min_seen = diff_pos_opp_norm
                min_seen_xyz = target_opp_xyz

            if done:
                break

        if min_seen is not None:
            results.append((min_seen, min_seen_xyz))
    return results


def main():
    env = SkillEnv(proc_id=1, history=4, reset_mode="ready", skill_profile="aggressive")
    print(f"Loading PPO from {PPO_MODEL_PATH} ...")
    ppo = PPO.load(PPO_MODEL_PATH, device="cpu")

    print(f"defend_positions.py says: left={DEFEND_POSITIONS['left']}  "
          f"right={DEFEND_POSITIONS['right']}  center={DEFEND_POSITIONS['center']}")
    print("skills.py says (racket-pose-target convention): negative=left, positive=right\n")

    for attack in ["left", "right"]:
        print(f"-- attack={attack} (ball aimed toward {attack} side of receiving court) --")
        results = trace_one_attack(env, ppo, attack)
        close = [(d, xyz) for d, xyz in results if d < 0.05]
        print(f"  {len(close)}/{len(results)} player-2-defending phases got within 0.05 "
              f"(plausible real contact)")
        pool = close if close else results
        if not close:
            print("  (no phase got close enough to trust -- showing all phases instead)")
        if pool:
            import statistics as _stats
            xs = [xyz[0] for _, xyz in pool]
            ys = [xyz[1] for _, xyz in pool]
            zs = [xyz[2] for _, xyz in pool]
            print(f"  real curr_target_opp at closest-approach: "
                  f"x=[{min(xs):.3f},{max(xs):.3f}] mean={sum(xs)/len(xs):.3f}  "
                  f"y=[{min(ys):.3f},{max(ys):.3f}] mean={sum(ys)/len(ys):.3f} median={_stats.median(ys):.3f}  "
                  f"z=[{min(zs):.3f},{max(zs):.3f}] mean={sum(zs)/len(zs):.3f}")
            print(f"  all y values, sorted: {sorted(round(y, 3) for y in ys)}")

    env.close()

    print("\n=== verdict ===")
    print("If attack='left' phases show NEGATIVE target_opp.y and attack='right' phases show")
    print("POSITIVE target_opp.y: that matches skills.py's convention, and")
    print("defend_positions.py's left=+0.35/right=-0.35 is the OPPOSITE sign --")
    print("defend_target_pose('left') currently sends the racket to where 'right' should go.")


if __name__ == "__main__":
    main()
