"""
Diagnostic: what does the REAL, working ball-reactive targeting
(update_target_racket_pose_opp, unmodified, no defend override at all)
actually compute for curr_target_opp while player 2 is defending --
compared against the FIXED guesses defend_target_pose() used
(DEFEND_POSE_X=-0.3, DEFEND_POSE_Z=0.66, fixed face-forward quaternion)?

Context: scripts/trace_defend_target_pose.py showed that neither a 0.35
nor a 0.12 lateral offset produces any contact, even when the gantry
gets close to the chosen lateral target during what should be the
plausible interception window. The remaining untested variable is
whether the FIXED height/orientation guess is simply wrong -- this
script checks that directly by recording what the real (contact-
achieving) mechanism's target actually looks like, instead of guessing
again and re-running the same blind trace-and-check loop a third time.

No defend override is used here at all -- env.set_defend_target_positions
is never called, so player 2 behaves exactly as in the "off" condition
(which we already know achieves real contact, e.g. attack=right cells
showed off_wr=0.36-0.44, avg_contacts=2.1-2.6 in the main validation).

Run:
    MUJOCO_GL=egl PYTHONPATH=. venv/bin/python scripts/trace_real_defend_target.py
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
from nash_skills.defend_target_pose import DEFEND_POSE_X, DEFEND_POSE_Z, _DEFEND_QUAT

PPO_MODEL_PATH = "logs/best_model_tracker1/best_model"
N_RALLIES = 6
N_STEPS_TO_TRACE = 300


def _swallow_step(env, action):
    buf = io.StringIO()
    with redirect_stdout(buf):
        return env.step(action)


def main():
    env = SkillEnv(proc_id=1, history=4, reset_mode="ready", skill_profile="aggressive")
    print(f"Loading PPO from {PPO_MODEL_PATH} ...")
    ppo = PPO.load(PPO_MODEL_PATH, device="cpu")

    print(f"Fixed guess used by defend_target_pose(): x={DEFEND_POSE_X}, z={DEFEND_POSE_Z}, "
          f"quat={np.round(_DEFEND_QUAT, 3)}\n")

    contacts_seen = []

    for rally_idx in range(N_RALLIES):
        env.set_skills("right", "right")
        # No defend override at all -- plain "off" behavior.
        obs, info = env.reset()

        print(f"\n-- rally {rally_idx} --")
        min_seen = None
        min_seen_target = None
        for step in range(N_STEPS_TO_TRACE):
            ppo1 = _build_ppo_obs(obs, info, player=1)
            ppo2 = _build_ppo_obs(obs, info, player=2)
            a1, _ = ppo.predict(ppo1, deterministic=True)
            a2, _ = ppo.predict(ppo2, deterministic=True)
            action = np.zeros(18, dtype=np.float32)
            action[:9] = a1[:9]
            action[9:] = a2[:9]
            obs, _r, done, _t, info = _swallow_step(env, action)

            # env._role_tracker.update() is normally only called as a side
            # effect of env_wrapper.py's defend-gating blocks (only runs
            # when set_defend_target_positions/set_defend_positions is
            # active) -- since this script deliberately never calls
            # either (pure "off" baseline), feed it directly here instead.
            ball_x = float(env._env.data.body('ball').xpos[0])
            ball_vel_x = float(env._env.data.qvel[-6])
            role = env._role_tracker.update(ball_x, ball_vel_x)
            diff_pos_opp_norm = float(np.linalg.norm(info['diff_pos_opp']))
            target_opp = env._env.curr_target_opp.copy()

            if step % 10 == 0:
                pos = target_opp[:3]
                print(f"  step={step:>4} role={str(role):>4} diff_pos_opp_norm={diff_pos_opp_norm:>8.4f} "
                      f"target_pos=({pos[0]:>6.3f},{pos[1]:>6.3f},{pos[2]:>6.3f})")

            if role == 2 and (min_seen is None or diff_pos_opp_norm < min_seen):
                min_seen = diff_pos_opp_norm
                min_seen_target = target_opp.copy()

            if done:
                print(f"  (done at step {step})")
                break

        if min_seen_target is not None:
            pos = min_seen_target[:3]
            quat = min_seen_target[3:]
            print(f"  BEST (closest) player-2-defending target this rally: min_diff_pos_opp_norm={min_seen:.4f}  "
                  f"target_pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})  target_quat={np.round(quat,3)}")
            contacts_seen.append((min_seen, pos.copy(), quat.copy()))
        else:
            print("  (player 2 never had a defending phase this rally)")

    env.close()

    print("\n=== summary across all player-2-defending phases seen ===")
    close_contacts = [c for c in contacts_seen if c[0] < 0.05]
    print(f"{len(close_contacts)}/{len(contacts_seen)} phases got within 0.05 (plausible real contact)")
    if close_contacts:
        xs = [c[1][0] for c in close_contacts]
        ys = [c[1][1] for c in close_contacts]
        zs = [c[1][2] for c in close_contacts]
        print(f"Real contact-time target_pos ranges: x=[{min(xs):.3f},{max(xs):.3f}]  "
              f"y=[{min(ys):.3f},{max(ys):.3f}]  z=[{min(zs):.3f},{max(zs):.3f}]")
        print(f"Fixed guess was: x={DEFEND_POSE_X}, z={DEFEND_POSE_Z} -- compare directly above.")


if __name__ == "__main__":
    main()
