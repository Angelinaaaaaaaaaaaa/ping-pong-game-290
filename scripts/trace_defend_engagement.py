"""
Diagnostic trace for the defend-engagement bug found in test_defend_physics.py:
roughly half of attack=left/defend=left rallies fully reach the target
(min_dist=0.0) and the rest show zero movement at all (min_dist=0.35) --
an all-or-nothing split, not a ran-out-of-time one.

UPDATE: the original symptom was fully explained by a test-harness bug
(this script used an all-zeros action instead of driving the PPO
controller, so no real hitting ever happened) -- now fixed. With real
hitting, a 4-rally trace showed clean role flips and steering that
converges smoothly every time, no stuck cases. It also showed the old
naive `ball_vel_x >= 0` role check passing through a small/noisy value
right at every hit (e.g. -0.0805), which motivated replacing it with
nash_skills.defend_role.DefenderRoleTracker (latch on a clear velocity
threshold, flip only on an actual net crossing).

This script now logs BOTH the old naive instantaneous role and the new
tracker's resolved role side by side, so the delayed-flip behavior (new
only flips once ball_x actually crosses TABLE_SHIFT, not merely when
ball_vel_x's sign changes at the hit) can be directly confirmed against
real data instead of assumed.

Run:
    MUJOCO_GL=egl PYTHONPATH=. venv/bin/python scripts/trace_defend_engagement.py
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

PPO_MODEL_PATH = "logs/best_model_tracker1/best_model"
N_RALLIES = 4
N_STEPS_TO_TRACE = 400


def _swallow_step(env, action):
    buf = io.StringIO()
    with redirect_stdout(buf):
        return env.step(action)


def main():
    env = SkillEnv(proc_id=1, history=4)
    print(f"Loading PPO from {PPO_MODEL_PATH} ...")
    ppo = PPO.load(PPO_MODEL_PATH, device="cpu")

    for rally_idx in range(N_RALLIES):
        env.set_skills("left", "left")
        env.set_defend_positions(None, "left")
        obs, info = env.reset()

        print(f"\n=== rally {rally_idx} ===")
        print(f"{'step':>4} {'ball_x':>8} {'ball_vel_x':>11} {'naive':>6} {'tracker':>8} {'gantry2_y':>10}")

        prev_naive = None
        prev_tracked = None
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
            naive_role = 2 if ball_vel_x >= 0 else 1  # old (fragile) instantaneous logic
            tracked_role = env._role_tracker.defending_player  # new latch+crossing logic
            gantry2_y = float(env._env.data.qpos[10])

            naive_changed = naive_role != prev_naive
            tracked_changed = tracked_role != prev_tracked
            if naive_changed or tracked_changed or step % 10 == 0:
                tags = []
                if naive_changed and prev_naive is not None:
                    tags.append("NAIVE FLIP")
                if tracked_changed and prev_tracked is not None:
                    tags.append("TRACKER FLIP")
                tag = "  <-- " + ", ".join(tags) if tags else ""
                print(f"{step:>4} {ball_x:>8.4f} {ball_vel_x:>11.4f} {str(naive_role):>6} {str(tracked_role):>8} {gantry2_y:>10.4f}{tag}")
            prev_naive = naive_role
            prev_tracked = tracked_role

            if done:
                print(f"  (done at step {step}, final ball_x={ball_x:.4f})")
                break

        final_y = float(env._env.data.qpos[10])
        print(f"  final gantry2_y = {final_y:.4f}  (target = 0.35)  "
              f"{'REACHED' if abs(final_y - 0.35) < 0.05 else 'STUCK'}")

    env.close()


if __name__ == "__main__":
    main()
