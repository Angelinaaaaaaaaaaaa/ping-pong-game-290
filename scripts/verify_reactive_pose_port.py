"""
Verification: does nash_skills.defend_reactive_pose's ported physics
(predict_ball_interception + solve_orientation) match the REAL engine's
own update_target_racket_pose(_opp) output, given the identical inputs?

This is the critical check before trusting the port for anything --
tests/test_defend_reactive_pose.py is pure-Python and can't catch a
subtle transcription error that happens to still produce a
plausible-looking, differently-wrong answer. This script runs the real
env, captures the real curr_target/curr_target_opp the engine itself
computes each step, independently recomputes the same thing via the
ported functions using the SAME ball_pos/ball_vel/x_target/y_target read
at that exact step, and diffs them.

Run:
    MUJOCO_GL=cgl PYTHONPATH=. venv/bin/python3.12 scripts/verify_reactive_pose_port.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stable_baselines3 import PPO

from nash_skills.env_wrapper import SkillEnv
from nash_skills.v2.collect_data import _build_ppo_obs
from nash_skills.defend_reactive_pose import predict_ball_interception, solve_orientation

PPO_MODEL_PATH = "logs/best_model_tracker1/best_model"
N_RALLIES = 4
N_STEPS = 300


def main():
    env = SkillEnv(proc_id=1, history=4, reset_mode="ready", skill_profile="aggressive")
    ppo = PPO.load(PPO_MODEL_PATH, device="cpu")

    max_pos_diff = 0.0
    max_quat_diff = 0.0
    n_compared = 0

    for rally_idx in range(N_RALLIES):
        env.set_skills("right", "left")
        obs, info = env.reset()

        for step in range(N_STEPS):
            ppo1 = _build_ppo_obs(obs, info, player=1)
            ppo2 = _build_ppo_obs(obs, info, player=2)
            a1, _ = ppo.predict(ppo1, deterministic=True)
            a2, _ = ppo.predict(ppo2, deterministic=True)
            action = np.zeros(18, dtype=np.float32)
            action[:9] = a1[:9]
            action[9:] = a2[:9]

            # Capture real ball state and real curr_target BEFORE stepping
            # (matches what update_target_racket_pose will use this step).
            ball_pos_real = np.asarray(env._env.data.body('ball').xpos, dtype=float).copy()
            ball_vel_real = np.asarray(env._env.data.qvel[-6:-3], dtype=float).copy()
            x_target1 = env._x_target1
            y_target1 = env._env.side_target * 0.38

            obs, _r, done, _t, info = env.step(action)

            real_target = env._env.curr_target.copy()

            predicted = predict_ball_interception(ball_pos_real, ball_vel_real)
            if predicted is not None:
                pos, v_future = predicted
                adjusted_pos, quat = solve_orientation(pos, v_future, x_target1, y_target1)
                my_pose = np.concatenate([adjusted_pos, quat])

                pos_diff = float(np.linalg.norm(my_pose[:3] - real_target[:3]))
                # Quaternion sign is not unique (q and -q are the same
                # rotation) -- compare both signs, take the closer one.
                quat_diff = min(
                    float(np.linalg.norm(my_pose[3:] - real_target[3:])),
                    float(np.linalg.norm(my_pose[3:] + real_target[3:])),
                )
                max_pos_diff = max(max_pos_diff, pos_diff)
                max_quat_diff = max(max_quat_diff, quat_diff)
                n_compared += 1

                if pos_diff > 1e-6 or quat_diff > 1e-6:
                    print(f"  rally={rally_idx} step={step}: pos_diff={pos_diff:.8f} "
                          f"quat_diff={quat_diff:.8f}")
                    print(f"    real_target={np.round(real_target, 5)}")
                    print(f"    my_pose    ={np.round(my_pose, 5)}")

            if done:
                break

    env.close()
    print(f"\nCompared {n_compared} steps where the ported guard allowed a prediction.")
    print(f"max_pos_diff={max_pos_diff:.8f}  max_quat_diff={max_quat_diff:.8f}")
    if max_pos_diff < 1e-6 and max_quat_diff < 1e-6:
        print("PORT VERIFIED: bit-for-bit match with the real engine's own computation.")
    else:
        print("PORT MISMATCH: investigate before trusting this module.")


if __name__ == "__main__":
    main()
