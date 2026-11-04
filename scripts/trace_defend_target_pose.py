"""
Diagnostic trace for the Option B (target-pose mechanism) result: the
1200-rally matched-pair validation (scripts/mechanism_validation_target_pose.py)
came back WORSE than hard-override (avg_delta_tested_wr=-0.307 vs -0.243),
with reached=0.00 in 11/12 cells and avg_contacts=0.00 in 8/12 -- i.e. the
gantry essentially never gets anywhere near the defend target pose, and
every "on" rally ends fast and decisively (trunc=0.00 everywhere, avg_len
108-190 vs off's 500-830).

This traces ONE rally step-by-step to distinguish two very different
explanations:
  (A) the PPO IS trying to move toward the defend target (diff_pos_opp
      norm shrinking over time), just too slowly to arrive before the
      ball needs returning -- a TIMING problem, not a response problem.
  (B) the PPO essentially ignores the out-of-distribution target
      entirely (diff_pos_opp norm flat/not shrinking) -- it never
      learned any general "move toward curr_target" behavior, only
      specific responses to the 5 skill targets it was actually trained
      against.

Modeled directly on scripts/trace_defend_engagement.py's working pattern
(same PPO-driving loop, same env wrapper), swapping set_defend_positions
for set_defend_target_positions and tracking diff_pos_opp (what the PPO
actually sees/minimizes) instead of raw gantry qpos.

Run:
    MUJOCO_GL=egl PYTHONPATH=. venv/bin/python scripts/trace_defend_target_pose.py
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
N_STEPS_TO_TRACE = 300

# The earlier shrink-to-0.12 diagnostic (testing whether magnitude/timing
# was the problem) is reverted -- scripts/trace_real_defend_target.py
# found the real cause: DEFEND_POSE_X/Z were simply wrong (z off by
# ~0.16-0.21), now fixed in nash_skills/defend_target_pose.py. This traces
# the REAL "left" position (0.35, unmodified) with the corrected pose.


def _swallow_step(env, action):
    buf = io.StringIO()
    with redirect_stdout(buf):
        return env.step(action)


def main():
    env = SkillEnv(proc_id=1, history=4, reset_mode="ready", skill_profile="aggressive")
    print(f"Loading PPO from {PPO_MODEL_PATH} ...")
    ppo = PPO.load(PPO_MODEL_PATH, device="cpu")

    for rally_idx in range(N_RALLIES):
        env.set_skills("right", "right")
        env.set_defend_target_positions(None, "left")
        # (real 0.35 "left", corrected x/z pose)
        obs, info = env.reset()

        print(f"\n=== rally {rally_idx} ===")
        print(f"{'step':>4} {'ball_x':>8} {'role':>5} {'gantry2_y':>10} {'diff_pos_opp_norm':>18} {'engaged':>8} {'done_latch':>10}")

        first_engaged_step = None
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
            tracked_role = env._role_tracker.defending_player
            gantry2_y = float(env._env.data.qpos[10])
            diff_pos_opp_norm = float(np.linalg.norm(info['diff_pos_opp']))
            engaged = env._pose_engagement.ever_engaged(2)
            done_latch = env._pose_engagement.is_done(2)

            if engaged and first_engaged_step is None:
                first_engaged_step = step

            if step % 5 == 0 or done:
                print(f"{step:>4} {ball_x:>8.4f} {str(tracked_role):>5} {gantry2_y:>10.4f} "
                      f"{diff_pos_opp_norm:>18.4f} {str(engaged):>8} {str(done_latch):>10}")

            if done:
                print(f"  (done at step {step}, final ball_x={ball_x:.4f})")
                break

        print(f"  first_engaged_step={first_engaged_step}  final diff_pos_opp_norm={diff_pos_opp_norm:.4f}  "
              f"final gantry2_y={gantry2_y:.4f} (target y=0.35 for real 'left')")

    env.close()


if __name__ == "__main__":
    main()
