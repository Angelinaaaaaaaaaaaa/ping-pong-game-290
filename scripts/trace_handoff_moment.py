"""
Diagnostic: now that DefendPoseEngagementTracker's "done" latch actually
fires (reached went from 0.00 to ~1.00 in most cells after the racket-
pos-vs-target fix), why does avg_delta_tested_wr stay ~-0.27 and
on_contacts stay ~0.00 even in cells where reached=1.00?

Traces individual "on" rallies step by step around the handoff moment
(use_defend_target flipping True->False) to see: how much flight-time
budget is left when handoff happens, what diff_pos_opp_norm does right
after handoff (does the real ball-reactive target "jump" far away from
the pre-position pose, suggesting a discontinuity the policy was never
trained on?), and whether a contact ever happens after that point.

Run:
    MUJOCO_GL=cgl PYTHONPATH=. venv/bin/python3.12 scripts/trace_handoff_moment.py
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
from nash_skills.eval_matchup import _capture_env_step, _parse_contact_lines

PPO_MODEL_PATH = "logs/best_model_tracker1/best_model"
N_RALLIES = 6
MAX_STEPS = 400

# Matches cell [11/12] in mechanism_validation_target_pose.py: attack=right,
# defend_target=center, tested_player=1 -- reached=1.00, on_contacts=0.00.
ATTACK = "right"
DEFEND_TARGET = "center"
TESTED_PLAYER = 1


def main():
    env = SkillEnv(proc_id=1, history=4, reset_mode="ready", skill_profile="aggressive")
    print(f"Loading PPO from {PPO_MODEL_PATH} ...")
    ppo = PPO.load(PPO_MODEL_PATH, device="cpu")

    for rally_idx in range(N_RALLIES):
        np.random.seed(1000 + rally_idx)
        env.set_skills(ATTACK, ATTACK)
        if TESTED_PLAYER == 1:
            env.set_defend_target_positions(DEFEND_TARGET, None)
        else:
            env.set_defend_target_positions(None, DEFEND_TARGET)
        obs, info = env.reset()

        print(f"\n-- rally {rally_idx} --")
        was_using_target = False
        handoff_step = None
        contacts = 0
        done = False

        for step in range(MAX_STEPS):
            ppo1 = _build_ppo_obs(obs, info, player=1)
            ppo2 = _build_ppo_obs(obs, info, player=2)
            a1, _ = ppo.predict(ppo1, deterministic=True)
            a2, _ = ppo.predict(ppo2, deterministic=True)
            action = np.zeros(18, dtype=np.float32)
            action[:9] = a1[:9]
            action[9:] = a2[:9]

            (obs, _r, done, _t, info), lines = _capture_env_step(env, action)
            ec, oc, _es, _osu = _parse_contact_lines(lines)
            contacts += ec if TESTED_PLAYER == 1 else oc

            is_defending = (env._role_tracker.defending_player == TESTED_PLAYER)
            using_target = env._pose_engagement.is_done(TESTED_PLAYER) is False and is_defending
            diff_key = 'diff_pos' if TESTED_PLAYER == 1 else 'diff_pos_opp'
            diff_norm = float(np.linalg.norm(info[diff_key]))
            ball_x = float(env._env.data.body('ball').xpos[0])

            if was_using_target and not using_target and is_defending:
                handoff_step = step
                print(f"  step={step:>4} HANDOFF just happened (done latched) "
                      f"ball_x={ball_x:+.3f} diff_norm={diff_norm:.4f}")
            elif handoff_step is not None and step <= handoff_step + 15 and is_defending:
                print(f"  step={step:>4} post-handoff ball_x={ball_x:+.3f} "
                      f"diff_norm={diff_norm:.4f} contacts_so_far={contacts}")
            was_using_target = using_target

            if done:
                break

        print(f"  rally ended at step={step}, done={done}, handoff_step={handoff_step}, "
              f"total_tested_contacts={contacts}")

    env.close()


if __name__ == "__main__":
    main()
