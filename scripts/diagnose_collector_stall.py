"""
Diagnostic: collect_data_attack_defend.py's medium-scale collection pass
stalled at ~0.4% acceptance (1/260 rallies) under the new reactive
defend mechanism with grid sampling -- the pre-reactive-mechanism pilot
run saw 71/71 accepted under random sampling. This traces a handful of
rallies under the EXACT same resampling pattern (fresh random joint
action for both players at every net crossing), printing why each one
is accepted or discarded, to find the actual failure mode rather than
guessing.

Run:
    MUJOCO_GL=cgl PYTHONPATH=. venv/bin/python3.12 scripts/diagnose_collector_stall.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stable_baselines3 import PPO

from nash_skills.env_wrapper import SkillEnv
from nash_skills.skills import SKILL_NAMES
from nash_skills.defend_positions import DEFEND_POSITION_NAMES
from nash_skills.v2.collect_data import _build_ppo_obs
from nash_skills.v2.labeling import detect_winner
from nash_skills.eval_matchup import _capture_env_step, _parse_contact_lines

PPO_MODEL_PATH = "logs/best_model_tracker1/best_model"
MAX_STEPS = 2500
N_RALLIES = 10


def sample_joint_action(rng):
    return (
        str(rng.choice(SKILL_NAMES)), str(rng.choice(DEFEND_POSITION_NAMES)),
        str(rng.choice(SKILL_NAMES)), str(rng.choice(DEFEND_POSITION_NAMES)),
    )


def main():
    rng = np.random.default_rng(0)
    env = SkillEnv(proc_id=1, reset_mode="ready", skill_profile="aggressive")
    print(f"Loading PPO from {PPO_MODEL_PATH} ...")
    model = PPO.load(PPO_MODEL_PATH)

    for rally_idx in range(N_RALLIES):
        ego_attack, ego_defend, opp_attack, opp_defend = sample_joint_action(rng)
        env.set_skills(ego_attack, opp_attack)
        env.set_defend_reactive_positions(ego_defend, opp_defend)
        obs, info = env.reset()

        n_decisions = 0
        prev_defending_player = env._role_tracker.defending_player
        steps = 0
        done = False
        resample_log = []

        while not done and steps < MAX_STEPS:
            ppo1 = _build_ppo_obs(obs, info, player=1)
            ppo2 = _build_ppo_obs(obs, info, player=2)
            a1, _ = model.predict(ppo1, deterministic=True)
            a2, _ = model.predict(ppo2, deterministic=True)
            action = np.zeros(18, dtype=np.float32)
            action[:9] = a1[:9]
            action[9:] = a2[:9]

            (obs, _reward, done, _trunc, info), lines = _capture_env_step(env, action)
            steps += 1

            defending_player = env._role_tracker.defending_player
            if defending_player is not None and defending_player != prev_defending_player:
                n_decisions += 1
                resample_log.append((steps, prev_defending_player, defending_player))
                ego_attack, ego_defend, opp_attack, opp_defend = sample_joint_action(rng)
                env.set_skills(ego_attack, opp_attack)
                env.set_defend_reactive_positions(ego_defend, opp_defend)

            prev_defending_player = defending_player

        winner = detect_winner([obs], done=done, info=info)
        accepted = done and n_decisions > 0 and winner in (1, 2)

        print(f"-- rally {rally_idx}: steps={steps}  done={done}  n_decisions={n_decisions}  "
              f"winner={winner}  ACCEPTED={accepted} --")
        if not accepted:
            reason = []
            if not done:
                reason.append(f"not done (hit {MAX_STEPS}-step cap or env truncated)")
            if n_decisions == 0:
                reason.append("zero decisions recorded (role never latched)")
            if winner not in (1, 2):
                reason.append(f"winner={winner!r} not in (1, 2)")
            print(f"   discard reason: {'; '.join(reason)}")
        print(f"   transitions (step, prev_defender, new_defender): {resample_log[:8]}"
              f"{' ...' if len(resample_log) > 8 else ''}")

    env.close()


if __name__ == "__main__":
    main()
