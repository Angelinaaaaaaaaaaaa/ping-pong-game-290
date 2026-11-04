"""
Soft-bias alpha sweep: validates whether BLENDING (rather than
REPLACING) the defending player's PPO gantry command with a gentle
proportional pull toward their defend target avoids the harm found in
the hard-override mechanism validation (scripts/mechanism_validation_matched.py,
1200 matched rally-attempts, every one of 12 cells negative,
avg_delta_tested_wr=-0.243, avg_delta_contacts=-2.065, harm persisting
even when the override released control back to the PPO before the
rally ended 94%+ of the time -- ruling out "release happens too late"
and pointing at the hard override itself pushing the PPO out of its
training distribution).

final_command = clip(ppo_command + alpha * Kp * (target - current), -1, 1)
(nash_skills.defend_soft_bias.blend_gantry_command)

Sweeps alpha in {0, 0.1, 0.25, 0.5} -- alpha=0 is a built-in sanity
check (pure PPO, must show zero measured effect by construction) --
across the SAME confound-free design as the hard-override validation:
attack skill fixed/identical for both players, tested_player in {1,2}
(P1/P2 orientation swap), matched seeds shared ACROSS ALL FOUR alpha
values per (attack, defend_target, tested_player, pair_index) cell (not
just a pairwise on/off match this time) -- np.random.seed(x) called
globally right before each reset(), since KukaTennisEnv.reset()'s own
seed= parameter is a confirmed no-op (see mechanism_validation_matched.py's
docstring for how this was verified).

Does NOT touch nash_skills.env_wrapper / defend_steering /
defend_engagement -- bypasses SkillEnv's own hard-override path
entirely (same bypass pattern as ab_defend_controllers.py), since this
tests a different, not-yet-production steering law. Role determination
still uses nash_skills.defend_role.DefenderRoleTracker, completely
unchanged.

Tracks, per rally: win/loss, contacts/successes
(nash_skills.eval_matchup, reused not reinvented), rally length,
truncation, AND intervention_magnitude -- the actual
|final_command - ppo_command| applied, averaged over all
defending-phase steps -- "how hard did this override actually push,"
to find where intervention starts to hurt.

Run:
    MUJOCO_GL=egl PYTHONPATH=. venv/bin/python \\
        scripts/soft_bias_sweep.py --rallies-per-cell 50
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import statistics

import numpy as np
from stable_baselines3 import PPO

from nash_skills.env_wrapper import SkillEnv
from nash_skills.defend_positions import DEFEND_POSITION_NAMES, world_defend_qpos
from nash_skills.defend_role import DefenderRoleTracker
from nash_skills.defend_soft_bias import blend_gantry_command
from nash_skills.winner_inference import infer_terminal_winner
from nash_skills.eval_matchup import _capture_env_step, _parse_contact_lines
from nash_skills.v2.collect_data import _build_ppo_obs

PPO_MODEL_PATH = "logs/best_model_tracker1/best_model"
ATTACK_SKILLS = ["left", "right"]
TESTED_PLAYERS = (1, 2)
ALPHAS = (0.0, 0.1, 0.25, 0.5)
MAX_STEPS = 1000
DEFAULT_RALLIES_PER_CELL = 50


def run_one_rally(env, ppo, role_tracker, attack_skill: str, tested_player: int,
                   defend_target: str, alpha: float, seed: int):
    np.random.seed(seed)  # the real source of reset-time randomness --
                           # reset()'s own seed= param is a no-op here

    env.set_skills(attack_skill, attack_skill)
    env._defend1, env._defend2 = None, None  # disable SkillEnv's own
                                              # hard-override path --
                                              # this script blends the
                                              # gantry command itself
    role_tracker.reset()
    obs, info = env.reset()

    target = world_defend_qpos(tested_player, defend_target)
    steps = 0
    done = False
    ego_contacts = opp_contacts = ego_successes = opp_successes = 0
    intervention_mags = []

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

        if defending_player == tested_player:
            if tested_player == 1:
                qpos = (float(env._env.data.qpos[0]), float(env._env.data.qpos[1]))
                ppo_cmd = (float(a1[0]), float(a1[1]))
            else:
                qpos = (float(env._env.data.qpos[9]), float(env._env.data.qpos[10]))
                ppo_cmd = (float(a2[0]), float(a2[1]))

            final_cmd, mag = blend_gantry_command(ppo_cmd, qpos, target, alpha)
            intervention_mags.append(mag)
            if tested_player == 1:
                action[0], action[1] = final_cmd
            else:
                action[9], action[10] = final_cmd

        (obs, _r, done, _t, info), lines = _capture_env_step(env, action)
        ec, oc, es, osu = _parse_contact_lines(lines)
        ego_contacts += ec
        opp_contacts += oc
        ego_successes += es
        opp_successes += osu
        steps += 1

        if done or steps >= MAX_STEPS:
            break

    if done:
        winner = infer_terminal_winner(obs, info, fallback="position")
    else:
        winner = None

    if tested_player == 1:
        tested_wins = winner == "ego"
        tested_contacts, tested_successes = ego_contacts, ego_successes
    else:
        tested_wins = winner == "opp"
        tested_contacts, tested_successes = opp_contacts, opp_successes

    return {
        "winner": winner,
        "steps": steps,
        "tested_wins": tested_wins,
        "tested_contacts": tested_contacts,
        "tested_successes": tested_successes,
        "avg_intervention_mag": statistics.mean(intervention_mags) if intervention_mags else 0.0,
    }


def _summarize(rows):
    n = len(rows)
    total_contacts = sum(r["tested_contacts"] for r in rows)
    return {
        "n": n,
        "tested_wr": sum(1 for r in rows if r["tested_wins"]) / n,
        "trunc": sum(1 for r in rows if r["winner"] is None) / n,
        "avg_len": statistics.mean(r["steps"] for r in rows),
        "avg_contacts": total_contacts / n,
        "tested_success_rate": (
            sum(r["tested_successes"] for r in rows) / total_contacts if total_contacts > 0 else None
        ),
        "avg_intervention_mag": statistics.mean(r["avg_intervention_mag"] for r in rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rallies-per-cell", type=int, default=DEFAULT_RALLIES_PER_CELL)
    ap.add_argument("--base-seed", type=int, default=0)
    args = ap.parse_args()

    env = SkillEnv(proc_id=1, history=4, reset_mode="ready", skill_profile="aggressive")
    print(f"Loading PPO from {PPO_MODEL_PATH} ...")
    ppo = PPO.load(PPO_MODEL_PATH, device="cpu")
    role_tracker = DefenderRoleTracker()

    n_cells = len(ATTACK_SKILLS) * len(DEFEND_POSITION_NAMES) * len(TESTED_PLAYERS)
    total = n_cells * args.rallies_per_cell * len(ALPHAS)
    print(f"{n_cells} cells x {args.rallies_per_cell} matched seeds x {len(ALPHAS)} alphas "
          f"= {total} total rally-attempts\n")

    seed_counter = args.base_seed
    by_alpha = {a: [] for a in ALPHAS}
    cell_num = 0

    for attack in ATTACK_SKILLS:
        for defend_target in DEFEND_POSITION_NAMES:
            for tested_player in TESTED_PLAYERS:
                cell_num += 1
                cell_rows = {a: [] for a in ALPHAS}
                for i in range(args.rallies_per_cell):
                    pair_seed = seed_counter
                    seed_counter += 1
                    for alpha in ALPHAS:
                        row = run_one_rally(env, ppo, role_tracker, attack, tested_player,
                                             defend_target, alpha, pair_seed)
                        cell_rows[alpha].append(row)
                        by_alpha[alpha].append(row)

                print(f"[{cell_num}/{n_cells}] attack={attack:<6} defend_target={defend_target:<7} "
                      f"tested_player={tested_player}")
                for alpha in ALPHAS:
                    s = _summarize(cell_rows[alpha])
                    tsr = f"{s['tested_success_rate']:.2f}" if s["tested_success_rate"] is not None else "n/a"
                    print(f"    alpha={alpha:<5} tested_wr={s['tested_wr']:.2f}  trunc={s['trunc']:.2f}  "
                          f"avg_len={s['avg_len']:.1f}  avg_contacts={s['avg_contacts']:.2f}  "
                          f"success_rate={tsr}  avg_intervention_mag={s['avg_intervention_mag']:.4f}")
                print()

    env.close()

    print("=" * 90)
    print("CROSS-ALPHA SUMMARY (aggregated over all 12 cells)")
    print("=" * 90)
    baseline = _summarize(by_alpha[0.0])
    print(f"{'alpha':<8}{'tested_wr':<11}{'d_wr_vs_a0':<12}{'avg_contacts':<13}{'d_contacts_vs_a0':<18}"
          f"{'avg_len':<10}{'trunc':<8}{'avg_intervention_mag':<20}")
    for alpha in ALPHAS:
        s = _summarize(by_alpha[alpha])
        d_wr = s["tested_wr"] - baseline["tested_wr"]
        d_contacts = s["avg_contacts"] - baseline["avg_contacts"]
        print(f"{alpha:<8}{s['tested_wr']:<11.3f}{d_wr:<+12.3f}{s['avg_contacts']:<13.2f}"
              f"{d_contacts:<+18.2f}{s['avg_len']:<10.1f}{s['trunc']:<8.2f}{s['avg_intervention_mag']:<20.4f}")

    print("\n(d_wr_vs_a0 / d_contacts_vs_a0 are relative to alpha=0, i.e. pure PPO with zero "
          "intervention -- NOT the hard-override baseline. Negative means this alpha still "
          "hurts the tested player relative to no intervention at all.)")


if __name__ == "__main__":
    main()
