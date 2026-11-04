"""
Confound-free, matched-pair mechanism validation for the velocity/
trajectory-informed reactive defend-position mechanism
(nash_skills.defend_reactive_pose / env.set_defend_reactive_positions) --
the fix for Option B's (defend_target_pose / set_defend_target_positions)
root cause: a static, ball-independent pose that creates a discontinuity
when control hands back to ball-reactive tracking (confirmed this
session via scripts/trace_handoff_moment.py: a 0.51-unit jump at
handoff, ten times the 0.05 convergence tolerance, with too little
remaining flight-time to close it). This mechanism instead recomputes a
real ball-trajectory interception prediction every step and adds a
lateral bias that fades to zero as the ball approaches -- no freeze, no
handoff, no discontinuity by construction.

Identical design to mechanism_validation_target_pose.py in every respect
except what the "on" condition calls (set_defend_reactive_positions
instead of set_defend_target_positions) and the per-rally diagnostic
(no engagement latch exists for this mechanism -- nothing to report as
"reached"; instead reports the final diff_pos(_opp)_norm at the last
defending step as a loose convergence proxy).

Run:
    MUJOCO_GL=cgl PYTHONPATH=. venv/bin/python3.12 \\
        scripts/mechanism_validation_reactive_pose.py --rallies-per-cell 8
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import statistics

import numpy as np
from stable_baselines3 import PPO

from nash_skills.env_wrapper import SkillEnv
from nash_skills.defend_positions import DEFEND_POSITION_NAMES
from nash_skills.winner_inference import infer_terminal_winner
from nash_skills.eval_matchup import _capture_env_step, _parse_contact_lines
from nash_skills.v2.collect_data import _build_ppo_obs

PPO_MODEL_PATH = "logs/best_model_tracker1/best_model"
ATTACK_SKILLS = ["left", "right"]
TESTED_PLAYERS = (1, 2)
DEFAULT_MAX_STEPS = 1000
DEFAULT_RALLIES_PER_CELL = 50


def run_one_rally(env, ppo, attack_skill: str, tested_player: int, defend_target, seed: int, max_steps: int):
    np.random.seed(seed)

    env.set_skills(attack_skill, attack_skill)
    if tested_player == 1:
        env.set_defend_reactive_positions(defend_target, None)
    else:
        env.set_defend_reactive_positions(None, defend_target)
    obs, info = env.reset()

    steps = 0
    done = False
    ego_contacts = opp_contacts = ego_successes = opp_successes = 0
    last_diff_norm = None

    while True:
        ppo1 = _build_ppo_obs(obs, info, player=1)
        ppo2 = _build_ppo_obs(obs, info, player=2)
        a1, _ = ppo.predict(ppo1, deterministic=True)
        a2, _ = ppo.predict(ppo2, deterministic=True)
        action = np.zeros(18, dtype=np.float32)
        action[:9] = a1[:9]
        action[9:] = a2[:9]

        (obs, _r, done, _t, info), lines = _capture_env_step(env, action)
        ec, oc, es, osu = _parse_contact_lines(lines)
        ego_contacts += ec
        opp_contacts += oc
        ego_successes += es
        opp_successes += osu
        steps += 1

        if env._role_tracker.defending_player == tested_player:
            diff_key = 'diff_pos' if tested_player == 1 else 'diff_pos_opp'
            last_diff_norm = float(np.linalg.norm(info[diff_key]))

        if done or steps >= max_steps:
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
        "last_diff_norm": last_diff_norm,
    }


def _summarize(rows):
    n = len(rows)
    total_contacts = sum(r["tested_contacts"] for r in rows)
    steps = [r["steps"] for r in rows]
    return {
        "n": n,
        "tested_wr": sum(1 for r in rows if r["tested_wins"]) / n,
        "trunc": sum(1 for r in rows if r["winner"] is None) / n,
        "avg_len": statistics.mean(steps),
        "median_len": statistics.median(steps),
        "avg_contacts": total_contacts / n,
        "tested_success_rate": (
            sum(r["tested_successes"] for r in rows) / total_contacts if total_contacts > 0 else None
        ),
    }


def _print_summary(label, s):
    tsr = f"{s['tested_success_rate']:.2f}" if s["tested_success_rate"] is not None else "n/a"
    print(f"    {label:<10} tested_wr={s['tested_wr']:.2f}  trunc={s['trunc']:.2f}  "
          f"avg_len={s['avg_len']:.1f}  median_len={s['median_len']:.1f}  "
          f"avg_contacts={s['avg_contacts']:.2f}  tested_success_rate={tsr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rallies-per-cell", type=int, default=DEFAULT_RALLIES_PER_CELL)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    args = ap.parse_args()

    env = SkillEnv(proc_id=1, history=4, reset_mode="ready", skill_profile="aggressive")
    print(f"Loading PPO from {PPO_MODEL_PATH} ...")
    ppo = PPO.load(PPO_MODEL_PATH, device="cpu")

    n_cells = len(ATTACK_SKILLS) * len(DEFEND_POSITION_NAMES) * len(TESTED_PLAYERS)
    print(f"Reactive mechanism: {n_cells} cells x {args.rallies_per_cell} "
          f"matched pairs x 2 conditions = {n_cells * args.rallies_per_cell * 2} total rally-attempts\n")

    seed_counter = args.base_seed
    all_cells = []
    cell_num = 0

    for attack in ATTACK_SKILLS:
        for defend_target in DEFEND_POSITION_NAMES:
            for tested_player in TESTED_PLAYERS:
                cell_num += 1
                on_rows, off_rows = [], []
                for i in range(args.rallies_per_cell):
                    pair_seed = seed_counter
                    seed_counter += 1
                    on_rows.append(run_one_rally(env, ppo, attack, tested_player, defend_target, pair_seed, args.max_steps))
                    off_rows.append(run_one_rally(env, ppo, attack, tested_player, None, pair_seed, args.max_steps))

                s_on = _summarize(on_rows)
                s_off = _summarize(off_rows)
                diffs = [r["last_diff_norm"] for r in on_rows if r["last_diff_norm"] is not None]
                avg_final_diff = statistics.mean(diffs) if diffs else float("nan")
                delta_wr = s_on["tested_wr"] - s_off["tested_wr"]
                delta_contacts = s_on["avg_contacts"] - s_off["avg_contacts"]

                print(f"[{cell_num}/{n_cells}] attack={attack:<6} defend_target={defend_target:<7} "
                      f"tested_player={tested_player}  (avg_final_diff_norm={avg_final_diff:.3f})")
                _print_summary("on", s_on)
                _print_summary("off", s_off)
                print(f"    {'delta':<10} tested_wr={delta_wr:+.2f}  avg_contacts={delta_contacts:+.2f}\n")

                all_cells.append({
                    "attack": attack, "defend_target": defend_target, "tested_player": tested_player,
                    "on": s_on, "off": s_off, "avg_final_diff_norm": avg_final_diff,
                    "delta_wr": delta_wr, "delta_contacts": delta_contacts,
                })

    env.close()

    print("=" * 90)
    print("CROSS-CELL SUMMARY (velocity/trajectory-informed reactive mechanism)")
    print("=" * 90)
    print(f"{'attack':<8}{'defend':<9}{'player':<8}{'on_wr':<8}{'off_wr':<8}{'d_wr':<8}"
          f"{'on_contacts':<13}{'off_contacts':<13}{'d_contacts':<11}")
    for c in all_cells:
        print(f"{c['attack']:<8}{c['defend_target']:<9}{c['tested_player']:<8}"
              f"{c['on']['tested_wr']:<8.2f}{c['off']['tested_wr']:<8.2f}{c['delta_wr']:<+8.2f}"
              f"{c['on']['avg_contacts']:<13.2f}{c['off']['avg_contacts']:<13.2f}"
              f"{c['delta_contacts']:<+11.2f}")

    avg_delta_wr = statistics.mean(c["delta_wr"] for c in all_cells)
    avg_delta_contacts = statistics.mean(c["delta_contacts"] for c in all_cells)
    print(f"\nAggregate across all {len(all_cells)} cells: "
          f"avg_delta_tested_wr={avg_delta_wr:+.3f}  avg_delta_contacts={avg_delta_contacts:+.3f}")
    print("(compare directly against hard-override's -0.243, target-pose's -0.271..-0.308, and "
          "soft-bias's monotonic harm -- this is the first mechanism whose target was never "
          "ball-independent, so if it's still negative, the problem is not the discontinuity.)")


if __name__ == "__main__":
    main()
