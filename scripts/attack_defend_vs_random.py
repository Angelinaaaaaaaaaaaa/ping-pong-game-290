"""
Attacker/defender defend-steering vs a RANDOM attacker -- the gap
flagged after mechanism_validation_matched.py and soft_bias_sweep.py:
every experiment in that track used a FIXED, deliberate attack skill
("left" or "right"), never a uniformly random one. This script fills
that gap with the minimal change needed: each rally draws ONE random
skill (uniform over all 5) and gives it to BOTH players for the whole
rally, replacing the fixed attack_skill loop -- same symmetric-attack
convention as mechanism_validation_matched.py, just randomized instead
of chosen. Everything else is identical: matched defend_on/defend_off
pairs sharing the same seed (np.random.seed(x) called globally right
before each reset() -- reset()'s own seed= parameter is a confirmed
no-op, see mechanism_validation_matched.py's docstring for how this was
verified), tested_player in {1,2} (P1/P2 orientation swap), the same
metrics (tested-player win rate, contacts, success rate, rally length,
truncation).

Uses the confirmed-best hard-override controller (bangbang_release --
nash_skills.defend_steering + defend_engagement, already wired into
production SkillEnv.set_defend_positions, unmodified) as the "on"
condition, for direct comparability with mechanism_validation_matched.py's
fixed-attack result (every one of 12 cells negative there,
avg_delta_tested_wr=-0.243, avg_delta_contacts=-2.065).

Run:
    MUJOCO_GL=egl PYTHONPATH=. venv/bin/python \\
        scripts/attack_defend_vs_random.py --rallies-per-cell 50
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import statistics

import numpy as np
from stable_baselines3 import PPO

from nash_skills.env_wrapper import SkillEnv
from nash_skills.skills import SKILL_NAMES
from nash_skills.defend_positions import DEFEND_POSITION_NAMES
from nash_skills.winner_inference import infer_terminal_winner
from nash_skills.eval_matchup import _capture_env_step, _parse_contact_lines
from nash_skills.v2.collect_data import _build_ppo_obs

PPO_MODEL_PATH = "logs/best_model_tracker1/best_model"
TESTED_PLAYERS = (1, 2)
MAX_STEPS = 1000
DEFAULT_RALLIES_PER_CELL = 50


def run_one_rally(env, ppo, tested_player: int, defend_target, seed: int):
    """
    One rally under a matched seed. defend_target=None is the "off"
    condition for the tested player; a real position name is "on".
    The attack skill is drawn uniformly at random (same draw for both
    players, consumed right after the seed call so it's still part of
    the reproducible matched sequence) instead of being fixed.
    """
    np.random.seed(seed)
    attack_skill = str(np.random.choice(SKILL_NAMES))

    env.set_skills(attack_skill, attack_skill)
    if tested_player == 1:
        env.set_defend_positions(defend_target, None)
    else:
        env.set_defend_positions(None, defend_target)
    obs, info = env.reset()

    steps = 0
    done = False
    ego_contacts = opp_contacts = ego_successes = opp_successes = 0

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

    target_reached = env._engagement.is_done(tested_player) if defend_target is not None else None

    return {
        "attack_skill": attack_skill,
        "winner": winner,
        "steps": steps,
        "tested_wins": tested_wins,
        "tested_contacts": tested_contacts,
        "tested_successes": tested_successes,
        "target_reached": target_reached,
    }


def _summarize(rows):
    n = len(rows)
    total_contacts = sum(r["tested_contacts"] for r in rows)
    return {
        "n": n,
        "tested_wr": sum(1 for r in rows if r["tested_wins"]) / n,
        "trunc": sum(1 for r in rows if r["winner"] is None) / n,
        "avg_len": statistics.mean(r["steps"] for r in rows),
        "median_len": statistics.median(r["steps"] for r in rows),
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
    args = ap.parse_args()

    env = SkillEnv(proc_id=1, history=4, reset_mode="ready", skill_profile="aggressive")
    print(f"Loading PPO from {PPO_MODEL_PATH} ...")
    ppo = PPO.load(PPO_MODEL_PATH, device="cpu")

    n_cells = len(DEFEND_POSITION_NAMES) * len(TESTED_PLAYERS)
    print(f"attack=random (drawn per rally, uniform over {SKILL_NAMES})")
    print(f"{n_cells} cells x {args.rallies_per_cell} matched pairs x 2 conditions = "
          f"{n_cells * args.rallies_per_cell * 2} total rally-attempts\n")

    seed_counter = args.base_seed
    all_cells = []
    cell_num = 0

    for defend_target in DEFEND_POSITION_NAMES:
        for tested_player in TESTED_PLAYERS:
            cell_num += 1
            on_rows, off_rows = [], []
            for i in range(args.rallies_per_cell):
                pair_seed = seed_counter
                seed_counter += 1
                on_rows.append(run_one_rally(env, ppo, tested_player, defend_target, pair_seed))
                off_rows.append(run_one_rally(env, ppo, tested_player, None, pair_seed))

            s_on = _summarize(on_rows)
            s_off = _summarize(off_rows)
            reached_rate = sum(1 for r in on_rows if r["target_reached"]) / len(on_rows)
            delta_wr = s_on["tested_wr"] - s_off["tested_wr"]
            delta_contacts = s_on["avg_contacts"] - s_off["avg_contacts"]

            print(f"[{cell_num}/{n_cells}] defend_target={defend_target:<7} "
                  f"tested_player={tested_player}  (reached={reached_rate:.2f})")
            _print_summary("on", s_on)
            _print_summary("off", s_off)
            print(f"    {'delta':<10} tested_wr={delta_wr:+.2f}  avg_contacts={delta_contacts:+.2f}\n")

            all_cells.append({
                "defend_target": defend_target, "tested_player": tested_player,
                "on": s_on, "off": s_off, "reached_rate": reached_rate,
                "delta_wr": delta_wr, "delta_contacts": delta_contacts,
            })

    env.close()

    print("=" * 80)
    print("CROSS-CELL SUMMARY (attacker uses a random skill each rally)")
    print("=" * 80)
    print(f"{'defend':<9}{'player':<8}{'on_wr':<8}{'off_wr':<8}{'d_wr':<8}"
          f"{'on_contacts':<13}{'off_contacts':<13}{'d_contacts':<11}{'reached':<9}")
    for c in all_cells:
        print(f"{c['defend_target']:<9}{c['tested_player']:<8}"
              f"{c['on']['tested_wr']:<8.2f}{c['off']['tested_wr']:<8.2f}{c['delta_wr']:<+8.2f}"
              f"{c['on']['avg_contacts']:<13.2f}{c['off']['avg_contacts']:<13.2f}"
              f"{c['delta_contacts']:<+11.2f}{c['reached_rate']:<9.2f}")

    avg_off_wr = statistics.mean(c["off"]["tested_wr"] for c in all_cells)
    avg_delta_wr = statistics.mean(c["delta_wr"] for c in all_cells)
    avg_delta_contacts = statistics.mean(c["delta_contacts"] for c in all_cells)
    print(f"\nTested player's raw win rate vs random attacker, NO defend override "
          f"(the actual 'vs random' baseline): {avg_off_wr:.3f}")
    print(f"Aggregate across all {len(all_cells)} cells: "
          f"avg_delta_tested_wr={avg_delta_wr:+.3f}  avg_delta_contacts={avg_delta_contacts:+.3f}")
    print("(negative delta means defend_on is WORSE than matched defend_off even against a "
          "random attacker.)")


if __name__ == "__main__":
    main()
