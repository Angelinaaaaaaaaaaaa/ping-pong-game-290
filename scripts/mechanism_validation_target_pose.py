"""
Confound-free, matched-pair mechanism validation for Option B: defend
positioning expressed as a racket-POSE target
(nash_skills.defend_target_pose), fed through the SAME already-trained
target-tracking interface every attack skill already uses, instead of
overriding the PPO's gantry ACTION directly
(nash_skills.defend_steering's hard-override, or defend_soft_bias's
blend -- both already found net-harmful at every tested setting,
against both fixed and random attackers, across 1200 + 2400 + 600
matched rally-attempts in scripts/mechanism_validation_matched.py,
soft_bias_sweep.py, and attack_defend_vs_random.py).

Identical design to mechanism_validation_matched.py in every respect
EXCEPT what the "on" condition actually does -- same matched seeds
(np.random.seed(x) before each reset(), since reset()'s own seed=
parameter is a confirmed no-op), same tested_player P1/P2 orientation
swap, same metrics, same scale. This is deliberate: it's the only way to
tell whether Option B actually closes the gap that hard-override/
soft-bias left open, on a directly comparable basis.

"on" calls env.set_defend_target_positions(...) (new, opt-in,
nash_skills/env_wrapper.py) instead of set_defend_positions(...) --
nothing else about the harness changes. target_reached now reads
env._pose_engagement (Option B's own, separate engagement tracker) in
place of env._engagement (the hard-override mechanism's).

Run:
    MUJOCO_GL=egl PYTHONPATH=. venv/bin/python \\
        scripts/mechanism_validation_target_pose.py --rallies-per-cell 50
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
MAX_STEPS = 1000
DEFAULT_RALLIES_PER_CELL = 50


def run_one_rally(env, ppo, attack_skill: str, tested_player: int, defend_target, seed: int):
    """
    One rally under a matched seed. defend_target=None is the "off"
    condition for the tested player; a real position name is "on"
    (Option B: fed as a racket-pose target, not a gantry-action
    override). Returns per-rally metrics attributed to the tested
    physical player.
    """
    np.random.seed(seed)

    env.set_skills(attack_skill, attack_skill)
    if tested_player == 1:
        env.set_defend_target_positions(defend_target, None)
    else:
        env.set_defend_target_positions(None, defend_target)
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

    target_reached = env._pose_engagement.is_done(tested_player) if defend_target is not None else None

    return {
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
    args = ap.parse_args()

    env = SkillEnv(proc_id=1, history=4, reset_mode="ready", skill_profile="aggressive")
    print(f"Loading PPO from {PPO_MODEL_PATH} ...")
    ppo = PPO.load(PPO_MODEL_PATH, device="cpu")

    n_cells = len(ATTACK_SKILLS) * len(DEFEND_POSITION_NAMES) * len(TESTED_PLAYERS)
    print(f"Option B (target-pose mechanism): {n_cells} cells x {args.rallies_per_cell} "
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
                    on_rows.append(run_one_rally(env, ppo, attack, tested_player, defend_target, pair_seed))
                    off_rows.append(run_one_rally(env, ppo, attack, tested_player, None, pair_seed))

                s_on = _summarize(on_rows)
                s_off = _summarize(off_rows)
                reached_rate = sum(1 for r in on_rows if r["target_reached"]) / len(on_rows)
                delta_wr = s_on["tested_wr"] - s_off["tested_wr"]
                delta_contacts = s_on["avg_contacts"] - s_off["avg_contacts"]

                print(f"[{cell_num}/{n_cells}] attack={attack:<6} defend_target={defend_target:<7} "
                      f"tested_player={tested_player}  (reached={reached_rate:.2f})")
                _print_summary("on", s_on)
                _print_summary("off", s_off)
                print(f"    {'delta':<10} tested_wr={delta_wr:+.2f}  avg_contacts={delta_contacts:+.2f}\n")

                all_cells.append({
                    "attack": attack, "defend_target": defend_target, "tested_player": tested_player,
                    "on": s_on, "off": s_off, "reached_rate": reached_rate,
                    "delta_wr": delta_wr, "delta_contacts": delta_contacts,
                })

    env.close()

    print("=" * 90)
    print("CROSS-CELL SUMMARY (Option B: target-pose mechanism)")
    print("=" * 90)
    print(f"{'attack':<8}{'defend':<9}{'player':<8}{'on_wr':<8}{'off_wr':<8}{'d_wr':<8}"
          f"{'on_contacts':<13}{'off_contacts':<13}{'d_contacts':<11}{'reached':<9}")
    for c in all_cells:
        print(f"{c['attack']:<8}{c['defend_target']:<9}{c['tested_player']:<8}"
              f"{c['on']['tested_wr']:<8.2f}{c['off']['tested_wr']:<8.2f}{c['delta_wr']:<+8.2f}"
              f"{c['on']['avg_contacts']:<13.2f}{c['off']['avg_contacts']:<13.2f}"
              f"{c['delta_contacts']:<+11.2f}{c['reached_rate']:<9.2f}")

    avg_delta_wr = statistics.mean(c["delta_wr"] for c in all_cells)
    avg_delta_contacts = statistics.mean(c["delta_contacts"] for c in all_cells)
    print(f"\nAggregate across all {len(all_cells)} cells: "
          f"avg_delta_tested_wr={avg_delta_wr:+.3f}  avg_delta_contacts={avg_delta_contacts:+.3f}")
    print("(negative delta_wr or delta_contacts means defend_on is WORSE than matched defend_off "
          "for the tested player -- compare this directly against mechanism_validation_matched.py's "
          "avg_delta_tested_wr=-0.243 and soft_bias_sweep.py's results to see whether the target-pose "
          "channel actually closes the gap.)")


if __name__ == "__main__":
    main()
