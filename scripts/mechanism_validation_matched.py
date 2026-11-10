"""
Confound-free, matched-pair mechanism validation for the attacker/
defender defend-steering mechanism.

Per the user's explicit design: hold attack skill fixed and IDENTICAL
for both players (not resampled) so defend-steering's effect is
isolated from any attack-skill asymmetry, while varying which physical
player receives defend assistance (the "P1/P2 orientation swap", to
control for any physical-player-specific asymmetry distinct from
skill-identity asymmetry) and toward which target.

Matched defend_on/defend_off pairs share the exact same starting
conditions via a non-obvious mechanism, confirmed by reading
mujoco_env_comp.py directly before building this: KukaTennisEnv.reset()
accepts a `seed` parameter but NEVER USES it internally -- every actual
source of reset-time randomness (reset_ball_throw(): serve position
offset, serve velocity, which player serves) draws from the GLOBAL
np.random state via np.random.uniform/np.random.choice, not a seeded
local Generator. True matching is achieved here by calling
np.random.seed(x) immediately before EACH reset() call, with the SAME x
for both the defend_on and defend_off member of a pair -- passing
seed=x to reset() itself would silently do nothing.

Design:
  attack_skill  in {left, right}      -- fixed, identical for both
                                          players the whole rally
  defend_target in {left, center, right} -- what the tested player
                                          steers toward when defending,
                                          in the "on" condition
  tested_player in {1, 2}             -- which physical player receives
                                          defend assistance (the
                                          orientation swap); the OTHER
                                          player never has a defend
                                          position in either condition
  condition     in {on, off}          -- on: tested_player gets
                                          defend_target. off:
                                          tested_player gets None.
                                          Both players still always
                                          hold a skill choice via
                                          set_skills every rally --
                                          Design B's action-space is
                                          untouched; only the tested
                                          player's defend-position
                                          commitment toggles.

For each (attack_skill, defend_target, tested_player) cell, N matched
(on, off) pairs are run, one shared seed per pair index.

Metrics beyond win/loss, attributed to the tested player specifically
(per explicit request to avoid a misleading win-rate-only read): reuses
nash_skills.eval_matchup._capture_env_step / _parse_contact_lines for
contact/return-success counts (not reinvented); rally length;
truncation; target_reached (from the production env._engagement --
confirms the override actually released before the rally ended).

Does not touch nash_skills.env_wrapper / defend_steering /
defend_engagement / defend_role -- all reused unmodified, same
confirmed-best bangbang_release production path as every other
diagnostic this session.

Run:
    MUJOCO_GL=egl PYTHONPATH=. venv/bin/python \\
        scripts/mechanism_validation_matched.py --rallies-per-cell 50
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
    condition for the tested player; a real position name is "on".
    Returns per-rally metrics attributed to the tested physical player.
    """
    np.random.seed(seed)  # the actual, global source of reset-time
                           # randomness -- reset()'s own seed= param is
                           # a documented no-op in this codebase

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
        decisive = infer_terminal_winner(obs, info, fallback=None) is not None
    else:
        winner = None
        decisive = False

    if tested_player == 1:
        tested_wins = winner == "ego"
        tested_contacts, tested_successes = ego_contacts, ego_successes
    else:
        tested_wins = winner == "opp"
        tested_contacts, tested_successes = opp_contacts, opp_successes

    target_reached = env._engagement.is_done(tested_player) if defend_target is not None else None

    return {
        "winner": winner,
        "decisive": decisive,
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
    print(f"{n_cells} cells x {args.rallies_per_cell} matched pairs x 2 conditions = "
          f"{n_cells * args.rallies_per_cell * 2} total rally-attempts\n")

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
    print("CROSS-CELL SUMMARY")
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
          "for the tested player -- the override is net-harmful on this evidence if so.)")


if __name__ == "__main__":
    main()
