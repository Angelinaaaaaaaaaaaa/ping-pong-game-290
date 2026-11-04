"""
Symmetric defend-override diagnostic: P1/P2 asymmetry control.

Every fixed-role experiment so far this session (test_defend_physics.py,
ab_defend_controllers.py) only ever gave player 2 gantry assistance when
defending -- player 1 played pure PPO in every role, including when P1
was defending. The controller A/B found the best controller
(bangbang_release, the existing production design) still leaves
ego_wr=0.67 against a no-override baseline of ~0.05-0.10. This script
tests whether that gap is (partly or fully) an artifact of that
structural test asymmetry, by comparing three conditions under
IDENTICAL settings in one script:

  none       -- env.set_defend_positions(None, None)
  p2_only    -- env.set_defend_positions(None, defend_choice)
  symmetric  -- env.set_defend_positions(defend_choice, defend_choice)

IMPORTANT BOUNDARY (confirmed with the user before building this): this
only changes which physical player's gantry gets steering assistance at
a given instant (resolved by ball direction via the existing, unmodified
DefenderRoleTracker -- the same role resolution already used by the
p2_only condition, and unavoidable physics in any design, including real
ping-pong, since only one player is ever receiving a given shot). It
does NOT change the action-space: both players still always hold both a
skill choice (set_skills) and a defend-position choice
(set_defend_positions) every rally -- Design B's defining feature is
untouched. This is a mechanics-layer diagnostic, not a step toward
Design A's role-gated action space, even informally.

All three conditions use the existing, UNCHANGED production path
(SkillEnv.set_defend_positions -> defend_steering.py +
defend_engagement.py, already confirmed the best of 4 controllers in the
prior A/B) -- no bypassing, no new steering logic. Uses
skill_profile="aggressive", reset_mode="ready" (this project's own
collect_data.py defaults) throughout, to avoid a confound discovered
while planning this script: test_defend_physics.py's much-worse-looking
earlier P2-only numbers used skill_profile="current" by omission, so
they are NOT directly comparable to ab_defend_controllers.py's P2-only
"aggressive" row. This script reruns all three conditions fresh under
one consistent setting rather than comparing across scripts that
silently differed.

Metrics beyond terminal win/loss, per explicit request (to avoid a
"50/50 but both bad" false read): reuses nash_skills.eval_matchup's
existing _capture_env_step / _parse_contact_lines (already built,
already reused by the teammate's V3 pipeline) for per-rally contact and
return-success counts, not reinvented here.

Run:
    MUJOCO_GL=egl PYTHONPATH=. venv/bin/python scripts/ab_defend_symmetry.py
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
N_RALLIES_PER_CELL = 10
ATTACK_SKILLS = ["left", "right"]
CONDITIONS = ("none", "p2_only", "symmetric")
MAX_STEPS = 1000


def run_one_rally(env, ppo, condition: str, ego_attack: str, defend_choice):
    env.set_skills(ego_attack, ego_attack)
    if condition == "none":
        env.set_defend_positions(None, None)
    elif condition == "p2_only":
        env.set_defend_positions(None, defend_choice)
    elif condition == "symmetric":
        env.set_defend_positions(defend_choice, defend_choice)
    else:
        raise ValueError(f"Unknown condition '{condition}'")

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

    p1_reached = env._engagement.is_done(1) if condition == "symmetric" else None
    p2_reached = env._engagement.is_done(2) if condition in ("p2_only", "symmetric") else None

    return {
        "winner": winner,
        "decisive": decisive,
        "steps": steps,
        "ego_contacts": ego_contacts,
        "opp_contacts": opp_contacts,
        "ego_successes": ego_successes,
        "opp_successes": opp_successes,
        "p1_reached": p1_reached,
        "p2_reached": p2_reached,
    }


def _summarize(rows):
    n = len(rows)
    total_opp_contacts = sum(r["opp_contacts"] for r in rows)
    return {
        "n": n,
        "ego_wins": sum(1 for r in rows if r["winner"] == "ego"),
        "opp_wins": sum(1 for r in rows if r["winner"] == "opp"),
        "trunc": sum(1 for r in rows if r["winner"] is None),
        "avg_len": statistics.mean(r["steps"] for r in rows),
        "avg_opp_contacts": statistics.mean(r["opp_contacts"] for r in rows),
        "avg_opp_successes": statistics.mean(r["opp_successes"] for r in rows),
        "opp_success_rate": (
            sum(r["opp_successes"] for r in rows) / total_opp_contacts
            if total_opp_contacts > 0 else None
        ),
    }


def _print_row(label, s):
    osr = f"{s['opp_success_rate']:.2f}" if s["opp_success_rate"] is not None else "n/a"
    print(f"  {label:<28}ego_wins={s['ego_wins']}/{s['n']} opp_wins={s['opp_wins']}/{s['n']} "
          f"trunc={s['trunc']}/{s['n']} avg_len={s['avg_len']:.1f} "
          f"opp_contacts={s['avg_opp_contacts']:.2f} opp_successes={s['avg_opp_successes']:.2f} "
          f"opp_success_rate={osr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rallies", type=int, default=N_RALLIES_PER_CELL)
    args = ap.parse_args()

    env = SkillEnv(proc_id=1, history=4, reset_mode="ready", skill_profile="aggressive")
    print(f"Loading PPO from {PPO_MODEL_PATH} ...")
    ppo = PPO.load(PPO_MODEL_PATH, device="cpu")

    all_results = {}
    for condition in CONDITIONS:
        print(f"\n{'='*70}\ncondition = {condition}\n{'='*70}")
        results = []
        defends = ["n/a"] if condition == "none" else DEFEND_POSITION_NAMES
        for attack in ATTACK_SKILLS:
            for defend in defends:
                defend_choice = None if defend == "n/a" else defend
                rows = [
                    run_one_rally(env, ppo, condition, attack, defend_choice)
                    for _ in range(args.rallies)
                ]
                results.extend((attack, defend, r) for r in rows)
                _print_row(f"attack={attack} defend={defend}", _summarize(rows))
        all_results[condition] = results

    env.close()

    print(f"\n{'='*70}\ncross-condition summary (aggregated over all cells)\n{'='*70}")
    for condition in CONDITIONS:
        rows = [r for _a, _d, r in all_results[condition]]
        _print_row(f"[{condition}] ALL", _summarize(rows))


if __name__ == "__main__":
    main()
