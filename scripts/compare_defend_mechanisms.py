"""
Head-to-head comparison of the two Design B defend mechanisms under one
matched-seed protocol (nash_skills.defend_mechanisms):

  reactive     the receiver's interception gets a fading lateral bias (current)
  preparation  the player who just hit moves to its defend pose while the ball
               travels away from it (paper Sec. 3.2)

--mode one-sided  (gate + "does the choice matter")
    Both players use the same attack skill k; only the tested player has a
    defend mechanism. Each (k, tested player, seed) is played 7 times from the
    same seed: no mechanism, then each mechanism with each of the 3 positions.
    center_safe is excluded: center_safe vs center_safe never ends even with
    no mechanism (documented in analysis/experiment_plan.md).

--mode both-sided  (Design B's real usage)
    Both players get a random static (attack, defend) profile per seed over all
    5 skills; each seed is played with no mechanism, reactive, and preparation.

Reading fixed before running:
  gate     mechanism minus no-mechanism, decided-only win rate of the tested
           player, pooled, 95% paired-bootstrap CI; passes if the CI's lower
           end is above -0.05
  content  matched minus mismatched position (same side as the incoming
           attack vs the opposite side); the defend axis carries strategic
           content if this CI excludes 0
  health   truncation rate and rally length (both modes)
  sanity   reactive "center" applies zero bias, so it must reproduce the
           no-mechanism rally exactly on the same seed

Matched seeds need one extra step: KukaTennisEnv.reset() does not reset the
racket targets (its reset_target() call is commented out), so each rally
starts from the previous rally's last target and same-seed replays diverge.
Workers therefore build SkillEnv(reset_targets=True), which restores both
targets to the construction-time home pose before every reset; with that,
same-seed replays are identical.

--prep-side-scale multiplies the left/right preparation poses' lateral offset
(defend_target_pose.DEFEND_POSE_Y) for the side-offset sweep; --ppo picks the
low-level controller checkpoint.

Run:
    MUJOCO_GL=cgl PYTHONPATH=. venv/bin/python3.12 scripts/compare_defend_mechanisms.py \\
        --mode one-sided --pairs 40 --workers 12
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import multiprocessing as mp
import time

import numpy as np

from nash_skills.skills import SKILL_NAMES
from nash_skills.defend_positions import DEFEND_POSITION_NAMES
from nash_skills.defend_mechanisms import DEFEND_MECHANISMS, set_defend_choices

PPO_MODEL_PATH = "logs/best_model_tracker1/best_model"
ONE_SIDED_SKILLS = ("left", "left_short", "right_short", "right")
SKILL_SIDE = {"left": "left", "left_short": "left", "right_short": "right", "right": "right"}
OPPOSITE = {"left": "right", "right": "left"}
DEFAULT_MAX_STEPS = 2500
N_BOOTSTRAP = 2000
GATE_FLOOR = -0.05

_ENV = None
_PPO = None
# Mechanisms compared against "off" (set per process by main/_init_worker).
MECHANISMS = ("reactive", "preparation")


def _init_worker(ppo_path=PPO_MODEL_PATH, side_scale=1.0, mechanisms=MECHANISMS, skill_profile="aggressive"):
    global _ENV, _PPO, MECHANISMS
    MECHANISMS = tuple(mechanisms)
    from stable_baselines3 import PPO
    import nash_skills.defend_target_pose as dtp
    from nash_skills.env_wrapper import SkillEnv
    dtp.DEFEND_POSE_Y = {k: v * side_scale for k, v in dtp.DEFEND_POSE_Y.items()}
    _ENV = SkillEnv(proc_id=1, history=4, reset_mode="ready", skill_profile=skill_profile,
                    reset_targets=True)
    _PPO = PPO.load(ppo_path, device="cpu")


def play_rally(seed, skills, defend1, defend2, mechanism, max_steps):
    """One rally from `seed`. mechanism=None plays with no defend mechanism."""
    from nash_skills.eval_matchup import _capture_env_step, _parse_contact_lines
    from nash_skills.winner_inference import infer_terminal_winner
    from nash_skills.v2.collect_data import _build_ppo_obs

    env, ppo = _ENV, _PPO
    np.random.seed(seed)
    env.set_skills(*skills)
    env.set_defend_reactive_positions(None, None)
    env.set_defend_preparation_positions(None, None)
    if mechanism is not None:
        set_defend_choices(env, mechanism, defend1, defend2)
    obs, info = env.reset()

    steps, done, returns = 0, False, [0, 0]
    while not done and steps < max_steps:
        a1, _ = ppo.predict(_build_ppo_obs(obs, info, player=1), deterministic=True)
        a2, _ = ppo.predict(_build_ppo_obs(obs, info, player=2), deterministic=True)
        action = np.zeros(18, dtype=np.float32)
        action[:9], action[9:] = a1[:9], a2[:9]
        (obs, _r, done, _t, info), lines = _capture_env_step(env, action)
        _ec, _oc, es, osu = _parse_contact_lines(lines)
        returns[0] += es
        returns[1] += osu
        steps += 1
    winner = infer_terminal_winner(obs, info, fallback="position") if done else None
    return {"winner": winner, "steps": steps, "returns": returns}


def _one_sided_task(task):
    skill, tested, seed, max_steps = task
    rallies = {"off": play_rally(seed, (skill, skill), None, None, None, max_steps)}
    for mechanism in MECHANISMS:
        for pos in DEFEND_POSITION_NAMES:
            d1, d2 = (pos, None) if tested == 1 else (None, pos)
            rallies[f"{mechanism}:{pos}"] = play_rally(seed, (skill, skill), d1, d2, mechanism, max_steps)
    return {"skill": skill, "tested": tested, "seed": seed, "rallies": rallies}


def _both_sided_task(task):
    seed, max_steps = task
    rng = np.random.default_rng(seed)
    skills = (str(rng.choice(SKILL_NAMES)), str(rng.choice(SKILL_NAMES)))
    defends = (str(rng.choice(DEFEND_POSITION_NAMES)), str(rng.choice(DEFEND_POSITION_NAMES)))
    rallies = {"off": play_rally(seed, skills, None, None, None, max_steps)}
    for mechanism in MECHANISMS:
        rallies[mechanism] = play_rally(seed, skills, defends[0], defends[1], mechanism, max_steps)
    return {"skills": skills, "defends": defends, "seed": seed, "rallies": rallies}


def tested_win(rally, tested):
    """1/0 for a decided rally from the tested player's view, None if truncated."""
    if rally["winner"] is None:
        return None
    return int(rally["winner"] == ("ego" if tested == 1 else "opp"))


def paired_diffs(units, cond_a, cond_b):
    """Per unit: mean over its decided (a, b) pairs of win_a - win_b, or None."""
    out = []
    for unit in units:
        diffs = []
        for a, b in zip(cond_a(unit), cond_b(unit)):
            wa, wb = tested_win(a, unit["tested"]), tested_win(b, unit["tested"])
            if wa is not None and wb is not None:
                diffs.append(wa - wb)
        out.append(float(np.mean(diffs)) if diffs else None)
    return [d for d in out if d is not None]


def bootstrap_ci(values, rng, n=N_BOOTSTRAP):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    means = rng.choice(values, size=(n, values.size), replace=True).mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _rate_table(units, key):
    if not units:
        return {"win": float("nan"), "decided": 0, "trunc": float("nan"), "steps": float("nan")}
    rallies = [u["rallies"][key] for u in units]
    decided = [tested_win(r, u["tested"]) for r, u in zip(rallies, units)]
    decided = [d for d in decided if d is not None]
    return {
        "win": float(np.mean(decided)) if decided else float("nan"),
        "decided": len(decided),
        "trunc": float(np.mean([r["winner"] is None for r in rallies])),
        "steps": float(np.mean([r["steps"] for r in rallies])),
    }


def summarize_one_sided(units, rng):
    keys = ["off"] + [f"{m}:{p}" for m in MECHANISMS for p in DEFEND_POSITION_NAMES]
    summary = {"conditions": {k: _rate_table(units, k) for k in keys}, "mechanisms": {}}
    for mech in MECHANISMS:
        gate = paired_diffs(units, lambda u, m=mech: [u["rallies"][f"{m}:{p}"] for p in DEFEND_POSITION_NAMES],
                            lambda u: [u["rallies"]["off"]] * len(DEFEND_POSITION_NAMES))
        content = paired_diffs(units, lambda u, m=mech: [u["rallies"][f"{m}:{SKILL_SIDE[u['skill']]}"]],
                               lambda u, m=mech: [u["rallies"][f"{m}:{OPPOSITE[SKILL_SIDE[u['skill']]]}"]])
        per_skill = {
            s: {p: _rate_table([u for u in units if u["skill"] == s], f"{mech}:{p}")["win"]
                for p in DEFEND_POSITION_NAMES}
            for s in ONE_SIDED_SKILLS
        }
        g, c = bootstrap_ci(gate, rng), bootstrap_ci(content, rng)
        summary["mechanisms"][mech] = {
            "gate": g, "gate_pass": bool(g[1] > GATE_FLOOR), "gate_units": len(gate),
            "matched_minus_mismatched": c, "content_detected": bool(c[1] > 0 or c[2] < 0),
            "content_units": len(content), "win_by_skill_and_position": per_skill,
        }
    if "reactive" in MECHANISMS:
        same = [u["rallies"]["reactive:center"] == u["rallies"]["off"] for u in units]
        summary["sanity_reactive_center_equals_off"] = float(np.mean(same))
    return summary


def summarize_both_sided(units):
    out = {}
    for key in ("off",) + tuple(MECHANISMS):
        rallies = [u["rallies"][key] for u in units]
        out[key] = {
            "trunc": float(np.mean([r["winner"] is None for r in rallies])),
            "steps_mean": float(np.mean([r["steps"] for r in rallies])),
            "steps_median": float(np.median([r["steps"] for r in rallies])),
            "returns_per_rally": float(np.mean([sum(r["returns"]) for r in rallies])),
        }
    return out


def _fmt(ci):
    return f"{ci[0]:+.3f} [{ci[1]:+.3f}, {ci[2]:+.3f}]"


def print_one_sided(summary):
    print(f"{'condition':<22}{'win':>7}{'decided':>9}{'trunc':>8}{'steps':>8}")
    for key, row in summary["conditions"].items():
        print(f"{key:<22}{row['win']:>7.3f}{row['decided']:>9}{row['trunc']:>8.3f}{row['steps']:>8.0f}")
    for mech, s in summary["mechanisms"].items():
        print(f"\n[{mech}] gate (mechanism - off): {_fmt(s['gate'])}  n={s['gate_units']}  "
              f"{'PASS' if s['gate_pass'] else 'FAIL'}")
        print(f"[{mech}] matched - mismatched:     {_fmt(s['matched_minus_mismatched'])}  "
              f"n={s['content_units']}  {'content detected' if s['content_detected'] else 'no content detected'}")
        for skill, row in s["win_by_skill_and_position"].items():
            cells = "  ".join(f"{p}={w:.2f}" for p, w in row.items())
            print(f"    attack={skill:<12}{cells}")
    if "sanity_reactive_center_equals_off" in summary:
        print(f"\nsanity: reactive:center reproduces off in {summary['sanity_reactive_center_equals_off']:.0%} of seeds")


def main():
    global MECHANISMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("one-sided", "both-sided"), default="one-sided")
    ap.add_argument("--pairs", type=int, default=40, help="seeds per (skill, tested player) or both-sided seeds")
    ap.add_argument("--base-seed", type=int, default=50_000)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--output", default=None)
    ap.add_argument("--ppo", default=PPO_MODEL_PATH, help="low-level controller checkpoint")
    ap.add_argument("--prep-side-scale", type=float, default=1.0,
                    help="scale of the left/right preparation poses' lateral offset")
    ap.add_argument("--mechanisms", nargs="+", choices=DEFEND_MECHANISMS, default=list(MECHANISMS))
    ap.add_argument("--skill-profile", default="aggressive")
    args = ap.parse_args()
    MECHANISMS = tuple(args.mechanisms)

    if args.mode == "one-sided":
        tasks, fn, seed = [], _one_sided_task, args.base_seed
        for skill in ONE_SIDED_SKILLS:
            for tested in (1, 2):
                for _ in range(args.pairs):
                    tasks.append((skill, tested, seed, args.max_steps))
                    seed += 1
    else:
        tasks = [(args.base_seed + i, args.max_steps) for i in range(args.pairs)]
        fn = _both_sided_task
    output = args.output or f"logs/defend_mechanism_comparison_{args.mode}.json"
    per_task = 1 + len(MECHANISMS) * (len(DEFEND_POSITION_NAMES) if args.mode == "one-sided" else 1)
    print(f"{args.mode}: {len(tasks)} seeds x {per_task} rallies on {args.workers} workers", flush=True)

    start, units = time.time(), []
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker,
                                      initargs=(args.ppo, args.prep_side_scale, MECHANISMS,
                                                args.skill_profile)) as pool:
        for i, unit in enumerate(pool.imap_unordered(fn, tasks), 1):
            units.append(unit)
            if i % max(1, len(tasks) // 10) == 0:
                print(f"  {i}/{len(tasks)} seeds done ({time.time() - start:.0f}s)", flush=True)

    rng = np.random.default_rng(0)
    summary = summarize_one_sided(units, rng) if args.mode == "one-sided" else summarize_both_sided(units)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump({"args": vars(args), "summary": summary, "units": units}, f)
    if args.mode == "one-sided":
        print_one_sided(summary)
    else:
        for key, row in summary.items():
            print(f"{key:<12} trunc={row['trunc']:.3f}  steps mean={row['steps_mean']:.0f} "
                  f"median={row['steps_median']:.0f}  returns/rally={row['returns_per_rally']:.2f}")
    print(f"\nSaved {output}  ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
