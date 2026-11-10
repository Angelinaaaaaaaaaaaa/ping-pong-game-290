"""
Live-MuJoCo verification for the attacker/defender defend-position feature.

Answers two questions in one small experiment, per the agreed plan:

  Step 1 (physics): does the defender's gantry actually move to the
  requested position, and does player 2 need coordinate mirroring or not?
  Logged directly: requested defend position, actual final gantry qpos,
  ball direction, who was receiving.

  Step 2 (does it matter): fixing P1's attack skill at something with a
  clear lateral bias ("left", which aims deep left -- see nash_skills.skills),
  sweep P2's defend position across all three choices and compare outcomes.
  If "matching" the attack's side wins more than "mismatched" defend
  choices, defend-positioning is doing real work. If there's no difference,
  check the qpos log before concluding it doesn't matter -- it could
  instead mean the steering isn't taking effect.

Baseline control (added after the first real-hitting run came back with
opp_wr near 0/10 in every combination, including supposedly "matched"
ones): run the same attack skills with NO defend-steering override at all
(opp_defend=None -- player 2 entirely under its own PPO policy) to tell
apart two hypotheses: (a) the defend-steering override stays active
through player 2's final approach and fights the PPO controller's own
positioning for the actual return stroke, suppressing returns regardless
of defend choice; vs (b) a pre-existing first-attacker baseline asymmetry
unrelated to defend-steering at all. Result: (a) -- baseline opp never
lost outright (ego_wins=0/10 both attack skills) while the always-on
override gave ego_wins=9-10/10 in every combo.

Override-timing fix (this version): defend_position is now a
PRE-POSITIONING action only (nash_skills.defend_engagement). Steering
stops the moment the target is reached and gantry control releases back
to the PPO controller for the rest of that defending phase, rather than
holding the override through the return swing itself. This script now
also records, per rally: whether steering ever engaged at all, whether
the target was reached (and therefore released) before the rally ended,
rally length, and a "decisive" winner flag (resolved by an explicit
env signal / racket-bound reconstruction / ball velocity, as opposed to
the looser x-position fallback) -- see nash_skills.winner_inference.

Deliberately small (N rallies per combination) -- this is a physics sanity
check, not a statistically tight result. Not run as part of any automated
suite; invoke directly once MuJoCo is available (server or local macOS with
MUJOCO_GL=cgl/osmesa, never egl on macOS).

Run:
    MUJOCO_GL=egl PYTHONPATH=. venv/bin/python scripts/test_defend_physics.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import io
import statistics
from contextlib import redirect_stdout

import numpy as np
from stable_baselines3 import PPO

from nash_skills.env_wrapper import SkillEnv
from nash_skills.defend_positions import DEFEND_POSITION_NAMES, world_defend_qpos
from nash_skills.winner_inference import infer_terminal_winner
from nash_skills.v2.collect_data import _build_ppo_obs

PPO_MODEL_PATH = "logs/best_model_tracker1/best_model"
N_RALLIES_PER_COMBO = 10
ATTACK_SKILLS = ["left", "right"]
MAX_STEPS = 1000


def _swallow_step(env, action):
    buf = io.StringIO()
    with redirect_stdout(buf):
        return env.step(action)


def run_one_rally(env, ppo, ego_attack: str, opp_defend, rng: np.random.Generator):
    """
    Ego (player 1) always attacks with `ego_attack`; opp (player 2) always
    defends toward `opp_defend` whenever the ball is heading their way.

    `opp_defend=None` is the no-override baseline: player 2 plays entirely
    under its own PPO policy, with no defend-position steering applied at
    all (env.set_defend_positions(None, None) is a documented no-op in
    SkillEnv.step). There is no target to track distance to in that case.

    Returns a dict with the outcome and the physics trace needed to verify
    correctness (did P2 actually get there), whether defend position
    affected the result, and whether the override-timing fix actually
    released control back to the PPO before the rally ended.
    """
    env.set_skills(ego_attack, ego_attack)  # opp's "attack" choice is unused
                                             # whenever they're defending, but
                                             # set_skills needs a value; reuse
                                             # ego_attack as a harmless default
    env.set_defend_positions(None, opp_defend)  # only P2 defends in this test;
                                                 # opp_defend=None disables the
                                                 # override entirely (baseline)
    obs, info = env.reset()

    steps = 0
    done = False
    min_dist_to_target = float("inf")
    if opp_defend is not None:
        target_x, target_y = world_defend_qpos(2, opp_defend)
    else:
        target_x, target_y = None, None

    while True:
        ppo1 = _build_ppo_obs(obs, info, player=1)
        ppo2 = _build_ppo_obs(obs, info, player=2)
        a1, _ = ppo.predict(ppo1, deterministic=True)
        a2, _ = ppo.predict(ppo2, deterministic=True)
        action = np.zeros(18, dtype=np.float32)
        action[:9] = a1[:9]
        action[9:] = a2[:9]
        # Defend steering overrides the gantry components (indices 0,1 or
        # 9,10) of whichever player is currently defending, ONLY until
        # their target is reached (nash_skills.defend_engagement); the
        # rest of each player's PPO-driven action (racket/arm, and the
        # gantry itself once released) passes through unmodified. No-op
        # entirely when opp_defend is None (the baseline).
        obs, _r, done, _t, info = _swallow_step(env, action)
        steps += 1

        if target_x is not None:
            qpos2 = (float(env._env.data.qpos[9]), float(env._env.data.qpos[10]))
            dist = ((qpos2[0] - target_x) ** 2 + (qpos2[1] - target_y) ** 2) ** 0.5
            min_dist_to_target = min(min_dist_to_target, dist)

        if done or steps >= MAX_STEPS:
            break

    final_qpos2 = (float(env._env.data.qpos[9]), float(env._env.data.qpos[10]))
    if done:
        winner = infer_terminal_winner(obs, info, fallback="position")
        # "Decisive": resolved by an explicit env signal, racket-bound
        # reconstruction, or ball velocity -- NOT the looser x-position
        # fallback. fallback=None returns None unless one of the
        # stronger tiers resolved it; if it did, that's the same value
        # `winner` holds (tiers are checked in the same priority order
        # regardless of the fallback param).
        decisive = infer_terminal_winner(obs, info, fallback=None) is not None
    else:
        winner = None
        decisive = False

    # Diagnostics for the override-timing fix: did steering ever activate
    # for player 2 this rally, and had it been released (target reached)
    # by the time the rally ended? Both are False/False for the baseline
    # (opp_defend=None), since should_steer is never even called with a
    # real defend_name there.
    ever_engaged = env._engagement.ever_engaged(2)
    target_reached = env._engagement.is_done(2)

    return {
        "ego_attack": ego_attack,
        "opp_defend": opp_defend,
        "requested_target": (target_x, target_y) if target_x is not None else None,
        "final_qpos2": final_qpos2,
        "min_dist_to_target": min_dist_to_target if target_x is not None else None,
        "steps": steps,
        "done": done,
        "winner": winner,          # "ego", "opp", or None (truncated)
        "decisive": decisive,
        "ever_engaged": ever_engaged,
        "target_reached": target_reached,
    }


def _summarize(rows):
    n = len(rows)
    opp_wins = sum(1 for r in rows if r["winner"] == "opp")
    ego_wins = sum(1 for r in rows if r["winner"] == "ego")
    trunc = sum(1 for r in rows if r["winner"] is None)
    opp_decisive = sum(1 for r in rows if r["winner"] == "opp" and r["decisive"])
    lengths = [r["steps"] for r in rows]
    return {
        "n": n,
        "ego_wins": ego_wins,
        "opp_wins": opp_wins,
        "trunc": trunc,
        "opp_decisive_wr": opp_decisive / n,
        "avg_len": statistics.mean(lengths),
        "median_len": statistics.median(lengths),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rallies", type=int, default=N_RALLIES_PER_COMBO)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    env = SkillEnv(proc_id=1, history=4)
    print(f"Loading PPO from {PPO_MODEL_PATH} ...")
    ppo = PPO.load(PPO_MODEL_PATH, device="cpu")

    print(f"Baseline control (no defend-steering override at all): "
          f"{len(ATTACK_SKILLS)} attack skills, {args.rallies} rallies each\n")

    baseline_results = []
    for attack in ATTACK_SKILLS:
        rows = [run_one_rally(env, ppo, attack, None, rng) for _ in range(args.rallies)]
        baseline_results.extend(rows)

        s = _summarize(rows)
        print(f"attack={attack:<6} defend=NONE(baseline)  "
              f"ego_wins={s['ego_wins']}/{s['n']}  opp_wins={s['opp_wins']}/{s['n']}  "
              f"trunc={s['trunc']}/{s['n']}  opp_decisive_wr={s['opp_decisive_wr']:.2f}  "
              f"rally_len avg={s['avg_len']:.1f} median={s['median_len']:.1f}")

    print(f"\nTesting {len(ATTACK_SKILLS)} attack skills x {len(DEFEND_POSITION_NAMES)} "
          f"defend positions, {args.rallies} rallies each (override releases on target-reach)\n")

    results = []
    for attack in ATTACK_SKILLS:
        for defend in DEFEND_POSITION_NAMES:
            rows = [run_one_rally(env, ppo, attack, defend, rng) for _ in range(args.rallies)]
            results.extend(rows)

            s = _summarize(rows)
            avg_min_dist = float(np.mean([r["min_dist_to_target"] for r in rows]))
            engaged = sum(1 for r in rows if r["ever_engaged"])
            reached = sum(1 for r in rows if r["target_reached"])
            print(f"attack={attack:<6} defend={defend:<7} target={world_defend_qpos(2, defend)}")
            print(f"    outcomes: ego_wins={s['ego_wins']}/{s['n']}  opp_wins={s['opp_wins']}/{s['n']}  "
                  f"trunc={s['trunc']}/{s['n']}  opp_decisive_wr={s['opp_decisive_wr']:.2f}  "
                  f"rally_len avg={s['avg_len']:.1f} median={s['median_len']:.1f}")
            print(f"    engagement: ever_engaged={engaged}/{s['n']}  target_reached(released)={reached}/{s['n']}  "
                  f"avg_min_dist_to_target={avg_min_dist:.4f}")

    env.close()

    print("\n--- compact comparison table ---")
    print(f"{'attack':<7}{'defend':<9}{'ego_wr':<8}{'opp_wr':<8}{'opp_dec_wr':<11}{'trunc':<7}"
          f"{'avg_len':<9}{'engaged':<9}{'released':<9}")
    for attack in ATTACK_SKILLS:
        brows = [r for r in baseline_results if r["ego_attack"] == attack]
        s = _summarize(brows)
        print(f"{attack:<7}{'NONE':<9}{s['ego_wins']/s['n']:<8.2f}{s['opp_wins']/s['n']:<8.2f}"
              f"{s['opp_decisive_wr']:<11.2f}{s['trunc']/s['n']:<7.2f}{s['avg_len']:<9.1f}"
              f"{'n/a':<9}{'n/a':<9}")
    for attack in ATTACK_SKILLS:
        for defend in DEFEND_POSITION_NAMES:
            drows = [r for r in results if r["ego_attack"] == attack and r["opp_defend"] == defend]
            s = _summarize(drows)
            engaged = sum(1 for r in drows if r["ever_engaged"]) / s["n"]
            reached = sum(1 for r in drows if r["target_reached"]) / s["n"]
            print(f"{attack:<7}{defend:<9}{s['ego_wins']/s['n']:<8.2f}{s['opp_wins']/s['n']:<8.2f}"
                  f"{s['opp_decisive_wr']:<11.2f}{s['trunc']/s['n']:<7.2f}{s['avg_len']:<9.1f}"
                  f"{engaged:<9.2f}{reached:<9.2f}")

    print("\n--- raw per-rally trace, baseline (first 3 per attack skill) ---")
    seen = {}
    for r in baseline_results:
        key = (r["ego_attack"], r["opp_defend"])
        seen.setdefault(key, 0)
        if seen[key] < 3:
            print(f"  attack={r['ego_attack']} defend=NONE "
                  f"final_qpos2={r['final_qpos2']} winner={r['winner']} decisive={r['decisive']} "
                  f"steps={r['steps']}")
            seen[key] += 1

    print("\n--- raw per-rally trace (first 3 per combo) for manual mirroring check ---")
    seen = {}
    for r in results:
        key = (r["ego_attack"], r["opp_defend"])
        seen.setdefault(key, 0)
        if seen[key] < 3:
            print(f"  attack={r['ego_attack']} defend={r['opp_defend']} "
                  f"requested={r['requested_target']} final_qpos2={r['final_qpos2']} "
                  f"min_dist={r['min_dist_to_target']:.4f} winner={r['winner']} decisive={r['decisive']} "
                  f"ever_engaged={r['ever_engaged']} target_reached={r['target_reached']} steps={r['steps']}")
            seen[key] += 1


if __name__ == "__main__":
    main()
