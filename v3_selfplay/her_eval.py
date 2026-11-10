"""
Self-play policies scored by Kexin's own evaluator, so they follow exactly her protocol. Her
v3/eval_q_policy.run_episode asks _distribution_for_role(policy, role, ...) for every policy it does not handle
itself, without the state. Inside `installed`, that call answers for registered self-play names from the state of
the state_for_attacker call run_episode makes just before it (which also says who attacks, so which physical
player's network acts). Nothing in her files changes; her module globals are restored on exit.

    # her Table 1 (v3/eval_policy_fixed_grid.py) with self-play as the tested policy
    PYTHONPATH=. python -m v3_selfplay.her_eval grid --register sp=runs/sp_s0.pth -- --policies sp \\
        --model models/v3/q_preposition_onehot_50k_aggressive.pth \\
        --potential-model models/v3/potential_50k_aggressive.pth --seeds 0 --episodes 100 --both-sides \\
        --output-dir out/grid_sp_s0
    # head-to-head against her rules, same seeding and scoring
    PYTHONPATH=. python -m v3_selfplay.her_eval h2h --register sp=runs/sp_s0.pth --tested sp \\
        --opponents potential_ibr potential_br potential_adaptive_match potential_minimax nash_q br_q \\
        --model ... --potential-model ... --seeds 0 --episodes 100 --both-sides --output-json out/h2h_sp_s0.json
"""
import argparse
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from v3_selfplay.game import HER_MAX_STEPS, make_her_env
from v3_selfplay.policy import ATTACKER, action_distribution, load_policies, policy_input

__all__ = ["Entry", "parse_entry", "installed", "head_to_head", "summarize_head_to_head", "main"]

ARGMAX_SUFFIX = ":argmax"


@dataclass(frozen=True)
class Entry:
    policies: dict          # {1: psi_1, 2: psi_2}
    argmax: bool = False


def parse_entry(spec):
    """'name=checkpoint' or 'name=checkpoint:argmax' -> (name, Entry)."""
    if "=" not in spec:
        raise ValueError(f"Expected name=checkpoint[{ARGMAX_SUFFIX}], got {spec!r}")
    name, path = spec.split("=", 1)
    argmax = path.endswith(ARGMAX_SUFFIX)
    path = path[:-len(ARGMAX_SUFFIX)] if argmax else path
    policies, _meta = load_policies(path)
    return name, Entry(policies=policies, argmax=argmax)


class _Hook:
    def __init__(self, entries, state_fn, distribution_fn):
        self.entries, self._state_fn, self._distribution_fn = dict(entries), state_fn, distribution_fn
        self.last = None

    def state_for_attacker(self, obs, info, attacker):
        state = self._state_fn(obs, info, attacker)
        self.last = (np.array(state, dtype=np.float32), int(attacker))
        return state

    def distribution_for_role(self, policy, role, matrix, solution, **kwargs):
        entry = self.entries.get(policy)
        if entry is None:
            return self._distribution_fn(policy, role, matrix, solution, **kwargs)
        if self.last is None:
            raise RuntimeError("asked for a self-play action before any state_for_attacker call")
        state, attacker = self.last
        player = attacker if role == ATTACKER else 3 - attacker
        return action_distribution(entry.policies[player], policy_input(state, role), role, argmax=entry.argmax)


@contextmanager
def installed(entries):
    """Within the block her evaluator accepts the names in `entries` ({name: Entry}) as policies."""
    import v3.eval_policy_fixed_grid as grid
    import v3.eval_q_policy as her
    saved = (her.state_for_attacker, her._distribution_for_role, her.POLICY_NAMES, grid.POLICY_NAMES)
    hook = _Hook(entries, her.state_for_attacker, her._distribution_for_role)
    names = tuple(n for n in entries if n not in saved[2])
    her.state_for_attacker, her._distribution_for_role = hook.state_for_attacker, hook.distribution_for_role
    her.POLICY_NAMES, grid.POLICY_NAMES = saved[2] + names, saved[3] + names
    try:
        yield hook
    finally:
        her.state_for_attacker, her._distribution_for_role, her.POLICY_NAMES, grid.POLICY_NAMES = saved


def head_to_head(*, tested, opponents, seeds, episodes, both_sides, env, ppo, q_model, potential_model, steps,
                 device="cpu", greedy_iterations=4):
    """`tested` against each opponent policy, seeded, run and recorded like v3/eval_policy_fixed_grid.py does its
    "other proposed" column: one run per (opponent, seed, side) with the match history reset per run."""
    import v3.eval_policy_fixed_grid as grid
    import v3.eval_q_policy as her
    from v3.potential_policy import new_action_counts
    rows = []
    for opponent in opponents:
        for seed in seeds:
            for tested_player in ([1, 2] if both_sides else [1]):
                policies, fixed_shots, fixed_defenses = grid._make_policy_config(
                    tested, opponent, tested_player, "other_proposed", "", "", "center_safe", "center")
                match_counts = {1: new_action_counts(), 2: new_action_counts()}
                run_episodes, run_decisions = [], []
                for k in range(episodes):
                    episode_seed = grid._episode_seed(seed, k)
                    her.seed_everything(episode_seed)
                    episode, decisions = her.run_episode(
                        episode_index=k, env=env, ppo=ppo, model=q_model, potential_model=potential_model,
                        policies=policies, match_action_counts=match_counts, fixed_shots=fixed_shots,
                        fixed_defenses=fixed_defenses, rng=np.random.default_rng(episode_seed), device=device,
                        max_steps=steps, greedy_iterations=greedy_iterations, render=False, debug_decisions=False,
                        episode_seed=episode_seed)
                    run_episodes.append(episode)
                    run_decisions.extend(decisions)
                rows.append(grid._raw_record(tested_policy=tested, opponent_type="head_to_head",
                                             opponent_policy=opponent, fixed_attack="", fixed_defense="", seed=seed,
                                             tested_player=tested_player, episodes=run_episodes,
                                             decisions=run_decisions))
    return rows


def summarize_head_to_head(rows):
    """Per opponent: her win rate (unfinished rallies count as non-wins) and the rate over finished rallies."""
    out = {}
    for opponent in dict.fromkeys(r["opponent_policy"] for r in rows):
        mine = [r for r in rows if r["opponent_policy"] == opponent]
        wins, losses = sum(int(r["wins"]) for r in mine), sum(int(r["losses"]) for r in mine)
        episodes = sum(int(r["episode_count"]) for r in mine)
        out[opponent] = {"episodes": episodes, "wins": wins, "losses": losses,
                         "unresolved": episodes - wins - losses,
                         "win_rate": wins / episodes if episodes else float("nan"),
                         "decided_win_rate": wins / (wins + losses) if wins + losses else float("nan")}
    return out


def _h2h_parser():
    ap = argparse.ArgumentParser(description="Self-play head-to-head against Kexin's V3 policies, her protocol")
    ap.add_argument("--register", action="append", default=[], metavar="NAME=CKPT[:argmax]")
    ap.add_argument("--tested", required=True)
    ap.add_argument("--opponents", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--potential-model", required=True)
    ap.add_argument("--ppo", default="logs/best_model_tracker1/best_model")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--steps", type=int, default=HER_MAX_STEPS)
    ap.add_argument("--both-sides", action="store_true")
    ap.add_argument("--greedy-iterations", type=int, default=4)
    ap.add_argument("--skill-profile", default="aggressive")
    ap.add_argument("--reset-mode", default="ready")
    ap.add_argument("--gantry-speed-scale", type=float, default=1.0)
    ap.add_argument("--planner", default="legacy")
    ap.add_argument("--output-json", required=True)
    return ap


def _run_h2h(argv):
    args = _h2h_parser().parse_args(argv)
    from stable_baselines3 import PPO
    from v3.potential_policy import load_potential_model
    from v3.q_policy import load_q_model
    entries = dict(parse_entry(spec) for spec in args.register)
    env = make_her_env(args.skill_profile, args.reset_mode, args.gantry_speed_scale, args.planner)
    try:
        with installed(entries):
            rows = head_to_head(tested=args.tested, opponents=args.opponents, seeds=args.seeds,
                                episodes=args.episodes, both_sides=args.both_sides, env=env, ppo=PPO.load(args.ppo),
                                q_model=load_q_model(args.model), potential_model=load_potential_model(args.potential_model),
                                steps=args.steps, greedy_iterations=args.greedy_iterations)
    finally:
        env.close()
    summary = summarize_head_to_head(rows)
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"args": vars(args), "summary": summary, "rows": rows}, f, indent=2)
    for opponent, s in summary.items():
        print(f"{args.tested} vs {opponent}: {100 * s['win_rate']:.1f}% of {s['episodes']} rallies "
              f"({s['unresolved']} unfinished; {100 * s['decided_win_rate']:.1f}% of finished)")
    return 0


def _run_grid(argv):
    if "--" not in argv:
        raise SystemExit("usage: grid --register NAME=CKPT[:argmax] ... -- <v3/eval_policy_fixed_grid.py args>")
    split = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="append", default=[], metavar="NAME=CKPT[:argmax]")
    args = ap.parse_args(argv[:split])
    entries = dict(parse_entry(spec) for spec in args.register)
    import v3.eval_policy_fixed_grid as grid
    with installed(entries):
        return grid.main(argv[split + 1:])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"grid": _run_grid, "h2h": _run_h2h}
    if not argv or argv[0] not in commands:
        raise SystemExit(f"usage: python -m v3_selfplay.her_eval {{{','.join(commands)}}} ...")
    return commands[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
