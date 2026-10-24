#!/usr/bin/env python3
"""Batch-evaluate V3 high-level policies against fixed and policy opponents.

This orchestration script preserves ``eval_q_policy.py`` as the detailed
single-matchup evaluator and reuses its per-rally runner and policy logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nash_skills.skills import SKILL_NAMES, SKILL_PROFILE_NAMES  # noqa: E402
from v3.constants import DEFENSE_POSITIONS  # noqa: E402
from v3.eval_fixed_preposition import HISTORY  # noqa: E402
from v3.eval_q_policy import (  # noqa: E402
    POLICY_NAMES,
    _policy_label,
    run_episode,
    seed_everything,
)
from v3.potential_policy import new_action_counts  # noqa: E402
from v3.q_policy import load_q_model  # noqa: E402
from v3.potential_policy import load_potential_model  # noqa: E402


DEFAULT_POLICIES = (
    "nash_q",
    "br_q",
    "greedy_q",
    "potential_ibr",
    "potential_br",
    "potential_adaptive_rally",
    "potential_adaptive_match",
    "potential_minimax",
    "random",
    "fixed",
)
OTHER_OPPONENT_CHOICES = tuple(name for name in POLICY_NAMES if name != "fixed")
DISPLAY_NAMES = {
    "nash_q": "Nash-Q",
    "br_q": "BR-Q",
    "greedy_q": "IBR",
    "potential_ibr": "Potential IBR",
    "potential_br": "Potential BR",
    "potential_adaptive_rally": "Potential Adaptive Rally",
    "potential_adaptive_match": "Potential Adaptive Match",
    "potential_minimax": "Potential Minimax",
    "random": "Random",
    "fixed": "Fixed",
}


def _episode_seed(seed: int, episode_index: int) -> int:
    sequence = np.random.SeedSequence([int(seed), int(episode_index)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _make_policy_config(
    tested_policy: str,
    opponent_policy: str,
    tested_player: int,
    opponent_type: str,
    fixed_attack: str,
    fixed_defense: str,
    tested_fixed_attack: str,
    tested_fixed_defense: str,
) -> tuple[dict[int, dict[str, str]], dict[int, str], dict[int, str]]:
    opponent_player = 2 if tested_player == 1 else 1
    policies = {
        tested_player: {"attack": tested_policy, "defense": tested_policy},
        opponent_player: {"attack": opponent_policy, "defense": opponent_policy},
    }
    fixed_shots = {1: tested_fixed_attack, 2: tested_fixed_attack}
    fixed_defenses = {1: tested_fixed_defense, 2: tested_fixed_defense}
    if opponent_type == "static" or opponent_policy == "fixed":
        fixed_shots[opponent_player] = fixed_attack
        fixed_defenses[opponent_player] = fixed_defense
    return policies, fixed_shots, fixed_defenses


def _frequency_counts(
    decisions: list[dict[str, Any]],
    action_field: str,
    player_field: str,
    player_id: int,
    names: tuple[str, ...] | list[str],
) -> dict[str, int]:
    counts = Counter(
        str(row[action_field])
        for row in decisions
        if int(row[player_field]) == player_id and row.get(action_field) is not None
    )
    return {name: int(counts.get(name, 0)) for name in names}


def _defined_mean(decisions: list[dict[str, Any]], key: str) -> float | str:
    values = [float(row[key]) for row in decisions if row.get(key) is not None]
    return float(np.mean(values)) if values else ""


def _raw_record(
    *,
    tested_policy: str,
    opponent_type: str,
    opponent_policy: str,
    fixed_attack: str,
    fixed_defense: str,
    seed: int,
    tested_player: int,
    episodes: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    wins = sum(int(row.get("winner_player") == tested_player) for row in episodes)
    losses = sum(
        int(row.get("winner_player") in (1, 2) and row.get("winner_player") != tested_player)
        for row in episodes
    )
    unresolved = len(episodes) - wins - losses
    attack_freq = {
        f"p{player}_attack_skill_frequencies_json": json.dumps(
            _frequency_counts(decisions, "attack_skill", "attacker_id", player, SKILL_NAMES),
            separators=(",", ":"),
        )
        for player in (1, 2)
    }
    defense_freq = {
        f"p{player}_defense_position_frequencies_json": json.dumps(
            _frequency_counts(decisions, "defense_preposition", "defender_id", player, DEFENSE_POSITIONS),
            separators=(",", ":"),
        )
        for player in (1, 2)
    }
    tested_attack = _frequency_counts(decisions, "attack_skill", "attacker_id", tested_player, SKILL_NAMES)
    tested_defense = _frequency_counts(decisions, "defense_preposition", "defender_id", tested_player, DEFENSE_POSITIONS)
    return {
        "tested_policy": tested_policy,
        "opponent_type": opponent_type,
        "opponent_policy": opponent_policy,
        "fixed_attack_skill": fixed_attack if opponent_type == "static" else "",
        "fixed_defense_position": fixed_defense if opponent_type == "static" else "",
        "seed": seed,
        "side_assignment": f"tested_p{tested_player}",
        "tested_player_id": tested_player,
        "episode_count": len(episodes),
        "wins": wins,
        "losses": losses,
        "unresolved_or_truncated": unresolved,
        "win_rate": wins / len(episodes) if episodes else "",
        "average_rally_length_steps": float(np.mean([row["steps"] for row in episodes])) if episodes else "",
        "total_decisions": len(decisions),
        "contact_rate": _defined_mean(decisions, "contact_observed"),
        "successful_return_rate": _defined_mean(decisions, "successful_return_observed"),
        "miss_rate": _defined_mean(decisions, "attacker_miss"),
        "out_of_bounds_rate": _defined_mean(decisions, "attacker_out_of_bounds"),
        "tested_attack_skill_frequencies_json": json.dumps(tested_attack, separators=(",", ":")),
        "tested_defense_position_frequencies_json": json.dumps(tested_defense, separators=(",", ":")),
        **attack_freq,
        **defense_freq,
    }


def _aggregate_win_rate(rows: list[dict[str, Any]]) -> float | str:
    total_episodes = sum(int(row["episode_count"]) for row in rows)
    if not total_episodes:
        return ""
    return sum(int(row["wins"]) for row in rows) / total_episodes


def _build_summary(
    raw_rows: list[dict[str, Any]],
    tested_policies: list[str],
    *,
    other_opponent: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    table: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    for policy in tested_policies:
        policy_rows = [row for row in raw_rows if row["tested_policy"] == policy]
        static_rows = [row for row in policy_rows if row["opponent_type"] == "static"]
        matchup_rates = []
        for attack in SKILL_NAMES:
            for defense in DEFENSE_POSITIONS:
                matchup_rows = [
                    row for row in static_rows
                    if row["fixed_attack_skill"] == attack and row["fixed_defense_position"] == defense
                ]
                matchup_rates.append({
                    "fixed_attack_skill": attack,
                    "fixed_defense_position": defense,
                    "win_rate": _aggregate_win_rate(matchup_rows),
                    "wins": sum(int(row["wins"]) for row in matchup_rows),
                    "episodes": sum(int(row["episode_count"]) for row in matchup_rows),
                })
        static_values = [float(row["win_rate"]) for row in matchup_rates if row["win_rate"] != ""]
        random_rate = _aggregate_win_rate([row for row in policy_rows if row["opponent_type"] == "random"])
        ibr_rate = _aggregate_win_rate([row for row in policy_rows if row["opponent_type"] == "ibr"])
        other_rate = _aggregate_win_rate([row for row in policy_rows if row["opponent_type"] == "other_proposed"])
        summary_row = {
            "Method": DISPLAY_NAMES.get(policy, policy),
            "Static mean": float(np.mean(static_values)) if static_values else "",
            "Static minimum": min(static_values) if static_values else "",
            "Random": random_rate,
            "IBR": ibr_rate,
            "Other proposed": other_rate,
        }
        table.append(summary_row)
        detail[policy] = {
            "display_name": summary_row["Method"],
            "static_matchup_win_rates": matchup_rates,
            "static_mean": summary_row["Static mean"],
            "static_minimum": summary_row["Static minimum"],
            "random_win_rate": random_rate,
            "ibr_win_rate": ibr_rate,
            "other_opponent": other_opponent,
            "other_opponent_win_rate": other_rate,
        }
    return table, detail


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format_rate(value: Any) -> str:
    return "—" if value == "" else f"{100 * float(value):.1f}%"


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = ["Method", "Static mean", "Static minimum", "Random", "IBR", "Other proposed"]
    formatted = [[row[headers[0]]] + [_format_rate(row[key]) for key in headers[1:]] for row in rows]
    widths = [max(len(headers[i]), *(len(row[i]) for row in formatted)) for i in range(len(headers))]
    print(" | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in formatted:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/v3/q_preposition_state74_onehot.pth")
    parser.add_argument("--potential-model", default="models/v3/potential_state74.pth")
    parser.add_argument("--ppo", default="logs/best_model_tracker1/best_model")
    parser.add_argument("--policies", nargs="+", choices=POLICY_NAMES, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--both-sides", action="store_true")
    parser.add_argument("--other-opponent", choices=OTHER_OPPONENT_CHOICES, default="potential_adaptive_match")
    parser.add_argument("--tested-fixed-shot", choices=SKILL_NAMES, default="center_safe")
    parser.add_argument("--tested-fixed-defense", choices=DEFENSE_POSITIONS, default="center")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--greedy-iterations", type=int, default=4)
    parser.add_argument("--reset-mode", choices=["clean", "ready", "carryover"], default="ready")
    parser.add_argument("--skill-profile", choices=SKILL_PROFILE_NAMES, default="aggressive")
    parser.add_argument("--gantry-speed-scale", type=float, default=1.0)
    parser.add_argument("--planner", choices=["legacy", "legacy_legal", "optimized"], default="legacy")
    parser.add_argument("--output-dir", default="skill_eval/v3/batch_policy_eval")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes < 1 or args.steps < 1:
        raise ValueError("--episodes and --steps must be positive")
    if not args.seeds or any(seed < 0 for seed in args.seeds):
        raise ValueError("--seeds must contain one or more nonnegative integers")
    tested_policies = args.policies if args.policies is not None else list(DEFAULT_POLICIES)
    if len(set(tested_policies)) != len(tested_policies):
        raise ValueError("--policies must not contain duplicates")

    from stable_baselines3 import PPO
    from v3.env_wrapper import V3PrePositionEnv

    q_model = load_q_model(args.model, device=args.device)
    potential_model = load_potential_model(args.potential_model, device=args.device)
    ppo = PPO.load(args.ppo)
    env = V3PrePositionEnv(
        proc_id=1,
        history=HISTORY,
        reset_mode=args.reset_mode,
        skill_profile=args.skill_profile,
        gantry_speed_scale=args.gantry_speed_scale,
        planner=args.planner,
    )

    assignments = [1, 2] if args.both_sides else [1]
    static_combinations = [(attack, defense) for attack in SKILL_NAMES for defense in DEFENSE_POSITIONS]
    opponent_specs: list[tuple[str, str, str, str]] = [
        ("static", "fixed", attack, defense) for attack, defense in static_combinations
    ]
    opponent_specs.extend([
        ("random", "random", "", ""),
        ("ibr", "greedy_q", "", ""),
        ("other_proposed", args.other_opponent, "", ""),
    ])
    total_runs = len(tested_policies) * len(opponent_specs) * len(args.seeds) * len(assignments)
    run_number = 0
    raw_rows: list[dict[str, Any]] = []

    try:
        for tested_policy in tested_policies:
            for opponent_type, opponent_policy, fixed_attack, fixed_defense in opponent_specs:
                for seed in args.seeds:
                    for tested_player in assignments:
                        run_number += 1
                        side_label = f"tested_p{tested_player}"
                        print(
                            f"[{run_number}/{total_runs}] policy={tested_policy} "
                            f"opponent={opponent_type}:{opponent_policy} "
                            f"fixed=({fixed_attack or '-'}, {fixed_defense or '-'}) "
                            f"seed={seed} side={side_label} episodes={args.episodes}",
                            flush=True,
                        )
                        policies, fixed_shots, fixed_defenses = _make_policy_config(
                            tested_policy,
                            opponent_policy,
                            tested_player,
                            opponent_type,
                            fixed_attack,
                            fixed_defense,
                            args.tested_fixed_shot,
                            args.tested_fixed_defense,
                        )
                        match_action_counts = {1: new_action_counts(), 2: new_action_counts()}
                        run_episodes: list[dict[str, Any]] = []
                        run_decisions: list[dict[str, Any]] = []
                        for episode_index in range(args.episodes):
                            current_seed = _episode_seed(seed, episode_index)
                            seed_everything(current_seed)
                            rng = np.random.default_rng(current_seed)
                            episode, decisions = run_episode(
                                episode_index=episode_index,
                                env=env,
                                ppo=ppo,
                                model=q_model,
                                potential_model=potential_model,
                                policies=policies,
                                match_action_counts=match_action_counts,
                                fixed_shots=fixed_shots,
                                fixed_defenses=fixed_defenses,
                                rng=rng,
                                device=args.device,
                                max_steps=args.steps,
                                greedy_iterations=args.greedy_iterations,
                                render=False,
                                debug_decisions=False,
                                episode_seed=current_seed,
                            )
                            run_episodes.append(episode)
                            run_decisions.extend(decisions)
                        raw_rows.append(_raw_record(
                            tested_policy=tested_policy,
                            opponent_type=opponent_type,
                            opponent_policy=opponent_policy,
                            fixed_attack=fixed_attack,
                            fixed_defense=fixed_defense,
                            seed=seed,
                            tested_player=tested_player,
                            episodes=run_episodes,
                            decisions=run_decisions,
                        ))
    finally:
        env.close()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows, summary_detail = _build_summary(raw_rows, tested_policies, other_opponent=args.other_opponent)
    _write_csv(output_dir / "batch_policy_eval_raw.csv", raw_rows)
    _write_csv(output_dir / "batch_policy_eval_summary.csv", summary_rows)
    summary_payload = {
        "metadata": {
            "policies": tested_policies,
            "seeds": args.seeds,
            "episodes_per_seed_matchup_side": args.episodes,
            "both_sides": args.both_sides,
            "other_opponent": args.other_opponent,
            "static_opponent_grid": {"attack_skills": list(SKILL_NAMES), "defense_positions": list(DEFENSE_POSITIONS)},
            "win_rate_denominator": "all episodes; unresolved/truncated episodes count as non-wins",
            "static_aggregation": "pool wins and episodes across seeds and side assignments within each fixed matchup, then mean/min across 15 matchup rates",
            "seed_handling": "per-episode SeedSequence derived from input seed and episode index; seeds Python, NumPy, Torch, local policy RNG, and env.reset(seed=...) before reset",
            "match_history_scope": "reset per policy/opponent/fixed matchup/seed/side-assignment run; retained across its episodes",
        },
        "summary_table": summary_rows,
        "policy_details": summary_detail,
        "raw_record_count": len(raw_rows),
    }
    with (output_dir / "batch_policy_eval_summary.json").open("w") as handle:
        json.dump(summary_payload, handle, indent=2)

    _print_table(summary_rows)
    print(f"Saved batch evaluation outputs under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
