#!/usr/bin/env python3
"""Diagnose data/model asymmetry in the 5-skill v3 Nash pipeline.

This script is intentionally read-only with respect to production code. It
loads a saved aligned rally pickle and optional v3 Q/Phi checkpoints, then
prints dataset tables, example timelines, training-target semantics, and
counterfactual model predictions over skill pairs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle as pkl
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_arch import SimpleModel
from nash_skills.skills import N_SKILLS, SKILL_NAMES, skill_from_index, skill_index
from nash_skills.v2.labeling import GAMMA, compute_returns
from nash_skills.v2.state_encoder import STATE_DIM


def skill_value(skill: str) -> float:
    return skill_index(skill) / (N_SKILLS - 1)


def decode_skill(value: float) -> str:
    idx = int(round(float(value) * (N_SKILLS - 1)))
    idx = max(0, min(N_SKILLS - 1, idx))
    return skill_from_index(idx)


def metadata_path_for(rally_path: str) -> Path:
    path = Path(rally_path)
    return path.with_name(f"{path.stem}_metadata.json")


def load_rallies(path: str) -> list[dict[str, Any]]:
    with open(path, "rb") as f:
        return pkl.load(f)


def load_metadata(path: str) -> dict[str, Any] | None:
    meta_path = metadata_path_for(path)
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def rate(wins: int, total: int) -> float:
    return wins / total if total else float("nan")


def fmt_rate(value: float) -> str:
    return "nan" if math.isnan(value) else f"{100 * value:.1f}%"


def print_counter(title: str, counts: Counter[str], total: int) -> None:
    print(f"\n{title}")
    for name in SKILL_NAMES:
        count = counts[name]
        print(f"  {name:12s} {count:7d}  {fmt_rate(rate(count, total))}")


def print_pair_table(title: str, values: dict[tuple[str, str], Any], formatter=str) -> None:
    print(f"\n{title}")
    print(" " * 14 + "".join(f"{s[:10]:>12s}" for s in SKILL_NAMES))
    for s1 in SKILL_NAMES:
        row = [formatter(values.get((s1, s2))) for s2 in SKILL_NAMES]
        print(f"{s1:12s}  " + "".join(f"{cell:>12s}" for cell in row))


def iter_decisions(rallies: list[dict[str, Any]]):
    for rally_index, rally in enumerate(rallies):
        winner = int(rally.get("winner", 0))
        states = rally.get("states", [])
        pairs = rally.get("skill_pairs", [])
        for t, (state, pair) in enumerate(zip(states, pairs)):
            yield rally_index, t, np.asarray(state), tuple(pair), winner


def summarize_dataset(rallies: list[dict[str, Any]], metadata: dict[str, Any] | None) -> None:
    total_states = sum(len(r.get("states", [])) for r in rallies)
    p1_wins = sum(1 for r in rallies if r.get("winner") == 1)
    p2_wins = sum(1 for r in rallies if r.get("winner") == 2)
    print("DATASET SUMMARY")
    print(f"  rallies: {len(rallies)}")
    print(f"  decision states: {total_states}")
    print(f"  P1 wins: {p1_wins}  P2 wins: {p2_wins}  P1 win rate: {fmt_rate(rate(p1_wins, p1_wins + p2_wins))}")

    p1_counts: Counter[str] = Counter()
    p2_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    pair_wins: Counter[tuple[str, str]] = Counter()
    p1_skill_totals: Counter[str] = Counter()
    p1_skill_wins: Counter[str] = Counter()
    p2_skill_totals: Counter[str] = Counter()
    p2_skill_wins: Counter[str] = Counter()
    slot_mismatches = 0
    side_counts: Counter[str] = Counter()

    for _rally_index, _t, state, pair, winner in iter_decisions(rallies):
        p1, p2 = pair
        p1_counts[p1] += 1
        p2_counts[p2] += 1
        pair_counts[pair] += 1
        p1_skill_totals[p1] += 1
        p2_skill_totals[p2] += 1
        if winner == 1:
            pair_wins[pair] += 1
            p1_skill_wins[p1] += 1
            p2_skill_wins[p2] += 1
        if len(state) >= 2:
            if decode_skill(state[-2]) != p1 or decode_skill(state[-1]) != p2:
                slot_mismatches += 1
        if len(state) >= 24:
            vx = float(state[21])
            if vx > 1e-6:
                side_counts["ball_heading_to_p2"] += 1
            elif vx < -1e-6:
                side_counts["ball_heading_to_p1"] += 1
            else:
                side_counts["ball_vx_zero"] += 1

    print_counter("P1 skill frequency across decision states", p1_counts, total_states)
    print_counter("P2 skill frequency across decision states", p2_counts, total_states)
    print_pair_table("5x5 skill-pair decision-state counts", pair_counts, lambda v: str(v or 0))

    print("\nP1 win rate conditioned on P1 skill")
    for s in SKILL_NAMES:
        print(f"  {s:12s} n={p1_skill_totals[s]:7d}  P1_win={fmt_rate(rate(p1_skill_wins[s], p1_skill_totals[s]))}")
    print("\nP1 win rate conditioned on P2 skill")
    for s in SKILL_NAMES:
        print(f"  {s:12s} n={p2_skill_totals[s]:7d}  P1_win={fmt_rate(rate(p2_skill_wins[s], p2_skill_totals[s]))}")
    print_pair_table(
        "P1 win rate by 5x5 skill pair",
        {pair: rate(pair_wins[pair], pair_counts[pair]) for pair in pair_counts},
        lambda v: "nan" if v is None else fmt_rate(v),
    )

    print("\nSkill-slot consistency")
    print(f"  state[-2:]/skill_pairs mismatches: {slot_mismatches}/{total_states}")
    print("  inferred crossing direction counts from encoded ball vx:")
    for key, count in side_counts.items():
        print(f"    {key}: {count}")

    if metadata and "attempts" in metadata:
        accepted = [row for row in metadata["attempts"] if row.get("accepted")]
        print("\nMetadata split by collection mode")
        mode_counts = Counter(row.get("mode", "") for row in accepted)
        mode_wins = Counter()
        for row in accepted:
            if row.get("winner") == 1:
                mode_wins[row.get("mode", "")] += 1
        for mode in sorted(mode_counts):
            print(f"  {mode:13s} rallies={mode_counts[mode]:5d}  P1_win={fmt_rate(rate(mode_wins[mode], mode_counts[mode]))}")
    else:
        print("\nMetadata split by collection mode: unavailable")


def print_example_timelines(rallies: list[dict[str, Any]], n_examples: int) -> None:
    print(f"\nEXAMPLE RALLY TIMELINES ({n_examples})")
    shown = 0
    for rally_index, rally in enumerate(rallies):
        states = rally.get("states", [])
        pairs = rally.get("skill_pairs", [])
        if not states or not pairs:
            continue
        print(f"\nRally {rally_index}: winner={rally.get('winner')} states={len(states)} pairs={len(pairs)}")
        print(f"  top-level skill1/skill2: {rally.get('skill1')} / {rally.get('skill2')}")
        for t, (state, pair) in enumerate(zip(states, pairs)):
            state = np.asarray(state)
            p1_slot = decode_skill(state[-2])
            p2_slot = decode_skill(state[-1])
            ball = state[18:24].tolist() if len(state) >= 24 else []
            mismatch = "" if (p1_slot, p2_slot) == tuple(pair) else "  MISMATCH"
            print(
                f"  s{t}: pair={tuple(pair)} slots=({p1_slot}, {p2_slot}) "
                f"ball_pos_vel={[round(float(x), 3) for x in ball]}{mismatch}"
            )
        shown += 1
        if shown >= n_examples:
            break


def print_training_semantics(rallies: list[dict[str, Any]], example_index: int | None) -> None:
    print("\nTRAINING TARGET SEMANTICS")
    if example_index is None:
        example_index = next((i for i, r in enumerate(rallies) if len(r.get("states", [])) >= 3), 0)
    rally = rallies[example_index]
    states = rally.get("states", [])
    pairs = rally.get("skill_pairs", [])
    g1, g2 = compute_returns(states, gamma=GAMMA, winner=int(rally.get("winner", 0)))
    print(f"  gamma={GAMMA}")
    print(f"  example rally index={example_index} winner={rally.get('winner')} L={len(states)}")
    print("  rows created by train_q_model_5skill_v3.build_dataset:")
    for t, (pair, v1, v2) in enumerate(zip(pairs, g1, g2)):
        print(f"    t={t} pair={tuple(pair)} Q1_target={v1:+.5f} Q2_target={v2:+.5f}")


def load_simple_model(path: str, potential: bool = False) -> SimpleModel:
    model = SimpleModel(STATE_DIM, [64, 32, 16], 1, last_layer_activation=None if potential else "tanh")
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def model_counterfactual_batch(states: np.ndarray) -> torch.Tensor:
    rows = []
    for state in states:
        base = torch.tensor(state, dtype=torch.float32)
        for p1 in range(N_SKILLS):
            for p2 in range(N_SKILLS):
                x = base.clone()
                x[-2] = p1 / (N_SKILLS - 1)
                x[-1] = p2 / (N_SKILLS - 1)
                rows.append(x)
    return torch.stack(rows)


def summarize_values(name: str, values: np.ndarray) -> None:
    print(
        f"  {name:4s} mean={values.mean():+.4f} std={values.std():.4f} "
        f"min={values.min():+.4f} p05={np.quantile(values, 0.05):+.4f} "
        f"p50={np.quantile(values, 0.50):+.4f} p95={np.quantile(values, 0.95):+.4f} max={values.max():+.4f}"
    )


def summarize_models(
    rallies: list[dict[str, Any]],
    q1_path: str | None,
    q2_path: str | None,
    phi_path: str | None,
    n_states: int,
) -> None:
    if not q1_path or not q2_path:
        print("\nMODEL DIAGNOSTICS: skipped, pass --q1 and --q2")
        return
    states = [state for _ri, _t, state, _pair, _winner in iter_decisions(rallies)]
    if not states:
        print("\nMODEL DIAGNOSTICS: no states")
        return
    states = np.asarray(states[:n_states], dtype=np.float32)
    batch = model_counterfactual_batch(states)
    q1 = load_simple_model(q1_path)
    q2 = load_simple_model(q2_path)
    with torch.no_grad():
        q1_vals = q1(batch).reshape(len(states), N_SKILLS, N_SKILLS).numpy()
        q2_vals = q2(batch).reshape(len(states), N_SKILLS, N_SKILLS).numpy()

    print("\nMODEL DIAGNOSTICS")
    print(f"  states probed: {len(states)}")
    summarize_values("Q1", q1_vals.reshape(-1))
    summarize_values("Q2", q2_vals.reshape(-1))
    qsum = q1_vals + q2_vals
    corr = np.corrcoef(q1_vals.reshape(-1), -q2_vals.reshape(-1))[0, 1]
    print(f"  corr(Q1, -Q2)={corr:+.4f}")
    print(f"  MAE(Q1 + Q2)={np.mean(np.abs(qsum)):.4f}")

    q1_argmax = Counter()
    q2_argmax = Counter()
    for a1, a2 in np.ndindex(len(states), 1):
        del a2
        p1, p2 = np.unravel_index(np.argmax(q1_vals[a1]), (N_SKILLS, N_SKILLS))
        q1_argmax[(skill_from_index(p1), skill_from_index(p2))] += 1
        p1, p2 = np.unravel_index(np.argmax(q2_vals[a1]), (N_SKILLS, N_SKILLS))
        q2_argmax[(skill_from_index(p1), skill_from_index(p2))] += 1
    print_pair_table("Q1 joint argmax pair counts", q1_argmax, lambda v: str(v or 0))
    print_pair_table("Q2 joint argmax pair counts", q2_argmax, lambda v: str(v or 0))

    print_pair_table(
        "Mean Q1 by skill pair",
        {(skill_from_index(i), skill_from_index(j)): q1_vals[:, i, j].mean() for i in range(N_SKILLS) for j in range(N_SKILLS)},
        lambda v: f"{v:+.3f}",
    )
    print_pair_table(
        "Mean Q2 by skill pair",
        {(skill_from_index(i), skill_from_index(j)): q2_vals[:, i, j].mean() for i in range(N_SKILLS) for j in range(N_SKILLS)},
        lambda v: f"{v:+.3f}",
    )

    if phi_path:
        phi = load_simple_model(phi_path, potential=True)
        with torch.no_grad():
            phi_vals = phi(batch).reshape(len(states), N_SKILLS, N_SKILLS).numpy()
        summarize_values("Phi", phi_vals.reshape(-1))
        hard_p1 = Counter(skill_from_index(int(np.argmax(phi_vals.max(axis=2)[i]))) for i in range(len(states)))
        hard_p2 = Counter(skill_from_index(int(np.argmax(phi_vals.max(axis=1)[i]))) for i in range(len(states)))
        br_p1_left = Counter(skill_from_index(int(np.argmax(phi_vals[i, :, skill_index("left")]))) for i in range(len(states)))
        br_p2_left = Counter(skill_from_index(int(np.argmax(phi_vals[i, skill_index("left"), :]))) for i in range(len(states)))
        print_counter("nash-p-hard P1 row selections from Phi", hard_p1, len(states))
        print_counter("nash-p-hard P2 column selections from Phi", hard_p2, len(states))
        print_counter("nash-p-br P1 selections when other=left", br_p1_left, len(states))
        print_counter("nash-p-br P2 selections when other=left", br_p2_left, len(states))
        print_pair_table(
            "Mean Phi by skill pair",
            {(skill_from_index(i), skill_from_index(j)): phi_vals[:, i, j].mean() for i in range(N_SKILLS) for j in range(N_SKILLS)},
            lambda v: f"{v:+.3f}",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose 5-skill v3 asymmetry without changing production code.")
    parser.add_argument("--rallies", default="data/rallies_5skill_v3_aligned_10k.pkl")
    parser.add_argument("--q1", default=None)
    parser.add_argument("--q2", default=None)
    parser.add_argument("--phi", default=None)
    parser.add_argument("--model-states", type=int, default=1000)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--example-index", type=int, default=None)
    args = parser.parse_args()

    rallies = load_rallies(args.rallies)
    metadata = load_metadata(args.rallies)
    summarize_dataset(rallies, metadata)
    print_example_timelines(rallies, args.examples)
    print_training_semantics(rallies, args.example_index)
    summarize_models(rallies, args.q1, args.q2, args.phi, args.model_states)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
