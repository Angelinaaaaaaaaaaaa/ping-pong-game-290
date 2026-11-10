"""
Cross-reference the learned potential Phi against the empirical win-rate
heatmap -- reuses existing tooling, no new matchup data or MuJoCo needed
(meeting note item 11, offline half).

diag_skill_selection_probs.py already samples encoded states from a rally
pickle and builds a per-state (N,N) Phi table via a real loaded model.
plot_winrate_heatmap.py already builds the empirical (N,N) ego-win-rate
matrix from the same rally pickle's decided outcomes. This script averages
Phi over many sampled states into one (N,N) matrix and correlates it,
cell-by-cell, against the empirical win-rate matrix: a learned Phi that
tracks real outcomes should show a strong positive correlation.

Usage
-----
    venv/bin/python nash_skills/v2/correlate_phi_winrate.py
    venv/bin/python nash_skills/v2/correlate_phi_winrate.py \\
        --rallies data_new/rallies_v2_1000.pkl \\
        --model   models_new/model_p_5skill_v3.pth \\
        --n-samples 300
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import pickle
from pathlib import Path
from typing import Callable, Optional

import numpy as np

__all__ = [
    "build_avg_phi_matrix",
    "pearson_correlation_masked",
    "correlate_phi_with_winrate",
]

_DEFAULT_RALLIES = "data/rallies_5skill_v2.pkl"
_DEFAULT_MODEL = "models/model_p_5skill_v3.pth"


def build_avg_phi_matrix(
    rallies: list,
    phi_fn: Callable[[np.ndarray], np.ndarray],
    state_dim: int,
    n_samples: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> Optional[np.ndarray]:
    """
    Sample up to n_samples encoded states (matching state_dim) from rallies,
    evaluate phi_fn on each to get an (N,N) Phi table, and average them.

    phi_fn is a plain callable (state_vec) -> (N,N) ndarray so this function
    stays torch-free and testable; the CLI wires it to a real loaded model.

    Returns None if no states of the requested dimension were found (or
    `rallies` is empty).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    samples = []
    for idx in rng.permutation(len(rallies)):
        for s in rallies[idx].get("states", []):
            arr = np.asarray(s, dtype=np.float32)
            if arr.shape[0] == state_dim:
                samples.append(arr)
            if len(samples) >= n_samples:
                break
        if len(samples) >= n_samples:
            break

    if not samples:
        return None

    tables = [phi_fn(s) for s in samples]
    return np.mean(np.stack(tables), axis=0)


def pearson_correlation_masked(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """
    Pearson r between two same-shape matrices, ignoring cells where either
    is NaN. Returns None if fewer than 2 valid pairs remain, or if either
    masked array is constant (zero variance -> correlation undefined).
    """
    a_flat = np.asarray(a, dtype=np.float64).flatten()
    b_flat = np.asarray(b, dtype=np.float64).flatten()
    mask = ~(np.isnan(a_flat) | np.isnan(b_flat))
    if mask.sum() < 2:
        return None

    av = a_flat[mask]
    bv = b_flat[mask]
    if np.std(av) == 0 or np.std(bv) == 0:
        return None

    return float(np.corrcoef(av, bv)[0, 1])


def correlate_phi_with_winrate(
    avg_phi: np.ndarray,
    winrate_matrix: np.ndarray,
    skill_names: list,
) -> dict:
    """
    Combine the Pearson correlation with a per-cell (ego, opp, phi, win_rate)
    breakdown for cells that have empirical win-rate data (non-NaN).
    """
    r = pearson_correlation_masked(avg_phi, winrate_matrix)
    n = len(skill_names)
    cells = []
    for i in range(n):
        for j in range(n):
            wr = winrate_matrix[i, j]
            if not np.isnan(wr):
                cells.append({
                    "ego_skill": skill_names[i],
                    "opp_skill": skill_names[j],
                    "phi": float(avg_phi[i, j]),
                    "win_rate": float(wr),
                })
    return {"pearson_r": r, "n_cells": len(cells), "cells": cells}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Correlate learned Phi against the empirical win-rate heatmap (no MuJoCo)."
    )
    ap.add_argument("--rallies", default=_DEFAULT_RALLIES,
                    help=f"Rally pickle (default: {_DEFAULT_RALLIES})")
    ap.add_argument("--model", default=_DEFAULT_MODEL,
                    help=f"Potential model path (default: {_DEFAULT_MODEL})")
    ap.add_argument("--n-samples", type=int, default=200, dest="n_samples",
                    help="Number of randomly sampled states to average Phi over (default: 200)")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for state sampling (default: 42)")
    args = ap.parse_args()

    from nash_skills.skills import SKILL_NAMES
    from nash_skills.v2.diag_skill_selection_probs import (
        _load_model,
        _build_phi_from_state,
        _validate_rally_schema,
    )
    from nash_skills.v2.plot_winrate_heatmap import build_winrate_matrix, build_counts_matrix

    rally_path = Path(args.rallies)
    if not rally_path.exists():
        raise SystemExit(f"Rally file not found: {args.rallies}")

    with open(rally_path, "rb") as f:
        rallies = pickle.load(f)
    print(f"Loaded {len(rallies)} rallies from {args.rallies}")
    _validate_rally_schema(rallies, args.rallies)

    model, state_dim = _load_model(args.model)
    if model is None:
        raise SystemExit(f"Could not load a usable model from {args.model}")

    winrate = build_winrate_matrix(rallies, SKILL_NAMES)
    _, counts = build_counts_matrix(rallies, SKILL_NAMES)

    rng = np.random.default_rng(args.seed)
    avg_phi = build_avg_phi_matrix(
        rallies,
        phi_fn=lambda s: _build_phi_from_state(s, model),
        state_dim=state_dim,
        n_samples=args.n_samples,
        rng=rng,
    )
    if avg_phi is None:
        raise SystemExit(f"No {state_dim}-dim states found in {args.rallies}")

    result = correlate_phi_with_winrate(avg_phi, winrate, SKILL_NAMES)

    print(f"\nPhi vs empirical win-rate correlation "
          f"({result['n_cells']} skill-pair cells with decided-rally data)")
    if result["pearson_r"] is not None:
        print(f"  Pearson r = {result['pearson_r']:.3f}")
    else:
        print("  Pearson r = N/A (not enough data)")

    print(f"\n{'ego':<14}{'opp':<14}{'phi':>10}{'win_rate':>10}{'n':>6}")
    for cell in sorted(result["cells"], key=lambda c: -c["phi"]):
        i = SKILL_NAMES.index(cell["ego_skill"])
        j = SKILL_NAMES.index(cell["opp_skill"])
        n_decided = int(counts[i, j])
        print(f"{cell['ego_skill']:<14}{cell['opp_skill']:<14}"
              f"{cell['phi']:>10.3f}{cell['win_rate']:>10.1%}{n_decided:>6}")


if __name__ == "__main__":
    main()
