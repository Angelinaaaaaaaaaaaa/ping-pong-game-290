"""
Ablation runner: sweep the skill-selection probability diagnostic across
every compatible (model, rally-data) pair found in two directories -- pure
Python + one forward pass per sampled state, no MuJoCo, no training.

Auto-pairs each model checkpoint in --models-dir with a rally pickle in
--rallies-dir that has matching state_dim (detected the same way as
diag_skill_selection_probs.py), then reuses that script's internals to
compute per-mode selection probabilities and prints one consolidated
comparison table instead of requiring one command per model.

Usage
-----
    PYTHONPATH=. venv/bin/python nash_skills/v2/run_selection_ablation.py \\
        --models-dir models_new --rallies-dir data_new \\
        --n-samples 200 --temperature 0.5 --epsilon 0.1
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from nash_skills.skills import SKILL_NAMES
from nash_skills.v2.diag_skill_selection_probs import (
    _load_model,
    _build_phi_from_state,
    _action_scores_ego,
    _mode_probs,
)

__all__ = ["match_rally_to_model", "compute_skew_summary"]

_EPS = 1e-9


def match_rally_to_model(state_dim: int, rally_dim_map: dict) -> Optional[str]:
    """
    Pick a rally file path whose detected state_dim matches `state_dim`.

    If multiple rally files match, returns the alphabetically first path
    for deterministic output. Returns None if no rally file matches.
    """
    candidates = sorted(p for p, d in rally_dim_map.items() if d == state_dim)
    return candidates[0] if candidates else None


def compute_skew_summary(avg_probs: np.ndarray, skill_names: list) -> dict:
    """
    Summarize an average selection-probability vector.

    Returns {"dominant": skill name with highest avg probability,
             "max_min_ratio": max/min ratio (epsilon-guarded against 0)}.
    """
    dominant = skill_names[int(np.argmax(avg_probs))]
    ratio = float(avg_probs.max() / (avg_probs.min() + _EPS))
    return {"dominant": dominant, "max_min_ratio": ratio}


def _detect_rally_state_dim(rally_path: Path) -> Optional[int]:
    try:
        with open(rally_path, "rb") as f:
            rallies = pickle.load(f)
    except Exception as e:
        warnings.warn(f"Could not load {rally_path}: {e}")
        return None
    for r in rallies:
        if not isinstance(r, dict):
            # Malformed/legacy pickle (e.g. a plain list) -- not a rally dict.
            return None
        states = r.get("states", [])
        if states:
            return len(np.asarray(states[0]))
    return None


def _detect_model_state_dim(model_path: Path) -> Optional[int]:
    import torch
    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    except Exception as e:
        warnings.warn(f"Could not load {model_path}: {e}")
        return None
    if "fc.0.weight" not in state_dict:
        return None  # FactoredModel or unsupported architecture
    return int(state_dict["fc.0.weight"].shape[1])


def _load_states_for_dim(rally_path: Path, state_dim: int, n_samples: int, rng) -> list:
    with open(rally_path, "rb") as f:
        rallies = pickle.load(f)
    states = []
    for r in rng.permutation(len(rallies)):
        for s in rallies[r].get("states", []):
            arr = np.asarray(s, dtype=np.float32)
            if arr.shape[0] == state_dim:
                states.append(arr)
            if len(states) >= n_samples:
                break
        if len(states) >= n_samples:
            break
    return states


def run_ablation(
    models_dir: str,
    rallies_dir: str,
    n_samples: int = 200,
    temperature: float = 0.5,
    epsilon: float = 0.1,
    strategy: str = "hard",
    seed: int = 42,
) -> list:
    """
    Run the diagnostic across every compatible (model, rallies) pair found
    in models_dir / rallies_dir. Returns a list of row dicts, one per
    (model, mode) combination that had matching data.
    """
    models_dir_p = Path(models_dir)
    rallies_dir_p = Path(rallies_dir)

    model_paths = sorted(models_dir_p.glob("model_p*.pth"))
    rally_paths = sorted(rallies_dir_p.glob("*.pkl"))

    rally_dim_map = {}
    for rp in rally_paths:
        dim = _detect_rally_state_dim(rp)
        if dim is not None:
            rally_dim_map[str(rp)] = dim

    rows = []
    for mp in model_paths:
        state_dim = _detect_model_state_dim(mp)
        if state_dim is None:
            rows.append({
                "model": mp.name, "rallies": None, "state_dim": None,
                "status": "SKIPPED (unsupported architecture)",
            })
            continue

        rally_path = match_rally_to_model(state_dim, rally_dim_map)
        if rally_path is None:
            rows.append({
                "model": mp.name, "rallies": None, "state_dim": state_dim,
                "status": f"SKIPPED (no {state_dim}-dim rally file found)",
            })
            continue

        model, _ = _load_model(str(mp))
        if model is None:
            rows.append({
                "model": mp.name, "rallies": Path(rally_path).name, "state_dim": state_dim,
                "status": "SKIPPED (model failed to load)",
            })
            continue

        rng = np.random.default_rng(seed)
        states = _load_states_for_dim(Path(rally_path), state_dim, n_samples, rng)
        if not states:
            rows.append({
                "model": mp.name, "rallies": Path(rally_path).name, "state_dim": state_dim,
                "status": "SKIPPED (no matching states in rally file)",
            })
            continue

        all_mode_probs = {m: [] for m in
                          ("argmax", "softmax", "epsilon_argmax", "epsilon_softmax")}
        for state_vec in states:
            phi = _build_phi_from_state(state_vec, model)
            scores = _action_scores_ego(phi, strategy)
            for m, pv in _mode_probs(scores, temperature, epsilon).items():
                all_mode_probs[m].append(pv)

        for mode, probs_list in all_mode_probs.items():
            avg = np.mean(np.array(probs_list), axis=0)
            summary = compute_skew_summary(avg, SKILL_NAMES)
            rows.append({
                "model": mp.name,
                "rallies": Path(rally_path).name,
                "state_dim": state_dim,
                "mode": mode,
                "n_samples": len(states),
                "dominant": summary["dominant"],
                "max_min_ratio": summary["max_min_ratio"],
                "status": "OK",
            })

    return rows


def _print_ablation_table(rows: list) -> None:
    ok_rows = [r for r in rows if r["status"] == "OK"]
    skip_rows = [r for r in rows if r["status"] != "OK"]

    if ok_rows:
        hdr = (f"{'model':<32} {'rallies':<28} {'dim':>4} {'mode':<16} "
               f"{'n':>5} {'dominant':<14} {'max/min':>10}")
        print(hdr)
        print("-" * len(hdr))
        for r in ok_rows:
            print(
                f"{r['model']:<32} {r['rallies']:<28} {r['state_dim']:>4} "
                f"{r['mode']:<16} {r['n_samples']:>5} {r['dominant']:<14} "
                f"{r['max_min_ratio']:>9.1f}x"
            )

    if skip_rows:
        print("\nSkipped:")
        for r in skip_rows:
            print(f"  {r['model']:<32} {r['status']}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sweep the skill-selection diagnostic across all compatible "
                    "(model, rally-data) pairs found in two directories (no MuJoCo)."
    )
    ap.add_argument("--models-dir", default="models_new", dest="models_dir")
    ap.add_argument("--rallies-dir", default="data_new", dest="rallies_dir")
    ap.add_argument("--n-samples", type=int, default=200, dest="n_samples")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--strategy", choices=["hard", "minimax"], default="hard")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Nash Skills — Selection Probability Ablation")
    print(f"  models_dir  = {args.models_dir}")
    print(f"  rallies_dir = {args.rallies_dir}")
    print(f"  n_samples   = {args.n_samples}")
    print(f"  temperature = {args.temperature}")
    print(f"  epsilon     = {args.epsilon}")
    print(f"  strategy    = {args.strategy}\n")

    rows = run_ablation(
        args.models_dir, args.rallies_dir,
        n_samples=args.n_samples, temperature=args.temperature,
        epsilon=args.epsilon, strategy=args.strategy, seed=args.seed,
    )
    _print_ablation_table(rows)


if __name__ == "__main__":
    main()
