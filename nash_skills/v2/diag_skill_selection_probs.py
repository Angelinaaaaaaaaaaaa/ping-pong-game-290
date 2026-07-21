"""
Diagnostic: compare probabilistic skill-selection modes without running MuJoCo.

Loads the potential model (model_p_5skill_v3.pth by default) and a rally
pickle.  For a random sample of states, builds the Φ table and shows the
selection probability each mode assigns to each skill — making it easy to
see whether one skill monopolises (a sign of Q-model bias).

No environment or PPO controller is touched.

Usage
-----
    PYTHONPATH=. venv/bin/python nash_skills/v2/diag_skill_selection_probs.py \\
        --rallies data/rallies_5skill_v2.pkl \\
        --model   models/model_p_5skill_v3.pth \\
        --n-samples 200 \\
        --temperature 0.5 \\
        --epsilon 0.1

All arguments are optional; see --help for defaults.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch

from nash_skills.skills import SKILL_NAMES, N_SKILLS
from nash_skills.v2.skill_selection import (
    softmax_probs,
    epsilon_mix_probs,
    select_skill_from_values,
)

_DEFAULT_RALLIES = "data/rallies_5skill_v2.pkl"
_DEFAULT_MODEL   = "models/model_p_5skill_v3.pth"


def _load_model(model_path: str):
    """Load SimpleModel potential — returns (model, state_dim) or None on failure.

    State dim is auto-detected from the first FC layer weight shape so the same
    script works for both 76-dim (v3) and 12-dim (gantry/gantry_sym) models.
    """
    try:
        from model_arch import SimpleModel
    except ImportError:
        warnings.warn("model_arch not importable — running in state-free mode.")
        return None, None
    if not Path(model_path).exists():
        warnings.warn(f"Model not found: {model_path} — running in state-free mode.")
        return None, None

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    if "fc.0.weight" not in state_dict:
        # Not a SimpleModel checkpoint -- e.g. FactoredModel (separate
        # state_encoder/skill_encoder/fusion branches). This script only
        # supports SimpleModel's flat-concat architecture; fail gracefully
        # rather than crashing on a KeyError.
        warnings.warn(
            f"{model_path} is not a SimpleModel checkpoint (missing 'fc.0.weight' "
            "-- likely FactoredModel). Skipping; this diagnostic does not yet "
            "support the factored architecture. Running in state-free mode."
        )
        return None, None

    # Infer input dim from first FC layer: shape is (out_features, in_features)
    state_dim = state_dict["fc.0.weight"].shape[1]
    model = SimpleModel(state_dim, [64, 32, 16], 1, last_layer_activation=None)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Auto-detected state_dim={state_dim} from {model_path}")
    return model, state_dim


def _build_phi_from_state(state_vec: np.ndarray, model) -> np.ndarray:
    """
    Build the N×N potential table for a single encoded state vector.

    Returns phi as a (N_SKILLS, N_SKILLS) float32 numpy array.
    The last two dims of state_vec are the normalised skill indices
    (convention for both 76-dim and 12-dim encoders); we overwrite them
    for each (s1, s2) pair.
    """
    from nash_skills.skills import skill_index
    skill_norm = np.array(
        [skill_index(s) / (N_SKILLS - 1) for s in SKILL_NAMES], dtype=np.float32
    )
    # Skill indices always occupy the last two dimensions of the state vector
    ego_dim = len(state_vec) - 2
    opp_dim = len(state_vec) - 1

    batch_inputs = []
    for i in range(N_SKILLS):
        for j in range(N_SKILLS):
            s = state_vec.copy()
            s[ego_dim] = skill_norm[i]
            s[opp_dim] = skill_norm[j]
            batch_inputs.append(s)

    batch = torch.tensor(np.stack(batch_inputs), dtype=torch.float32)
    with torch.no_grad():
        vals = model(batch)[:, 0].numpy()
    return vals.reshape(N_SKILLS, N_SKILLS)


def _action_scores_ego(phi: np.ndarray, strategy: str) -> np.ndarray:
    """Reduce N×N phi to N ego-action scores using the given strategy."""
    if strategy == "hard":
        return phi.max(axis=1)
    if strategy == "minimax":
        return phi.min(axis=1)
    raise ValueError(f"Unknown strategy: {strategy!r}")


def _mode_probs(action_scores: np.ndarray, temperature: float, epsilon: float) -> dict:
    """Return a dict of mode → probability vector over skills."""
    n = len(action_scores)
    uniform = np.ones(n) / n
    argmax_vec = np.zeros(n)
    argmax_vec[int(np.argmax(action_scores))] = 1.0

    sm = softmax_probs(action_scores, temperature)
    sm_eps = epsilon_mix_probs(sm, epsilon, n)
    am_eps = (1 - epsilon) * argmax_vec + epsilon * uniform

    return {
        "argmax":          argmax_vec,
        "softmax":         sm,
        "epsilon_argmax":  am_eps,
        "epsilon_softmax": sm_eps,
    }


def _print_prob_table(all_mode_probs: dict, title: str) -> None:
    """Print average probabilities across sampled states."""
    modes = list(all_mode_probs.keys())
    avg = {m: np.mean(all_mode_probs[m], axis=0) for m in modes}

    print(f"\n{title}")
    print("-" * (16 + 10 * N_SKILLS))
    hdr = f"{'mode':<18}" + "".join(f"{s:>10}" for s in SKILL_NAMES)
    print(hdr)
    print("-" * (16 + 10 * N_SKILLS))
    for m in modes:
        row = f"{m:<18}" + "".join(f"{p:>9.1%} " for p in avg[m])
        print(row)
    print()

    # Highlight skew: max/min ratio
    for m in modes:
        p = avg[m]
        dominant = SKILL_NAMES[int(np.argmax(p))]
        skew = p.max() / (p.min() + 1e-9)
        print(f"  {m:<18}  dominant={dominant}  max/min={skew:.1f}x")


def _synthetic_phi_demo(temperature: float, epsilon: float) -> None:
    """Demo on a hand-crafted phi table — no model needed."""
    print("\n[Synthetic demo — no model loaded]")
    print("Phi table (ego=row, opp=col): right_short dominates.")
    # Craft a phi where right_short (idx 3) is clearly best
    phi = np.array([
        [0.1, 0.2, 0.15, 0.2,  0.1 ],
        [0.2, 0.3, 0.25, 0.35, 0.2 ],
        [0.15,0.25,0.2, 0.3,  0.15],
        [0.6, 0.65,0.62, 0.7,  0.6 ],
        [0.1, 0.2, 0.15, 0.2,  0.1 ],
    ], dtype=np.float32)

    for strategy in ("hard", "minimax"):
        scores = _action_scores_ego(phi, strategy)
        mode_probs_single = _mode_probs(scores, temperature, epsilon)
        all_mode_probs = {m: np.array([v]) for m, v in mode_probs_single.items()}
        _print_prob_table(all_mode_probs, f"Synthetic phi — strategy={strategy}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Skill-selection probability diagnostic (no MuJoCo)")
    ap.add_argument("--rallies",     default=_DEFAULT_RALLIES,
                    help=f"Rally pickle (default: {_DEFAULT_RALLIES})")
    ap.add_argument("--model",       default=_DEFAULT_MODEL,
                    help=f"Potential model path (default: {_DEFAULT_MODEL})")
    ap.add_argument("--n-samples",   type=int, default=100,
                    help="Number of randomly sampled states to average over (default: 100)")
    ap.add_argument("--temperature", type=float, default=0.5,
                    help="Softmax temperature for softmax/epsilon_softmax modes (default: 0.5)")
    ap.add_argument("--epsilon",     type=float, default=0.1,
                    help="Exploration rate for epsilon_* modes (default: 0.1)")
    ap.add_argument("--strategy",    choices=["hard", "minimax"], default="hard",
                    help="How to aggregate phi → ego action scores (default: hard)")
    ap.add_argument("--seed",        type=int, default=42,
                    help="RNG seed for state sampling (default: 42)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print("Nash Skills — Skill Selection Probability Diagnostic")
    print(f"  temperature = {args.temperature}")
    print(f"  epsilon     = {args.epsilon}")
    print(f"  strategy    = {args.strategy}")
    print(f"  seed        = {args.seed}")

    model, _state_dim = _load_model(args.model)
    if model is None:
        _synthetic_phi_demo(args.temperature, args.epsilon)
        return

    rally_path = Path(args.rallies)
    if not rally_path.exists():
        print(f"\nRallies not found: {args.rallies} — using synthetic demo.")
        _synthetic_phi_demo(args.temperature, args.epsilon)
        return

    with open(rally_path, "rb") as f:
        rallies = pickle.load(f)
    print(f"\nLoaded {len(rallies)} rallies from {args.rallies}")

    # Collect encoded states from rallies — match the model's detected state_dim
    states = []
    for r in rng.permutation(len(rallies)):
        for s in rallies[r].get("states", []):
            arr = np.asarray(s, dtype=np.float32)
            if arr.shape[0] == _state_dim:
                states.append(arr)
            if len(states) >= args.n_samples:
                break
        if len(states) >= args.n_samples:
            break

    if not states:
        print(f"No valid {_state_dim}-dim states found in rally data — using synthetic demo.")
        _synthetic_phi_demo(args.temperature, args.epsilon)
        return

    print(f"Sampling {len(states)} states...")

    all_mode_probs: dict[str, list] = {m: [] for m in
                                        ("argmax", "softmax", "epsilon_argmax", "epsilon_softmax")}

    for state_vec in states:
        phi = _build_phi_from_state(state_vec, model)
        scores = _action_scores_ego(phi, args.strategy)
        for m, pv in _mode_probs(scores, args.temperature, args.epsilon).items():
            all_mode_probs[m].append(pv)

    all_mode_probs_np = {m: np.array(v) for m, v in all_mode_probs.items()}

    _print_prob_table(
        all_mode_probs_np,
        f"Selection probabilities (strategy={args.strategy}, n={len(states)} states)"
    )

    print("\nInterpretation:")
    print("  argmax      — current eval behavior (deterministic)")
    print("  softmax     — temperature-controlled exploration")
    print("  epsilon_argmax — argmax with ε-uniform noise")
    print("  epsilon_softmax — softmax + ε-uniform mixing")
    print("  A max/min ratio near 1.0x → balanced; >> 5x → skill bias.")


if __name__ == "__main__":
    main()
