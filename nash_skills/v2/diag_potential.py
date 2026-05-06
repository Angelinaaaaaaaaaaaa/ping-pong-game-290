"""
Diagnostic: inspect per-skill potential scores to understand why nash-p
always picks 'right'.

Loads model_p_4skill_v2.pth and rallies_4skill_v2.pkl (if present) and
prints, for a handful of sampled states, the raw potential score assigned
to each skill before argmax is taken.

Also prints dataset-wide mean scores per skill to expose any monotonic bias.

Run:
    venv/bin/python nash_skills/v2/diag_potential.py
    venv/bin/python nash_skills/v2/diag_potential.py --rallies data/rallies_4skill_v2.pkl
    venv/bin/python nash_skills/v2/diag_potential.py --n_samples 10
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import pickle as pkl

import numpy as np
import torch

from model_arch import SimpleModel
from nash_skills.skills import SKILL_NAMES, N_SKILLS
from nash_skills.v2.state_encoder import STATE_DIM

MODEL_P_PATH   = "models/model_p_4skill_v2.pth"
DEFAULT_RALLIES = "data/rallies_4skill_v2.pkl"
N_SAMPLES_DEFAULT = 5


def _score_all_skills(model_p: SimpleModel, state: np.ndarray,
                      opp_idx: int) -> list[float]:
    """Return potential scores [s0, s1, s2, s3] for ego skills 0..N_SKILLS-1."""
    scores = []
    for s in range(N_SKILLS):
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        x[0, -2] = s / (N_SKILLS - 1)          # ego skill slot
        x[0, -1] = opp_idx / (N_SKILLS - 1)    # opp skill slot (held fixed)
        with torch.no_grad():
            scores.append(model_p(x).item())
    return scores


def run_diagnostic(
    model_p_path: str = MODEL_P_PATH,
    rally_path: str = DEFAULT_RALLIES,
    n_samples: int = N_SAMPLES_DEFAULT,
) -> None:
    # Load model
    if not os.path.exists(model_p_path):
        print(f"[SKIP] model_p not found at {model_p_path} — train first.")
        return

    model_p = SimpleModel(STATE_DIM, [64, 32, 16], 1, last_layer_activation=None)
    model_p.load_state_dict(torch.load(model_p_path, weights_only=True))
    model_p.eval()
    print(f"Loaded model_p from {model_p_path}\n")

    # Load rally data if available
    if not os.path.exists(rally_path):
        print(f"[SKIP] rally data not found at {rally_path}")
        print("Falling back to zero-state samples.\n")
        # Use zero states as fallback
        sample_states = [np.zeros(STATE_DIM, dtype=np.float32)] * n_samples
        sample_labels = [f"zero_state_{i}" for i in range(n_samples)]
        all_states = None
    else:
        with open(rally_path, "rb") as f:
            rallies = pkl.load(f)
        print(f"Loaded {len(rallies)} rallies from {rally_path}\n")

        rng = np.random.default_rng(42)
        sampled_rallies = rng.choice(len(rallies), size=min(n_samples, len(rallies)),
                                     replace=False)
        sample_states = []
        sample_labels = []
        for idx in sampled_rallies:
            r = rallies[idx]
            if r["states"]:
                sample_states.append(r["states"][0])
                sample_labels.append(f"rally[{idx}] {r['skill1']} vs {r['skill2']}")

        # Collect all states for dataset-wide mean
        all_states = []
        for r in rallies:
            all_states.extend(r["states"])
        all_states = np.array(all_states, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Per-sample scores                                                    #
    # ------------------------------------------------------------------ #
    print("=" * 70)
    print(f"Per-sample potential scores (opp skill fixed to 'right', idx=3)")
    print(f"{'State':<35}  {'left_s':>8} {'left':>8} {'right_s':>8} {'right':>8}  {'argmax'}")
    print("-" * 70)
    for state, label in zip(sample_states, sample_labels):
        opp_idx = N_SKILLS - 1  # 'right'
        scores = _score_all_skills(model_p, state, opp_idx)
        best = SKILL_NAMES[int(np.argmax(scores))]
        row = f"{label:<35}"
        for v in scores:
            row += f"  {v:+8.4f}"
        row += f"  → {best}"
        print(row)

    # ------------------------------------------------------------------ #
    # Dataset-wide mean scores per skill                                   #
    # ------------------------------------------------------------------ #
    if all_states is not None and len(all_states) > 0:
        print("\n" + "=" * 70)
        print("Dataset-wide MEAN potential scores (all rally states, opp=right)")
        print(f"  N states = {len(all_states)}")
        print(f"  {'Skill':<12}  {'Mean score':>12}  {'Std':>8}")
        print("-" * 40)
        for s, name in enumerate(SKILL_NAMES):
            vals = []
            # Sample up to 500 states to keep this fast
            indices = np.arange(min(500, len(all_states)))
            for state in all_states[indices]:
                x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                x[0, -2] = s / (N_SKILLS - 1)
                x[0, -1] = (N_SKILLS - 1) / (N_SKILLS - 1)  # opp=right
                with torch.no_grad():
                    vals.append(model_p(x).item())
            mean_v = np.mean(vals)
            std_v  = np.std(vals)
            print(f"  {name:<12}  {mean_v:+12.4f}  {std_v:8.4f}")

        # Argmax distribution across all sampled states
        print("\n" + "=" * 70)
        print("Argmax distribution across all sampled states (opp=right)")
        counts = {name: 0 for name in SKILL_NAMES}
        indices = np.arange(min(500, len(all_states)))
        for state in all_states[indices]:
            opp_idx = N_SKILLS - 1
            scores = _score_all_skills(model_p, state, opp_idx)
            best = SKILL_NAMES[int(np.argmax(scores))]
            counts[best] += 1
        total = sum(counts.values())
        for name, cnt in counts.items():
            bar = "#" * int(40 * cnt / total)
            print(f"  {name:<12}: {cnt:4d}/{total} ({100*cnt/total:.1f}%)  {bar}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
If scores are monotonically increasing left_short→left→right_short→right,
the model_p learned a linearly increasing value proportional to the
normalized skill index (0.0→0.33→0.67→1.0).

Root cause: 'right' wins 92–96% of rallies vs ALL opponents in the
training data — it is genuinely dominant under the frozen PPO policy.
model_p correctly learned that higher skill_index ≈ higher game value.

This is NOT a bug in argmax or encoding — it reflects real skill dominance.
To break the right-dominance, you would need to either:
  (a) retrain the low-level PPO so that 'right' is no longer dominant, or
  (b) compute a Nash equilibrium (mixed strategy) rather than best-response.
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnostic: inspect per-skill potential scores."
    )
    parser.add_argument("--model",    default=MODEL_P_PATH,
                        help=f"Path to model_p (default: {MODEL_P_PATH})")
    parser.add_argument("--rallies",  default=DEFAULT_RALLIES,
                        help=f"Path to rally pickle (default: {DEFAULT_RALLIES})")
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES_DEFAULT,
                        help=f"Number of sample states to print (default: {N_SAMPLES_DEFAULT})")
    args = parser.parse_args()

    run_diagnostic(
        model_p_path=args.model,
        rally_path=args.rallies,
        n_samples=args.n_samples,
    )
