"""
Brute-force joint skill search vs. nash-p best-response approximation.

For each sampled state, we:
  1. Enumerate all 25 (s1, s2) joint skill pairs and score them with the
     potential model → brute_force_joint(state, model) → (s1*, s2*)
  2. Run the iterated best-response used by nash-p:
       s1 = argmax_s1 V(s1, s2=0)   (ego picks first, opp fixed at 0)
       s2 = argmax_s2 V(s1*, s2)    (opp best-responds to ego's pick)
     → nash_p_pick(state, model) → (s1, s2)
  3. Record agreement / disagreement across a corpus of states.

Run standalone:
    python nash_skills/brute_force_compare.py [--n-states 200]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from collections import Counter
from typing import List, Tuple, Dict

import numpy as np
import torch

from nash_skills.skills import SKILL_NAMES, N_SKILLS


# ─────────────────────────────────────────────────────────────────────────── #
# Core primitives                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

def joint_potential(state: np.ndarray, model) -> np.ndarray:
    """
    Score all N_SKILLS × N_SKILLS joint skill pairs.

    Returns an (N_SKILLS, N_SKILLS) float32 array where
    scores[i, j] = V(state | s1=i, s2=j).

    obs[-2] encodes s1 as i / (N_SKILLS-1)
    obs[-1] encodes s2 as j / (N_SKILLS-1)
    """
    n = N_SKILLS
    # Build 25-row batch: each row is a copy of state with obs[-2:] patched
    batch = np.tile(state, (n * n, 1))                # (25, 116)
    idx = 0
    for i in range(n):
        for j in range(n):
            batch[idx, -2] = i / (n - 1)
            batch[idx, -1] = j / (n - 1)
            idx += 1

    x = torch.tensor(batch, dtype=torch.float32)
    with torch.no_grad():
        vals = model(x).squeeze(1).numpy()            # (25,)

    return vals.reshape(n, n).astype(np.float32)


def brute_force_joint(state: np.ndarray, model) -> Tuple[int, int]:
    """Return (s1, s2) that maximise V jointly over all 25 pairs."""
    scores = joint_potential(state, model)
    flat_idx = int(np.argmax(scores))
    s1 = flat_idx // N_SKILLS
    s2 = flat_idx % N_SKILLS
    return s1, s2


def nash_p_pick(state: np.ndarray, model) -> Tuple[int, int]:
    """
    Iterated best-response approximation (mirrors eval_matchup.make_picker).

    Step 1: ego picks s1 = argmax_s V(state | s1=s, s2=0)
    Step 2: opp picks s2 = argmax_s V(state | s1=s1*, s2=s)
    """
    n = N_SKILLS

    # Ego best-response (s2 fixed at 0)
    ego_vals = np.empty(n, dtype=np.float32)
    for s in range(n):
        obs = state.copy()
        obs[-2] = s / (n - 1)
        obs[-1] = 0.0
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            ego_vals[s] = model(x).item()
    best_s1 = int(np.argmax(ego_vals))

    # Opp best-response (s1 fixed at best_s1)
    opp_vals = np.empty(n, dtype=np.float32)
    for s in range(n):
        obs = state.copy()
        obs[-2] = best_s1 / (n - 1)
        obs[-1] = s / (n - 1)
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            opp_vals[s] = model(x).item()
    best_s2 = int(np.argmax(opp_vals))

    return best_s1, best_s2


# ─────────────────────────────────────────────────────────────────────────── #
# Comparison over a state corpus                                              #
# ─────────────────────────────────────────────────────────────────────────── #

def compare_on_states(states: List[np.ndarray], model) -> Dict:
    """
    Run both methods on every state and measure agreement.

    Returns dict with:
      n_states       — number of states evaluated
      n_agree        — how many states both methods pick the same (s1, s2)
      agreement_rate — n_agree / n_states (None if n_states == 0)
      disagreements  — list of dicts {state_idx, bf_s1, bf_s2, np_s1, np_s2}
    """
    n_agree = 0
    disagreements = []

    for idx, state in enumerate(states):
        bf_s1, bf_s2 = brute_force_joint(state, model)
        np_s1, np_s2 = nash_p_pick(state, model)

        if bf_s1 == np_s1 and bf_s2 == np_s2:
            n_agree += 1
        else:
            disagreements.append({
                'state_idx': idx,
                'bf_s1': bf_s1, 'bf_s2': bf_s2,
                'np_s1': np_s1, 'np_s2': np_s2,
            })

    n = len(states)
    return {
        'n_states':       n,
        'n_agree':        n_agree,
        'agreement_rate': n_agree / n if n > 0 else None,
        'disagreements':  disagreements,
    }


# ─────────────────────────────────────────────────────────────────────────── #
# Disagreement analysis                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

def disagreement_tally(disagreements: List[Dict]) -> Dict:
    """
    Summarise which ego/opp skills differ most often across disagreements.

    Returns dict with:
      ego_bf_skill  — skill name bf chose most often for ego
      ego_np_skill  — skill name nash-p chose most often for ego
      opp_bf_skill  — skill name bf chose most often for opp
      opp_np_skill  — skill name nash-p chose most often for opp
      ego_bf_counts — Counter {skill_name: count}
      ego_np_counts — Counter {skill_name: count}
      opp_bf_counts — Counter {skill_name: count}
      opp_np_counts — Counter {skill_name: count}
    """
    if not disagreements:
        return {}

    ego_bf = Counter(SKILL_NAMES[d['bf_s1']] for d in disagreements)
    ego_np = Counter(SKILL_NAMES[d['np_s1']] for d in disagreements)
    opp_bf = Counter(SKILL_NAMES[d['bf_s2']] for d in disagreements)
    opp_np = Counter(SKILL_NAMES[d['np_s2']] for d in disagreements)

    return {
        'ego_bf_skill':  ego_bf.most_common(1)[0][0],
        'ego_np_skill':  ego_np.most_common(1)[0][0],
        'opp_bf_skill':  opp_bf.most_common(1)[0][0],
        'opp_np_skill':  opp_np.most_common(1)[0][0],
        'ego_bf_counts': dict(ego_bf),
        'ego_np_counts': dict(ego_np),
        'opp_bf_counts': dict(opp_bf),
        'opp_np_counts': dict(opp_np),
    }


# ─────────────────────────────────────────────────────────────────────────── #
# Main: load model + rally states, run comparison, print report               #
# ─────────────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(
        description="Brute-force joint skill search vs. nash-p approximation"
    )
    parser.add_argument('--n-states', type=int, default=200,
                        help='Number of states to sample from rally data (default 200)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    import pickle
    import json
    import os

    MODEL_P_PATH  = "models/model_p_5skill.pth"
    RALLY_PATH    = "data/rallies_5skill.pkl"
    OUTPUT_JSON   = "skill_eval/brute_force_compare.json"

    # Load potential model
    sys.path.insert(0, '.')
    from model_arch import SimpleModel

    model = SimpleModel(116, [64, 32, 16], 1, last_layer_activation=None)
    model.load_state_dict(torch.load(MODEL_P_PATH, weights_only=True))
    model.eval()
    print(f"Loaded potential model: {MODEL_P_PATH}")

    # Load rally states
    with open(RALLY_PATH, 'rb') as f:
        rallies = pickle.load(f)

    rng = np.random.default_rng(args.seed)
    all_states = []
    for r in rallies:
        all_states.extend(r['states'])
    all_states = [np.array(s, dtype=np.float32) for s in all_states]

    n_sample = min(args.n_states, len(all_states))
    chosen_idx = rng.choice(len(all_states), size=n_sample, replace=False)
    states = [all_states[i] for i in chosen_idx]
    print(f"Sampled {n_sample} states from {len(all_states)} total rally states\n")

    # Run comparison
    result = compare_on_states(states, model)
    tally  = disagreement_tally(result['disagreements'])

    # ── Report ────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  BRUTE-FORCE vs. NASH-P COMPARISON")
    print("=" * 60)
    print(f"  States evaluated : {result['n_states']}")
    print(f"  Full agreement   : {result['n_agree']} / {result['n_states']} "
          f"({result['agreement_rate']:.1%})")
    print(f"  Disagreements    : {len(result['disagreements'])}")
    print()

    if tally:
        print("  ── When they disagree (ego / s1) ──")
        print(f"    Brute-force prefers : {tally['ego_bf_skill']}")
        print(f"    Nash-p prefers      : {tally['ego_np_skill']}")
        print(f"    BF counts  : {tally['ego_bf_counts']}")
        print(f"    Nash counts: {tally['ego_np_counts']}")
        print()
        print("  ── When they disagree (opp / s2) ──")
        print(f"    Brute-force prefers : {tally['opp_bf_skill']}")
        print(f"    Nash-p prefers      : {tally['opp_np_skill']}")
        print(f"    BF counts  : {tally['opp_bf_counts']}")
        print(f"    Nash counts: {tally['opp_np_counts']}")
        print()

    # Agreement rate interpretation
    rate = result['agreement_rate'] or 0.0
    if rate >= 0.90:
        verdict = "EXCELLENT — approximation matches brute-force >90% of the time."
    elif rate >= 0.75:
        verdict = "GOOD — approximation mostly agrees; acceptable for a course project."
    elif rate >= 0.55:
        verdict = "MODERATE — notable divergence; discuss limitations in report."
    else:
        verdict = "POOR — approximation disagrees frequently; needs improvement."

    print(f"  Agreement rate: {rate:.1%}")
    print(f"  Verdict: {verdict}")
    print()

    # ── Joint potential surface for a sample state ────────────────────────
    sample_state = states[0]
    scores = joint_potential(sample_state, model)
    print("  ── Joint potential surface (sample state #0) ──")
    header = "      " + "  ".join(f"{n:>12}" for n in SKILL_NAMES)
    print(header)
    for i, s1_name in enumerate(SKILL_NAMES):
        row = "  ".join(f"{scores[i, j]:12.4f}" for j in range(N_SKILLS))
        print(f"  {s1_name:<12} {row}")
    print()

    # Save JSON
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    out = {
        'n_states': result['n_states'],
        'n_agree':  result['n_agree'],
        'agreement_rate': result['agreement_rate'],
        'n_disagreements': len(result['disagreements']),
        'tally': tally,
        'verdict': verdict,
        'sample_joint_surface': scores.tolist(),
    }
    with open(OUTPUT_JSON, 'w') as f:
        import json
        json.dump(out, f, indent=2)
    print(f"  Results saved to: {OUTPUT_JSON}")


if __name__ == '__main__':
    main()
