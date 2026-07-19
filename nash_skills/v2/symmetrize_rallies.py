"""
Symmetrize a rally pickle by augmenting each entry with its ego/opp flipped twin.

Motivation
----------
diag_q_models.py revealed a strong ego/opp asymmetry in the collected data:
ego wins ~79% of completed rallies, opp wins ~21%. The Q model learned this
asymmetry rather than the underlying skill matchups, producing "always pick X"
column-biased argmax patterns.

Since the physics is actually symmetric (ego and opp use the same frozen PPO
controller — only their perspective into the obs differs), we can produce
synthetic "opp-perspective" training samples from each existing rally by:

  - Swapping skill1 ↔ skill2
  - Re-encoding states using encode_opp_gantry on the SAME raw_obs
    (this gives the opp-side view of the same physical trajectory)
  - Flipping winner:  1 → 2 (ego won → opp won),  2 → 1,  0 → 0 (truncated)

The output pickle is 2× the size of the input. Training on it should produce
a Q model with balanced ego/opp predictions and (hopefully) genuine
skill-conditional argmax patterns rather than always preferring one skill.

Run:
    PYTHONPATH=. python nash_skills/v2/symmetrize_rallies.py \\
        --input data/rallies_v2_1000_gantry.pkl \\
        --output data/rallies_v2_1000_gantry_sym.pkl
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import pickle as pkl

import numpy as np

from nash_skills.v2.state_encoder_gantry import encode_opp_gantry


def flip_winner(w):
    if w == 1: return 2
    if w == 2: return 1
    return 0   # truncated stays truncated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input rally pickle path")
    ap.add_argument("--output", required=True, help="Output symmetrized pickle path")
    ap.add_argument("--mode", choices=["append", "replace_random"], default="append",
                    help="append = keep original + add flipped (2x size); "
                         "replace_random = 50/50 random flip per rally (1x size, faster)")
    args = ap.parse_args()

    print(f"Loading {args.input} ...")
    with open(args.input, "rb") as f:
        rallies = pkl.load(f)
    print(f"  Loaded {len(rallies)} rallies")

    if not rallies:
        raise ValueError("Empty rally list")
    if "raw_obs" not in rallies[0]:
        raise KeyError("Rally entries missing 'raw_obs' — cannot re-encode from opp perspective")

    # Count original winner distribution
    orig_w = {1: 0, 2: 0, 0: 0}
    for r in rallies:
        orig_w[r["winner"]] = orig_w.get(r["winner"], 0) + 1
    total = len(rallies)
    print(f"  Original winners: ego={orig_w[1]} ({orig_w[1]/total:.1%})  "
          f"opp={orig_w[2]} ({orig_w[2]/total:.1%})  "
          f"trunc={orig_w[0]} ({orig_w[0]/total:.1%})")

    new_rallies = []
    rng = np.random.default_rng(0)

    for r in rallies:
        # Original entry (keep as-is in append mode; random keep/flip in replace_random)
        if args.mode == "append":
            new_rallies.append(r)
            include_flip = True
        else:  # replace_random
            include_flip = rng.random() < 0.5
            if not include_flip:
                new_rallies.append(r)
                continue

        # Build flipped twin: opp perspective on same raw_obs
        flipped_states = [
            encode_opp_gantry(np.asarray(o, dtype=np.float32))
            for o in r["raw_obs"]
        ]
        new_rallies.append({
            "skill1":  r["skill2"],         # swap
            "skill2":  r["skill1"],
            "states":  flipped_states,
            "raw_obs": r["raw_obs"],         # raw obs is physical, no change
            "winner":  flip_winner(r["winner"]),
        })

    # Report new winner distribution
    new_w = {1: 0, 2: 0, 0: 0}
    for r in new_rallies:
        new_w[r["winner"]] = new_w.get(r["winner"], 0) + 1
    new_total = len(new_rallies)
    print(f"\nAfter symmetrize ({args.mode}):")
    print(f"  Total rallies: {new_total} ({new_total / total:.1f}x original)")
    print(f"  New winners: ego={new_w[1]} ({new_w[1]/new_total:.1%})  "
          f"opp={new_w[2]} ({new_w[2]/new_total:.1%})  "
          f"trunc={new_w[0]} ({new_w[0]/new_total:.1%})")
    print(f"  → ego/opp imbalance should be ~50/50 now")

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".",
                exist_ok=True)
    print(f"\nSaving {args.output} ...")
    with open(args.output, "wb") as f:
        pkl.dump(new_rallies, f)
    print("Done.")

    print(f"\nNext: train + diag")
    print(f"  PYTHONPATH=. python nash_skills/v2/train_q_model_5skill_v3.py \\")
    print(f"      --rallies {args.output} --output-suffix _gantry_sym")
    print(f"  python nash_skills/v2/diag_q_fixed_opp.py \\")
    print(f"      --model1 models/model1_5skill_v3_gantry_sym.pth \\")
    print(f"      --rallies {args.output}")


if __name__ == "__main__":
    main()
