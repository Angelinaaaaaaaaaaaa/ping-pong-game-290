"""
Re-encode an existing rally pickle into a new state encoding.

The current rally format (from collect_data.py) stores:
  - states  : list[np.ndarray (76,)]   ← encoded with old encoder
  - raw_obs : list[np.ndarray (116,)]  ← full env obs (kept for re-encoding)
  - winner, skill1, skill2

This script reads an old pickle, recomputes `states` from `raw_obs` using a
new encoder (default: encode_ego_gantry), and writes a new pickle.

The training pipeline (train_q_model_5skill_v3.py) auto-detects state_dim
from the first state vector, so no other changes are required.

Run:
    PYTHONPATH=. python nash_skills/v2/reencode_rallies.py \\
        --input data/rallies_v2_1000.pkl \\
        --output data/rallies_v2_1000_gantry.pkl
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import pickle as pkl

import numpy as np

from nash_skills.v2.state_encoder_gantry import encode_ego_gantry, STATE_DIM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input rally pickle path")
    ap.add_argument("--output", required=True, help="Output rally pickle path")
    args = ap.parse_args()

    print(f"Loading {args.input} ...")
    with open(args.input, "rb") as f:
        rallies = pkl.load(f)
    print(f"  Loaded {len(rallies)} rallies")

    if not rallies:
        raise ValueError("Empty rally list — nothing to re-encode")

    sample = rallies[0]
    if "raw_obs" not in sample:
        raise KeyError(
            "Rally entry missing 'raw_obs'. The new encoder needs raw obs to "
            "re-encode. Was this pickle collected by an older version of "
            "collect_data.py that didn't save raw_obs?"
        )
    print(f"  raw_obs shape: {np.asarray(sample['raw_obs'][0]).shape}")
    print(f"  old states shape: {np.asarray(sample['states'][0]).shape}")
    print(f"  → new STATE_DIM = {STATE_DIM}")

    new_rallies = []
    total_states = 0
    for r in rallies:
        new_states = [encode_ego_gantry(np.asarray(o, dtype=np.float32))
                      for o in r["raw_obs"]]
        new_rallies.append({
            "skill1":  r["skill1"],
            "skill2":  r["skill2"],
            "states":  new_states,
            "raw_obs": r["raw_obs"],     # keep raw for further re-encoding later
            "winner":  r["winner"],
        })
        total_states += len(new_states)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".",
                exist_ok=True)
    print(f"\nSaving {args.output} ...")
    with open(args.output, "wb") as f:
        pkl.dump(new_rallies, f)

    print(f"  Wrote {len(new_rallies)} rallies, {total_states} total states")
    print(f"  New state vector dim: {len(new_rallies[0]['states'][0])}")
    print(f"\nDone. Now train with the new pickle:")
    print(f"  python nash_skills/v2/train_q_model_5skill_v3.py --rallies {args.output}")


if __name__ == "__main__":
    main()
