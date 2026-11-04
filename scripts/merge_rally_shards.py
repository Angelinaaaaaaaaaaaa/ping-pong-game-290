"""
Merge collect_data_attack_defend.py shard pickles into one rally file, after
checking that every shard used the same encoder and defend mechanism.

    python scripts/merge_rally_shards.py --output runs/rallies_prep_grid.pkl runs/data_shards/grid_*.pkl
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import collections
import pickle

import numpy as np


def merge(paths):
    rallies = []
    for path in sorted(paths):
        with open(path, "rb") as f:
            rallies += pickle.load(f)
    tags = {(r.get("encoder"), r.get("defend_mechanism")) for r in rallies}
    if len(tags) != 1:
        raise ValueError(f"Shards disagree on (encoder, defend_mechanism): {tags}")
    return rallies


def describe(rallies):
    decisions = sum(len(r["states"]) for r in rallies)
    profiles = collections.Counter(
        (r["ego_attack_choices"][0], r["ego_defend_choices"][0], r["opp_attack_choices"][0], r["opp_defend_choices"][0])
        for r in rallies)
    return (f"{len(rallies)} rallies, {decisions} decisions ({decisions / len(rallies):.1f}/rally), "
            f"median {np.median([r['steps'] for r in rallies]):.0f} steps, "
            f"P1 wins {np.mean([r['winner'] == 1 for r in rallies]):.3f}, "
            f"first-decision profiles {len(profiles)}/225")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("shards", nargs="+")
    args = ap.parse_args()
    rallies = merge(args.shards)
    with open(args.output, "wb") as f:
        pickle.dump(rallies, f)
    print(f"{args.output}: {describe(rallies)}")


if __name__ == "__main__":
    main()
