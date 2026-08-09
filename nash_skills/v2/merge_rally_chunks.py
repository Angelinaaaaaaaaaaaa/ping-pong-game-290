"""
Merge parallel 5-skill v2 collection chunks into one rally pickle.

Example:
    python nash_skills/v2/merge_rally_chunks.py \
        data/chunks/rallies_5skill_v2_5000_shard*.pkl \
        --output data/rallies_5skill_v2_5000.pkl
"""

import argparse
import os
import pickle as pkl
from collections import Counter

from nash_skills.v2.labeling import check_balance, summarise_balance


def merge(paths, output_path):
    merged = []
    for path in paths:
        with open(path, "rb") as f:
            chunk = pkl.load(f)
        print(f"{path}: {len(chunk)} rallies")
        merged.extend(chunk)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pkl.dump(merged, f)

    print(f"\nSaved {len(merged)} rallies to {output_path}")

    winners = Counter(entry.get("winner", "MISSING") for entry in merged)
    print(f"Winner counts: {dict(sorted(winners.items(), key=lambda kv: str(kv[0])))}")

    counts = summarise_balance(merged)
    is_ok, ratio = check_balance(merged, threshold=5.0)
    print(f"Balance check: max/min ratio = {ratio:.2f} ({'OK' if is_ok else 'IMBALANCED'})")
    print("Per-pair counts:")
    for (skill1, skill2), count in sorted(counts.items()):
        print(f"  {skill1:12s} vs {skill2:12s}: {count}")

    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge v2 rally collection chunks.")
    parser.add_argument("inputs", nargs="+", help="Input chunk pickle paths")
    parser.add_argument("--output", required=True, help="Merged output pickle path")
    args = parser.parse_args()

    merge(args.inputs, args.output)
