"""
Prepare relabeled rally datasets for a future long-stable-rally labeling
ablation (meeting note item 4) -- pure Python, no MuJoCo.

IMPORTANT: this script only writes PREPARED DATASETS for later retraining.
It does not train or evaluate anything. Compare the resulting models on
both predictive performance and downstream game results once you actually
retrain on the outputs.

Three labeling strategies for truncated (undecided) rallies:

    discard     Remove truncated rallies entirely (today's collector
                behavior already does this at collection time; this mode
                lets you re-apply it to a pkl that still has them).
    tie0        Keep truncated rallies, relabel winner=0 explicitly (tie,
                reward 0 for both players). Caution: many zero labels can
                flatten the learned Q-value / potential-function targets.
    asym_small  Give a small positive reward to whichever player initiated
                the rally and a small negative reward to the other player.
                Requires a reliable 'initiator' field on each rally dict
                (1 = player 1 initiated, 2 = player 2). If that field is
                missing, this mode fails with MissingInitiatorFieldError
                instead of guessing who served.

Usage
-----
    # Dry run: only print counts and proposed output paths
    venv/bin/python nash_skills/v2/labeling_ablation.py \\
        --input data/rallies_5skill_v2.pkl --dry-run

    # Write all applicable modes
    venv/bin/python nash_skills/v2/labeling_ablation.py \\
        --input data/rallies_5skill_v2.pkl --modes discard tie0 asym_small
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import copy
import pickle
from pathlib import Path

__all__ = [
    "apply_discard",
    "apply_tie0",
    "apply_asym_small",
    "MissingInitiatorFieldError",
]

_DEFAULT_INPUT = "data/rallies_5skill_v2.pkl"
_ASYM_REWARD = 0.1
_ALL_MODES = ("discard", "tie0", "asym_small")


class MissingInitiatorFieldError(Exception):
    """Raised when asym_small is requested but no 'initiator' field exists."""


def _is_truncated(winner) -> bool:
    return winner not in (1, 2)


def apply_discard(rallies: list) -> list:
    """Remove truncated/undecided rallies. Does not mutate the input list."""
    return [r for r in rallies if not _is_truncated(r.get("winner"))]


def apply_tie0(rallies: list) -> list:
    """
    Keep every rally; truncated ones get winner explicitly set to 0 (tie).
    Decided rallies are copied through unchanged. Does not mutate the input.
    """
    result = []
    for r in rallies:
        r2 = copy.deepcopy(r)
        if _is_truncated(r2.get("winner")):
            r2["winner"] = 0
        result.append(r2)
    return result


def apply_asym_small(rallies: list, reward: float = _ASYM_REWARD) -> list:
    """
    Give the rally initiator a small positive reward and the other player a
    small negative reward for truncated rallies; decided rallies pass
    through unchanged (their winner field already carries a clear signal).

    Requires every rally dict to carry a reliable 'initiator' field
    (1 or 2). Raises MissingInitiatorFieldError if it's absent -- this mode
    must not guess who served.
    """
    if not rallies or "initiator" not in rallies[0]:
        raise MissingInitiatorFieldError(
            "asym_small requires a reliable 'initiator' field (1 or 2) on "
            "each rally dict, indicating which player served/initiated the "
            "rally. This field is not present in the input data. Re-collect "
            "with initiator tracking, or use --modes discard tie0 instead."
        )

    result = []
    for r in rallies:
        r2 = copy.deepcopy(r)
        if _is_truncated(r2.get("winner")):
            initiator = r2["initiator"]
            if initiator == 1:
                r2["reward1"] = reward
                r2["reward2"] = -reward
            else:
                r2["reward1"] = -reward
                r2["reward2"] = reward
        result.append(r2)
    return result


_MODE_FUNCS = {
    "discard": apply_discard,
    "tie0": apply_tie0,
    "asym_small": apply_asym_small,
}


def _output_path(input_path: str, mode: str) -> str:
    p = Path(input_path)
    return str(p.with_name(f"{p.stem}_label_{mode}{p.suffix}"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare relabeled rally datasets for a future labeling ablation (no MuJoCo)."
    )
    ap.add_argument("--input", default=_DEFAULT_INPUT,
                    help=f"Input rally pickle (default: {_DEFAULT_INPUT})")
    ap.add_argument("--modes", nargs="+", choices=_ALL_MODES, default=list(_ALL_MODES),
                    help="Labeling modes to generate (default: all)")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="Only print counts and proposed output paths; write nothing")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Rally file not found: {args.input}")

    with open(input_path, "rb") as f:
        rallies = pickle.load(f)
    print(f"Loaded {len(rallies)} rallies from {args.input}")
    print("NOTE: outputs are prepared datasets for future retraining, not evaluated results.\n")

    for mode in args.modes:
        out_path = _output_path(args.input, mode)
        if Path(out_path).resolve() == input_path.resolve():
            raise SystemExit(
                f"Refusing to overwrite the original input file: {out_path}"
            )

        try:
            result = _MODE_FUNCS[mode](rallies)
        except MissingInitiatorFieldError as e:
            print(f"[{mode}] SKIPPED: {e}")
            continue

        print(f"[{mode}] {len(rallies)} -> {len(result)} rallies")
        print(f"[{mode}] output: {out_path}")

        if args.dry_run:
            print(f"[{mode}] (dry-run, not written)")
            continue

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(result, f)
        print(f"[{mode}] saved.")

    if args.dry_run:
        print("\nDry run complete -- no files written.")


if __name__ == "__main__":
    main()
