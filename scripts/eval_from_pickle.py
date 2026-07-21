"""
Read a pickle of rallies and write CSV/JSON summaries compatible with
`eval_matchup_2skill.py` output.

Usage (from project root):
    python scripts/eval_from_pickle.py
    python scripts/eval_from_pickle.py --pickle data/rallies_nash.pkl \
        --output-csv skill_eval/from_pickle.csv --output-json skill_eval/from_pickle.json

The script makes conservative inferences:
- If the pickle is a dict, each key -> a dataset (one CSV row per key).
- If the pickle is a list, it's treated as a single dataset named by the pickle file.
- Winner is inferred from the final observation's ball x position (index 36) using
  `table_shift` (default 1.5) when explicit winner labels are not available.
- Contact/success counts and skill usage are set to 0 if not present in the pickle.
"""

import argparse
import csv
import json
import os
import pickle
from typing import Any

import numpy as np


def infer_winner_from_obs(last_obs, table_shift=1.5):
    try:
        ball_x = float(last_obs[36])
        return "ego" if ball_x > table_shift else "opp"
    except Exception:
        return None


def summarize_rallies(name: str, rallies: list, table_shift=1.5):
    episodes = len(rallies)
    rally_lengths = [len(r) for r in rallies]
    episode_steps = rally_lengths.copy()
    total_steps = sum(rally_lengths)
    truncated_episodes = 0

    ego_wins = 0
    opp_wins = 0

    # best-effort: try to detect stored per-rally metadata if present
    ego_contacts = 0
    opp_contacts = 0
    ego_successes = 0
    opp_successes = 0

    skill_usage = {"left": 0, "right": 0}

    for r in rallies:
        winner = None
        # r may be a list of raw obs arrays, or dicts with metadata
        if isinstance(r, dict):
            # common patterns: r['winner'], r.get('obs') etc.
            if "winner" in r:
                val = r["winner"]
                if val in ("ego", 0, "player0", "p0", "left"):
                    winner = "ego"
                elif val in ("opp", 1, "player1", "p1", "right"):
                    winner = "opp"
            if winner is None:
                # try nested obs
                if "obs" in r and r["obs"]:
                    winner = infer_winner_from_obs(r["obs"][-1], table_shift=table_shift)
            # contacts/successes
            ego_contacts += int(r.get("ego_contacts", 0))
            opp_contacts += int(r.get("opp_contacts", 0))
            ego_successes += int(r.get("ego_successes", 0))
            opp_successes += int(r.get("opp_successes", 0))
            su = r.get("skill_usage") or {}
            skill_usage["left"] += int(su.get("left", 0))
            skill_usage["right"] += int(su.get("right", 0))
        else:
            # assume list/array of obs frames
            if len(r) > 0:
                last = r[-1]
                winner = infer_winner_from_obs(last, table_shift=table_shift)

        if winner == "ego":
            ego_wins += 1
        elif winner == "opp":
            opp_wins += 1

    done_episodes = episodes - truncated_episodes
    win_rate = (ego_wins / episodes) if episodes > 0 else None
    win_rate_clean = (ego_wins / done_episodes) if done_episodes > 0 else None
    avg_rally_length = float(np.mean(rally_lengths)) if rally_lengths else None
    avg_steps_per_episode = float(np.mean(episode_steps)) if episode_steps else None

    row = {
        "strategy1": name,
        "strategy2": "from_pickle",
        "episodes": episodes,
        "truncated_episodes": truncated_episodes,
        "done_episodes": done_episodes,
        "ego_wins": ego_wins,
        "opp_wins": opp_wins,
        "win_rate": round(win_rate, 4) if win_rate is not None else "",
        "win_rate_clean": round(win_rate_clean, 4) if win_rate_clean is not None else "",
        "total_steps": total_steps,
        "avg_steps_per_episode": round(avg_steps_per_episode, 2) if avg_steps_per_episode is not None else "",
        "ego_contacts": ego_contacts,
        "opp_contacts": opp_contacts,
        "ego_successes": ego_successes,
        "opp_successes": opp_successes,
        "ego_success_rate": round((ego_successes / ego_contacts), 4) if ego_contacts > 0 else "",
        "opp_success_rate": round((opp_successes / opp_contacts), 4) if opp_contacts > 0 else "",
        "avg_rally_length": round(avg_rally_length, 2) if avg_rally_length is not None else "",
        "usage_left": skill_usage.get("left", 0),
        "usage_right": skill_usage.get("right", 0),
        "rally_lengths": rally_lengths,
        "episode_steps": episode_steps,
        "skill_usage": skill_usage,
    }

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle", default="data/rallies_nash.pkl")
    parser.add_argument("--output-csv", default="skill_eval/from_pickle.csv")
    parser.add_argument("--output-json", default="skill_eval/from_pickle.json")
    parser.add_argument("--table-shift", type=float, default=1.5)
    args = parser.parse_args()

    with open(args.pickle, "rb") as f:
        data = pickle.load(f)

    datasets = []

    if isinstance(data, dict):
        for k, v in data.items():
            datasets.append((str(k), v))
    elif isinstance(data, list):
        # If it's a list of datasets (list of lists), try to detect naming
        # Otherwise treat as a single dataset
        # Heuristic: if top-level elements are tuples (name, rallies)
        if data and isinstance(data[0], tuple) and len(data[0]) == 2 and isinstance(data[0][1], list):
            for k, v in data:
                datasets.append((str(k), v))
        else:
            base = os.path.splitext(os.path.basename(args.pickle))[0]
            datasets.append((base, data))
    else:
        raise RuntimeError("Unsupported pickle top-level type: %s" % type(data))

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    rows = []
    json_rows = []
    for name, rallies in datasets:
        row = summarize_rallies(name, rallies, table_shift=args.table_shift)
        rows.append(row)
        # JSON-friendly copy with numeric fields
        jr = {k: (v if not (isinstance(v, np.generic)) else v.item()) for k, v in row.items()}
        json_rows.append(jr)

    # Write CSV with selected columns (matching eval_matchup_2skill.csv layout)
    fieldnames = [
        "strategy1", "strategy2", "episodes", "truncated_episodes", "done_episodes",
        "ego_wins", "opp_wins", "win_rate", "win_rate_clean", "total_steps",
        "avg_steps_per_episode", "ego_contacts", "opp_contacts", "ego_successes", "opp_successes",
        "ego_success_rate", "opp_success_rate", "avg_rally_length", "usage_left", "usage_right",
    ]

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    with open(args.output_json, "w") as f:
        json.dump(json_rows, f, indent=2)

    print(f"Wrote CSV: {args.output_csv}")
    print(f"Wrote JSON: {args.output_json}")


if __name__ == "__main__":
    main()
