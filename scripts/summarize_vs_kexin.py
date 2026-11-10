"""
Route A tables: self-play (sampled / argmax) against Kexin's V3 alpha-potential rules inside our Design B game
(scripts/launch_vs_kexin_v3_ours.sh, runs_v3/vs_kexin), and her rules on their own against our 15 static
opponents and random. Win rates count finished rallies, as in eval_table1_attack_defend.py; unfinished rallies
are counted separately.

    python scripts/summarize_vs_kexin.py analysis/server_results/paper_runs/vs_kexin
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nash_skills.defend_positions import DEFEND_POSITION_NAMES
from nash_skills.skills import SKILL_NAMES

RULES = ("potential_ibr", "potential_br", "potential_adaptive_match", "potential_minimax")
RULE_LABELS = {"potential_ibr": "Potential IBR", "potential_br": "Potential BR",
               "potential_adaptive_match": "Potential Adaptive Match", "potential_minimax": "Potential Minimax"}
SELFPLAY = (("selfplay-nc", "Self-play (sampled)"), ("selfplay-nc-argmax", "Self-play (argmax)"))
STATIC = [(a, d) for a in SKILL_NAMES for d in DEFEND_POSITION_NAMES]
KIND = "kexin-v3"
SEEDS = (0, 1, 2)


def kexin_rule(label):
    """'kexin-v3:<rule>:<model path>' -> rule; None for other opponents."""
    parts = label.split(":")
    return parts[1] if len(parts) >= 3 and parts[0] == KIND else None


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _stats(cells):
    rates = [c["win"] for c in cells]
    return {"mean": float(np.mean(rates)), "std": float(np.std(rates, ddof=1)) if len(rates) > 1 else float("nan"),
            "wins": sum(c["wins"] for c in cells), "decided": sum(c["decided"] for c in cells),
            "truncated": sum(c["truncated"] for c in cells), "seeds": len(cells)}


def head_to_head(root):
    """{self-play kind: {rule: win-rate stats over seeds}} from <kind>_s<seed>.json."""
    out = {}
    for kind, _label in SELFPLAY:
        per_rule = {}
        for seed in SEEDS:
            result = _load(os.path.join(root, f"{kind}_s{seed}.json"))
            for label, cell in (result or {"summary": {"cells": {}}})["summary"]["cells"].items():
                rule = kexin_rule(label)
                if rule is not None:
                    per_rule[rule] = per_rule.get(rule, []) + [cell]
        out[kind] = {rule: _stats(cells) for rule, cells in per_rule.items()}
    return out


def her_rules_in_our_game(root):
    """{rule: static mean / min / random} from kexin_<rule>_static.json."""
    out = {}
    for rule in RULES:
        result = _load(os.path.join(root, f"kexin_{rule}_static.json"))
        if not result:
            continue
        cells = result["summary"]["cells"]
        static = [(f"{a}/{d}", cells[f"{a}/{d}"]["win"]) for a, d in STATIC if f"{a}/{d}" in cells]
        low_label, low = min(static, key=lambda kv: kv[1])
        unfinished = sum(c["truncated"] for c in cells.values())
        out[rule] = {"static_mean": float(np.mean([w for _l, w in static])), "static_min": low,
                     "static_min_opponent": low_label, "random": cells.get("random", {}).get("win", float("nan")),
                     "unfinished_share": unfinished / max(1, unfinished + sum(c["decided"] for c in cells.values()))}
    return out


def _pct(x):
    return "—" if x != x else f"{100 * x:.1f}"


def _cell_text(stats):
    spread = "" if stats["std"] != stats["std"] else f" ± {100 * stats['std']:.1f}"
    return f"{_pct(stats['mean'])}{spread}"


def markdown(root):
    h2h = head_to_head(root)
    lines = ["Self-play win rate (%) against Kexin's V3 α-potential player, inside our game "
             "(mean ± std over self-play seeds; finished rallies)", "",
             "| Self-play | " + " | ".join(RULE_LABELS[r] for r in RULES) + " |",
             "|---|" + "---|" * len(RULES)]
    for kind, label in SELFPLAY:
        row = h2h.get(kind, {})
        lines.append(f"| {label} | " + " | ".join(_cell_text(row[r]) if r in row else "—" for r in RULES) + " |")
    pooled = [s for row in h2h.values() for s in row.values()]
    if pooled:
        lines += ["", f"Rallies per cell: {pooled[0]['decided']} finished over {pooled[0]['seeds']} seeds; "
                      f"unfinished rallies overall: {sum(s['truncated'] for s in pooled)}."]
    mine = her_rules_in_our_game(root)
    if mine:
        lines += ["", "Kexin's rules on their own inside our game (win rate %, finished rallies)", "",
                  "| Rule | Static mean | Static min (opponent) | Random |", "|---|---|---|---|"]
        for rule in RULES:
            if rule in mine:
                m = mine[rule]
                lines.append(f"| {RULE_LABELS[rule]} | {_pct(m['static_mean'])} | {_pct(m['static_min'])} "
                             f"({m['static_min_opponent']}) | {_pct(m['random'])} |")
    return "\n".join(lines)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/summarize_vs_kexin.py <vs_kexin results dir>")
    print(markdown(sys.argv[1]))


if __name__ == "__main__":
    main()
