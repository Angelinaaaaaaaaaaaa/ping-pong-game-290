"""scripts/summarize_vs_kexin.py: Route A tables (self-play vs Kexin's V3 rules inside our game)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import pytest

import summarize_vs_kexin as sk

PHI = "models_kexin/v3/potential_50k_aggressive.pth"


def _cell(wins, decided, truncated=0):
    return {"wins": wins, "decided": decided, "truncated": truncated, "win": wins / decided}


def _write(root, name, cells):
    with open(os.path.join(root, f"{name}.json"), "w") as f:
        json.dump({"summary": {"cells": cells}}, f)


def test_rule_names_come_from_the_opponent_spec():
    assert sk.kexin_rule(f"kexin-v3:potential_minimax:{PHI}") == "potential_minimax"
    assert sk.kexin_rule("left/center") is None


def test_head_to_head_averages_seeds_and_pools_rallies(tmp_path):
    for s, wins in enumerate((150, 160, 170)):
        _write(tmp_path, f"selfplay-nc_s{s}", {f"kexin-v3:potential_minimax:{PHI}": _cell(wins, 200, truncated=s)})
    table = sk.head_to_head(str(tmp_path))
    row = table["selfplay-nc"]["potential_minimax"]
    assert row["mean"] == pytest.approx(0.8) and row["std"] == pytest.approx(0.05)
    assert row["wins"] == 480 and row["decided"] == 600 and row["truncated"] == 3 and row["seeds"] == 3


def test_her_rules_in_our_game_report_static_mean_min_and_random(tmp_path):
    static = {f"{a}/{d}": _cell(100 + i, 200) for i, (a, d) in enumerate(sk.STATIC)}
    _write(tmp_path, "kexin_potential_ibr_static", {**static, "random": _cell(120, 200)})
    row = sk.her_rules_in_our_game(str(tmp_path))["potential_ibr"]
    assert row["static_min"] == pytest.approx(0.5) and row["static_mean"] == pytest.approx(0.535)
    assert row["static_min_opponent"] == "left/left" and row["random"] == pytest.approx(0.6)


def test_markdown_lists_every_rule_and_both_self_play_variants(tmp_path):
    for kind in ("selfplay-nc", "selfplay-nc-argmax"):
        _write(tmp_path, f"{kind}_s0", {f"kexin-v3:{r}:{PHI}": _cell(120, 200) for r in sk.RULES})
    text = sk.markdown(str(tmp_path))
    assert "Self-play (sampled)" in text and "Self-play (argmax)" in text
    assert all(sk.RULE_LABELS[r] in text for r in sk.RULES) and "60.0" in text
