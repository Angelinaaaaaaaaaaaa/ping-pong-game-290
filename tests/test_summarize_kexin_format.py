"""scripts/summarize_kexin_format.py: per-rule rows averaged over seeds."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import pytest

import summarize_kexin_format as sk


REAL_STATIC = ["left/left", "left/right", "left/center", "right/left", "right/right"]


def _result(static_wins, random=0.6, ibr=0.5, selfplay=0.4):
    cells = {REAL_STATIC[i]: {"win": w} for i, w in enumerate(static_wins)}
    cells.update({"random": {"win": random}, "ibr": {"win": ibr},
                  "selfplay:runs/sp.pth": {"win": selfplay}})
    return {"summary": {"cells": cells}}


def test_row_stats_reads_every_column():
    s = sk.row_stats(_result([0.5, 0.9, 0.7]))
    assert s["static_mean"] == pytest.approx(0.7) and s["static_min"] == pytest.approx(0.5)
    assert s["random"] == 0.6 and s["ibr"] == 0.5 and s["other"] == 0.4


def test_table_averages_seeds_and_labels_rules(tmp_path):
    d = tmp_path / "kexin_format" / "aggressive" / "base"
    d.mkdir(parents=True)
    for s, w in enumerate((0.5, 0.7, 0.9)):
        (d / f"nash-q_s{s}.json").write_text(json.dumps(_result([w, w])))
    text = sk.table(str(tmp_path), "aggressive", "base")
    assert "Nash-Q" in text and "70.0%" in text
    assert "Potential IBR" not in text  # missing rules are skipped


def test_base_pga_row_comes_from_alpha_fixed(tmp_path):
    d = tmp_path / "alpha_fixed" / "aggressive"
    d.mkdir(parents=True)
    for s in range(3):
        (d / f"table1_s{s}.json").write_text(json.dumps(_result([0.8, 0.6])))
        (d / f"h2h_s{s}.json").write_text(json.dumps(_result([], selfplay=0.3)))
    rows = sk.collect(str(tmp_path), "aggressive", "base", "nash-p-pga")
    assert len(rows) == 3 and rows[0]["other"] == 0.3 and rows[0]["static_min"] == 0.6


def test_a_file_without_static_cells_gives_nan_not_an_error():
    s = sk.row_stats(_result([], selfplay=0.3))
    assert s["other"] == 0.3 and s["static_mean"] != s["static_mean"]
