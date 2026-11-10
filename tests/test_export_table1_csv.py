"""Tests for scripts/export_table1_csv.py on synthetic Table 1 JSONs (no MuJoCo)."""
import csv
import importlib.util
import json
import os

import pytest

_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts", "export_table1_csv.py")
_spec = importlib.util.spec_from_file_location("export_table1_csv", _PATH)
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)


def _write_run(path, method, model_dir, static_win, random_win):
    cells = {f"{a}/{d}": {"win": static_win, "decided": 200, "truncated": 1, "ci": [0.0, 1.0]}
             for a in ex.SKILL_NAMES for d in ex.DEFEND_POSITION_NAMES}
    cells["random"] = {"win": random_win, "decided": 200, "truncated": 0, "ci": [0.4, 0.6]}
    summary = {"cells": cells, "static_mean": static_win, "static_mean_ci": [static_win - 0.01, static_win + 0.01],
               "static_min": static_win, "static_min_opponent": "left/left"}
    rows = [{"opponent": "random", "rally_lengths": [100, 300, 500]}]
    with open(path, "w") as f:
        json.dump({"args": {"method": method, "model_dir": model_dir}, "summary": summary, "rows": rows}, f)


def _read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_labels_and_seeds_come_from_the_run_arguments():
    assert ex.method_label("nash-p-pga") == "alpha-potential (PGA, Eq. 18)"
    assert ex.method_label("selfplay-argmax:runs/selfplay/prep_seed1.pth") == "self-play (argmax)"
    assert ex.method_label("selfplay:runs/selfplay/prep_seed0_iter100.pth") == "self-play (sampled, iter 100)"
    assert ex.run_seed("nash-p-pga", "runs/phi/main_s2") == "2"
    assert ex.run_seed("selfplay:runs/selfplay/prep_seed1.pth", None) == "1"


def test_export_writes_runs_and_a_mean_over_seeds(tmp_path):
    _write_run(tmp_path / "a.json", "nash-p-pga", "runs/phi/main_s0", 0.6, 0.5)
    _write_run(tmp_path / "b.json", "nash-p-pga", "runs/phi/main_s1", 0.8, 0.7)
    ex.export(sorted(str(p) for p in tmp_path.glob("*.json")), str(tmp_path))
    summary = _read(tmp_path / "table1_summary.csv")
    mean_row = [r for r in summary if r["seed"].startswith("mean")][0]
    assert float(mean_row["static_mean"]) == pytest.approx(0.7)
    assert float(mean_row["random"]) == pytest.approx(0.6)
    assert mean_row["static_mean_range"] == "0.600-0.800"
    matrix = _read(tmp_path / "table1_static_matrix.csv")
    assert len(matrix) == 15 * 3
    assert {r["seed"] for r in matrix} == {"0", "1", "mean"}
