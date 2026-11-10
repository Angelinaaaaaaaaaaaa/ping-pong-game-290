"""Tests for scripts/export_paper_tables.py (no MuJoCo)."""
import importlib.util
import os

import pytest

_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts", "export_paper_tables.py")
_spec = importlib.util.spec_from_file_location("export_paper_tables", _PATH)
ep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ep)


def test_mean_range_formats_mean_and_spread():
    assert ep.mean_range([0.6, 0.8, 0.7]) == "0.700 (0.600-0.800)"
    assert ep.mean_range([0.5]) == "0.500"
    assert ep.mean_range([]) == ""


def test_other_proposed_for_alpha_is_one_minus_self_play_vs_alpha():
    sp_vs_alpha = [0.455, 0.495, 0.600]
    assert [round(x, 3) for x in ep.complement(sp_vs_alpha)] == [0.545, 0.505, 0.4]


def test_runs_are_grouped_by_method_label():
    rows = [{"method": "self-play (sampled)", "seed": "0", "static_mean": "0.6"},
            {"method": "self-play (sampled)", "seed": "mean (3 seeds)", "static_mean": "0.6"},
            {"method": "self-play (sampled, iter 100)", "seed": "0", "static_mean": "0.6"}]
    groups = ep.per_seed_runs(rows)
    assert len(groups["self-play (sampled)"]) == 1
    assert "self-play (sampled, iter 100)" in groups
