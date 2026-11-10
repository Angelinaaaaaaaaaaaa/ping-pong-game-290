"""scripts/summarize_alpha_ablations.py groups runs by ablation value and averages over seeds."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import pytest

import summarize_alpha_ablations as sa


def _run(root, profile, name, static_mean, n_train=100):
    d = root / profile / "phi" / name
    d.mkdir(parents=True)
    (d / "heldout_attack_defend.json").write_text(json.dumps({
        "heldout_q1_mse": 0.5, "n_train_rallies": n_train,
        "heldout_residuals": {"p95_abs": 0.7, "median_abs": 0.2, "mean_abs": 0.3},
        "heldout_residuals_constant_phi": {"mean_abs": 0.4}}))
    t = root / profile / "table1"
    t.mkdir(parents=True, exist_ok=True)
    (t / f"{name}.json").write_text(json.dumps({"summary": {
        "static_mean": static_mean, "static_min": 0.5,
        "cells": {"random": {"win": 0.6, "truncated": 1, "decided": 9}}}}))


def test_groups_by_value_and_averages_seeds(tmp_path):
    for s, v in enumerate((0.6, 0.8)):
        _run(tmp_path, "aggressive", f"gamma0.7_s{s}", v)
    _run(tmp_path, "aggressive", "gamma0.99_s0", 0.7)
    groups = sa.group(str(tmp_path), "aggressive", "gamma")
    assert list(groups) == [0.7, 0.99]
    _text, rows = sa.table(groups, "γ")
    stats = dict(rows)[0.7]
    assert stats["static_mean"][0] == pytest.approx(0.7)
    assert stats["static_mean"][2] == 2
    assert stats["trunc"][0] == pytest.approx(0.1)


def test_volume_runs_are_labelled_by_their_training_size(tmp_path):
    _run(tmp_path, "aggressive", "vol340_s0", 0.6, n_train=340)
    text, _rows = sa.table(sa.group(str(tmp_path), "aggressive", "vol"), "train rallies")
    assert "| 340 |" in text
