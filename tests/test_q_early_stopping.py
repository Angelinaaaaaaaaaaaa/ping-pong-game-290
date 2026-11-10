"""Q-model early stopping on the validation split, and the constant-Phi / predict-zero baselines
in the held-out report (nash_skills/v2/train_q_model_5skill_v3_attack_defend.py)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from nash_skills.v2 import train_q_model_5skill_v3_attack_defend as tq
from tests.test_train_q_model_5skill_v3_attack_defend import TestOutDir


def _train(tmp_path, **kw):
    path = tmp_path / "r.pkl"
    TestOutDir()._write_rallies(path, n_rallies=20)
    tq.train(str(path), n_epochs=6, lr=1e-3, out_dir=str(tmp_path / "run"), **kw)
    return json.load(open(tmp_path / "run" / "heldout_attack_defend.json"))


def test_q_select_modes():
    assert tq.Q_SELECT_MODES == ("last", "val")


def test_val_selection_records_the_chosen_epoch_and_validation_loss(tmp_path):
    rep = _train(tmp_path, q_select="val", phi_val_every=2)
    assert rep["q_select"] == "val"
    assert rep["q_best_epoch"] in (2, 4, 6)
    assert rep["q_val_mse"] >= 0
    assert rep["n_val_rallies"] >= 1


def test_val_selection_works_with_train_loss_phi_selection(tmp_path):
    rep = _train(tmp_path, q_select="val", phi_select="train_loss", phi_val_every=2)
    assert rep["n_val_rallies"] >= 1 and rep["phi_select"] == "train_loss"


def test_last_is_the_default(tmp_path):
    rep = _train(tmp_path)
    assert rep["q_select"] == "last" and rep["q_best_epoch"] == 6


def test_report_has_the_trivial_baselines(tmp_path):
    rep = _train(tmp_path)
    assert "heldout_residuals_constant_phi" in rep
    assert rep["heldout_q1_mse_predict_zero"] >= 0


def test_unknown_mode_raises(tmp_path):
    with pytest.raises(ValueError):
        _train(tmp_path, q_select="best")
