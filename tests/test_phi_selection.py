"""Tests for nash_skills.v2.phi_selection (Phi checkpoint selection on a validation
split, rally metadata) and its use in the Design B trainer."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch
import torch.nn as nn

from nash_skills.v2.phi_selection import PHI_SELECT_MODES, PhiValidator, rally_metadata
from nash_skills.v2.state_encoder_attack_defend import STATE_DIM_ATTACK_DEFEND
from nash_skills.v2.train_q_model_5skill_v3_attack_defend import build_constraints, residual_stats
from tests.test_train_q_model_5skill_v3_attack_defend import (
    TestOutDir, _EgoActionValue, _ExactPotential, _OppActionValue,
)


class _Const(nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 1)


def test_modes():
    assert PHI_SELECT_MODES == ("train_loss", "val_residual")


def test_residual_stats_include_the_median():
    s = residual_stats([1.0, -2.0, 3.0])
    assert s["median_abs"] == pytest.approx(2.0)


class TestPhiValidator:
    def _validator(self, stat="p95"):
        x = torch.zeros(16, STATE_DIM_ATTACK_DEFEND)
        return PhiValidator(x, build_constraints(), theta_mode="pure", seed=0, n_constraints=30,
                            batch_size=8, stat=stat)

    def test_exact_potential_scores_zero_and_beats_a_constant(self):
        v1, v2 = _EgoActionValue(), _OppActionValue()
        val = self._validator()
        assert val.score(_ExactPotential(v1, v2), v1, v2) == pytest.approx(0.0, abs=1e-6)
        assert val.score(_Const(), v1, v2) > 0.05

    def test_same_model_gives_the_same_score_every_call(self):
        v1, v2 = _EgoActionValue(), _OppActionValue()
        val = self._validator(stat="max")
        assert val.score(_Const(), v1, v2) == val.score(_Const(), v1, v2)

    def test_unknown_stat_raises(self):
        with pytest.raises(ValueError):
            self._validator(stat="min")


class TestRallyMetadata:
    def test_reads_the_shared_tags(self):
        rallies = [{"defend_mechanism": "preparation_cal", "decision_timing": "contact",
                    "skill_profile": "moderate_aggressive"}] * 3
        assert rally_metadata(rallies) == {"defend_mechanism": "preparation_cal", "decision_timing": "contact",
                                           "skill_profile": "moderate_aggressive"}

    def test_untagged_rallies_get_the_defaults_they_were_collected_with(self):
        assert rally_metadata([{}]) == {"defend_mechanism": "reactive", "decision_timing": "crossing",
                                        "skill_profile": "aggressive"}

    def test_mixed_tags_raise(self):
        with pytest.raises(ValueError, match="decision_timing"):
            rally_metadata([{"decision_timing": "contact"}, {"decision_timing": "crossing"}])


def test_trainer_selects_on_validation_and_reports_it(tmp_path):
    from nash_skills.v2 import train_q_model_5skill_v3_attack_defend as tq
    path = tmp_path / "r.pkl"
    TestOutDir()._write_rallies(path, n_rallies=12)
    report = tq.train(str(path), n_epochs=4, lr=1e-3, out_dir=str(tmp_path / "run"),
                      phi_select="val_residual", val_frac=0.2, phi_val_every=2)
    saved = json.load(open(tmp_path / "run" / "heldout_attack_defend.json"))
    assert saved["phi_select"] == "val_residual"
    assert saved["n_val_rallies"] >= 1
    assert saved["phi_best_epoch"] in (2, 4)
    assert "median_abs" in saved["heldout_residuals"]
    assert saved["decision_timing"] == "crossing"
    assert report["n_train_rallies"] + saved["n_val_rallies"] + report["n_holdout_rallies"] == 12
