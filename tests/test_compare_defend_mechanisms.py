"""Tests for the analysis helpers of scripts/compare_defend_mechanisms.py,
on synthetic rally outcomes (no MuJoCo)."""
import importlib.util
import os

import numpy as np
import pytest

_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts", "compare_defend_mechanisms.py")
_spec = importlib.util.spec_from_file_location("compare_defend_mechanisms", _PATH)
cdm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cdm)


def _rally(winner, steps=100):
    return {"winner": winner, "steps": steps, "returns": [1, 1]}


def _unit(skill, tested, outcome_for_key):
    keys = ["off"] + [f"{m}:{p}" for m in ("reactive", "preparation") for p in ("left", "center", "right")]
    return {"skill": skill, "tested": tested, "seed": 0,
            "rallies": {k: _rally(outcome_for_key(k)) for k in keys}}


class TestTestedWin:
    def test_player1_wins_when_ego_wins(self):
        assert cdm.tested_win(_rally("ego"), 1) == 1
        assert cdm.tested_win(_rally("opp"), 1) == 0

    def test_player2_wins_when_opp_wins(self):
        assert cdm.tested_win(_rally("opp"), 2) == 1

    def test_truncated_is_none(self):
        assert cdm.tested_win(_rally(None), 1) is None


class TestPairedDiffs:
    def test_drops_pairs_with_an_undecided_rally(self):
        unit = {"tested": 1, "rallies": {}}
        a = lambda _u: [_rally("ego"), _rally(None)]
        b = lambda _u: [_rally("opp"), _rally("opp")]
        assert cdm.paired_diffs([unit], a, b) == [1.0]

    def test_unit_with_no_decided_pair_is_skipped(self):
        unit = {"tested": 1, "rallies": {}}
        assert cdm.paired_diffs([unit], lambda _u: [_rally(None)], lambda _u: [_rally("ego")]) == []


class TestBootstrap:
    def test_constant_values_give_a_degenerate_interval(self):
        mean, lo, hi = cdm.bootstrap_ci([0.5] * 10, np.random.default_rng(0))
        assert mean == lo == hi == 0.5


class TestSummarizeOneSided:
    def _winner(self, tested_wins, tested):
        return ("ego" if tested == 1 else "opp") if tested_wins else ("opp" if tested == 1 else "ego")

    def test_matched_position_advantage_is_detected_and_attributed_to_its_side(self):
        units = []
        for skill in ("left", "right"):
            for tested in (1, 2):
                side = cdm.SKILL_SIDE[skill]

                def outcome(key, side=side, tested=tested):
                    wins = key == f"preparation:{side}"
                    return self._winner(wins, tested)
                units += [_unit(skill, tested, outcome) for _ in range(5)]
        s = cdm.summarize_one_sided(units, np.random.default_rng(0))
        prep = s["mechanisms"]["preparation"]
        assert prep["matched_minus_mismatched"][0] == pytest.approx(1.0)
        assert prep["content_detected"] is True
        assert s["mechanisms"]["reactive"]["content_detected"] is False
        assert prep["win_by_skill_and_position"]["left"]["left"] == 1.0
        assert prep["win_by_skill_and_position"]["left"]["right"] == 0.0

    def test_gate_compares_each_position_against_the_same_off_rally(self):
        units = [_unit("left", 1, lambda k: "opp" if k == "off" else "ego") for _ in range(4)]
        s = cdm.summarize_one_sided(units, np.random.default_rng(0))
        assert s["mechanisms"]["reactive"]["gate"][0] == pytest.approx(1.0)
        assert s["mechanisms"]["reactive"]["gate_pass"] is True
        assert s["sanity_reactive_center_equals_off"] == 0.0
