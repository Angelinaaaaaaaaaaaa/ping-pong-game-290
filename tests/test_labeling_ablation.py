"""Tests for nash_skills.v2.labeling_ablation.

TDD: written before implementation. Pure Python, no MuJoCo -- covers the
three long-rally labeling strategies (discard / tie0 / asym_small) as
dataset-preparation transforms for future retraining (meeting note item 4).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from nash_skills.v2.labeling_ablation import (
    apply_discard,
    apply_tie0,
    apply_asym_small,
    MissingInitiatorFieldError,
)


def _rally(winner, initiator=None, extra=None):
    r = {"skill1": "left", "skill2": "right", "states": [1, 2, 3], "winner": winner}
    if initiator is not None:
        r["initiator"] = initiator
    if extra:
        r.update(extra)
    return r


class TestApplyDiscard:
    def test_removes_truncated_rallies(self):
        rallies = [_rally(1), _rally(0), _rally(2), _rally(None)]
        result = apply_discard(rallies)
        assert len(result) == 2
        assert all(r["winner"] in (1, 2) for r in result)

    def test_keeps_all_decided_rallies(self):
        rallies = [_rally(1), _rally(2), _rally(1)]
        result = apply_discard(rallies)
        assert len(result) == 3

    def test_does_not_mutate_input(self):
        rallies = [_rally(1), _rally(0)]
        original_len = len(rallies)
        apply_discard(rallies)
        assert len(rallies) == original_len

    def test_empty_input(self):
        assert apply_discard([]) == []


class TestApplyTie0:
    def test_truncated_rallies_relabeled_as_tie(self):
        rallies = [_rally(0), _rally(None)]
        result = apply_tie0(rallies)
        assert all(r["winner"] == 0 for r in result)

    def test_decided_rallies_unchanged(self):
        rallies = [_rally(1), _rally(2)]
        result = apply_tie0(rallies)
        assert result[0]["winner"] == 1
        assert result[1]["winner"] == 2

    def test_same_length_as_input(self):
        rallies = [_rally(1), _rally(0), _rally(None), _rally(2)]
        result = apply_tie0(rallies)
        assert len(result) == len(rallies)

    def test_does_not_mutate_input(self):
        rallies = [_rally(None)]
        apply_tie0(rallies)
        assert rallies[0]["winner"] is None  # original untouched

    def test_empty_input(self):
        assert apply_tie0([]) == []


class TestApplyAsymSmall:
    def test_raises_when_initiator_field_missing(self):
        rallies = [_rally(1), _rally(0)]
        with pytest.raises(MissingInitiatorFieldError):
            apply_asym_small(rallies)

    def test_error_message_is_clear(self):
        rallies = [_rally(1)]
        with pytest.raises(MissingInitiatorFieldError, match="initiator"):
            apply_asym_small(rallies)

    def test_succeeds_when_initiator_field_present(self):
        rallies = [_rally(0, initiator=1), _rally(0, initiator=2), _rally(1, initiator=1)]
        result = apply_asym_small(rallies)
        assert len(result) == 3

    def test_truncated_rally_initiator_gets_small_positive(self):
        rallies = [_rally(0, initiator=1)]
        result = apply_asym_small(rallies)
        assert result[0]["reward1"] > 0
        assert result[0]["reward2"] < 0

    def test_truncated_rally_non_initiator_gets_small_negative(self):
        rallies = [_rally(0, initiator=2)]
        result = apply_asym_small(rallies)
        assert result[0]["reward2"] > 0
        assert result[0]["reward1"] < 0

    def test_decided_rally_unaffected_by_asym_reward(self):
        rallies = [_rally(1, initiator=1)]
        result = apply_asym_small(rallies)
        assert result[0]["winner"] == 1
