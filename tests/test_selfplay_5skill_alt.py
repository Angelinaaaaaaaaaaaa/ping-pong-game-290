"""Tests for selfplay_5skill_alt.py.

TDD: written before the fix. Pure Python, no MuJoCo -- covers the
win/draw/loss outcome classification bug found during code review: the
alternating trainer's win/draw counting used `ego_r > 0.5` as a proxy for
"win" because play_one_rally only returned the shaping-inflated ego_reward,
not the raw terminal outcome. Real collected-data audits showed truncated
rallies average ~0.6 shaped reward (above that 0.5 threshold), so most
truncated rallies were being logged as wins. The fix has play_one_rally also
return the raw ego_terminal so classification doesn't need to guess.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from selfplay_5skill_alt import classify_outcome
from nash_skills.skills import SKILL_NAMES


class TestClassifyOutcome:
    def test_win_is_positive_one(self):
        assert classify_outcome(1.0) == "win"

    def test_loss_is_negative_one(self):
        assert classify_outcome(-1.0) == "loss"

    def test_truncated_is_zero(self):
        assert classify_outcome(0.0) == "draw"

    def test_not_fooled_by_large_shaped_reward(self):
        # Regression case for the actual bug: a long truncated rally's raw
        # terminal is still 0.0 regardless of how much shaping accumulated
        # -- classify_outcome only ever sees the raw terminal, never the
        # shaped total, so this can't be misclassified as a win.
        assert classify_outcome(0.0) == "draw"

    def test_boundary_values_match_original_thresholds(self):
        # Matches the >0.5 / <0.5-abs thresholds already used by
        # selfplay_5skill.py's (correct) ego_terminal = ego_r - opp_r check.
        assert classify_outcome(0.6) == "win"
        assert classify_outcome(-0.6) == "loss"
        assert classify_outcome(0.4) == "draw"
        assert classify_outcome(-0.4) == "draw"


class TestSkillLabelsNoCollision:
    """
    Regression guard for the display bug found while reviewing real local
    output: truncating skill names to a fixed width for compact printing
    made 'left'/'left_short' and 'right_short'/'right' indistinguishable.
    """

    def test_full_names_are_unique(self):
        assert len(set(SKILL_NAMES)) == len(SKILL_NAMES)

    def test_four_char_truncation_collides(self):
        # Documents *why* the old s[:4] scheme was broken -- this is the
        # bug, not the fix; the fix is to stop truncating at all.
        truncated = [s[:4] for s in SKILL_NAMES]
        assert len(set(truncated)) < len(SKILL_NAMES)

    def test_five_char_truncation_also_collides(self):
        # right_short[:5] == right[:5] == "right"
        truncated = [s[:5] for s in SKILL_NAMES]
        assert len(set(truncated)) < len(SKILL_NAMES)
