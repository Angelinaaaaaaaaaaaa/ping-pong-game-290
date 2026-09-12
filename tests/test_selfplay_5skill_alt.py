"""Tests for selfplay_5skill_alt.py.

Pure Python, no MuJoCo.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nash_skills.skills import SKILL_NAMES


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
