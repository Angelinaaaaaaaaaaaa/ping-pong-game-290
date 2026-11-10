"""scripts/diagnose_unfinished_rallies.py: what happens to rallies unfinished at the evaluation's
step limit when they are allowed to run much longer."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

import diagnose_unfinished_rallies as du

# steps=None: still unfinished at the long limit
RECORDS = [{"steps": 400, "winner": "method"}, {"steps": 900, "winner": "opponent"},
           {"steps": 3000, "winner": "method"}, {"steps": 12000, "winner": "opponent"},
           {"steps": 4000, "winner": "method"}, {"steps": None, "winner": None}]


def test_outcomes_at_a_cutoff():
    assert du.outcomes_at(RECORDS, 2500) == {"wins": 1, "losses": 1, "unfinished": 4}
    assert du.outcomes_at(RECORDS, 20000) == {"wins": 3, "losses": 2, "unfinished": 1}


def test_scores_at_a_cutoff():
    s = du.scores_at(RECORDS, 2500)
    assert s["decided_win_rate"] == pytest.approx(0.5) and s["half_score"] == pytest.approx((1 + 0.5 * 4) / 6)
    assert s["unfinished_share"] == pytest.approx(4 / 6)
    assert du.scores_at([{"steps": None, "winner": None}], 2500)["decided_win_rate"] is None


def test_later_fate_of_rallies_unfinished_at_the_cutoff():
    fate = du.later_fate(RECORDS, 2500)
    assert fate == {"unfinished_at_cutoff": 4, "ended_later": 3, "method_won_later": 2, "never_ended": 1}


def test_parse_static_opponent():
    assert du.parse_opponent("center_safe/center") == ("center_safe", "center")
    assert du.parse_opponent("random") == "random"
