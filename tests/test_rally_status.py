"""nash_skills.rally_status: the paper's rally status ell_t from the environment's rally tracker."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nash_skills.rally_status import rally_status


def _state(last_hitter, bounces, had_chance=False):
    return {"last_hitter": last_hitter, "receiver_had_return_opportunity": had_chance,
            "bounce_history": [{"side": s, "last_hitter": h} for s, h in bounces]}


def test_counts_only_bounces_after_the_last_racket_contact():
    s = rally_status(_state("opp", [("ego", None), ("opp", "ego"), ("ego", "opp")], had_chance=True))
    assert s == {"last_hitter": "opp", "bounces_since_hit": 1, "receiver_had_chance": True}


def test_right_after_a_hit_nothing_has_bounced_yet():
    s = rally_status(_state("ego", [("ego", None), ("opp", "opp")]))
    assert s == {"last_hitter": "ego", "bounces_since_hit": 0, "receiver_had_chance": False}


def test_before_the_first_contact():
    assert rally_status(_state(None, [("ego", None)]))["bounces_since_hit"] == 1
    assert rally_status({"last_hitter": None})["bounces_since_hit"] == 0


def test_does_not_modify_the_tracker():
    st = _state("ego", [("opp", "ego")])
    before = repr(st)
    rally_status(st)
    assert repr(st) == before
