"""scripts/diagnose_static_opponent.py: attributing each reception to the preparation before it."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from diagnose_static_opponent import reception_outcomes


def _dec(attacker, p1, p2):
    return {"kind": "decision", "attacker": attacker, "prep": {1: p1, 2: p2}}


def test_returned_then_missed_with_the_preparation_in_force():
    events = [
        _dec(2, "center", "right"),          # P2 attacks first, P1 defends with center
        {"kind": "contact", "hitter": 2},     # P2 hits toward P1
        _dec(1, "center", "right"),          # now P1 attacks
        {"kind": "contact", "hitter": 1},     # P1 returned it
        _dec(2, "left", "right"),            # P1 defends again, now with left
        {"kind": "contact", "hitter": 2},
        _dec(1, "left", "right"),
        {"kind": "end", "reason": "missed_ego_return_after_valid_bounce"},
    ]
    assert reception_outcomes(events, me=1) == [("center", "returned"), ("left", "missed")]


def test_their_shot_out_and_truncation():
    # P1 attacks, P2 defends with "left"; P1's shot goes out before bouncing on P2's side
    events = [_dec(1, "center", "left"), {"kind": "contact", "hitter": 1},
              {"kind": "end", "reason": "ego_shot_out_before_valid_opp_bounce"}]
    assert reception_outcomes(events, me=2) == [("left", "their_out")]
    open_events = [_dec(1, "center", "left"), {"kind": "contact", "hitter": 1}, {"kind": "end", "reason": "truncated"}]
    assert reception_outcomes(open_events, me=2) == [("left", "other:truncated")]
