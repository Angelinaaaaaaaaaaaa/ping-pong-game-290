"""Tests for the aggregation in scripts/eval_table1_attack_defend.py (no MuJoCo)."""
import importlib.util
import os

import pytest

_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts", "eval_table1_attack_defend.py")
_spec = importlib.util.spec_from_file_location("eval_table1_attack_defend", _PATH)
t1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t1)


def _row(opponent, side, wins, decided, truncated=0):
    return {"opponent": opponent, "method_player": side, "wins": wins, "decided": decided,
            "truncated": truncated, "rally_lengths": []}


def test_static_expands_to_the_15_pairings():
    opponents = t1.expand_opponents(["static", "random"])
    assert len(opponents) == 16
    assert opponents[-1] == "random"
    assert ("left", "center") in opponents


def test_both_sides_are_pooled_and_reported_separately():
    s = t1.summarize([_row("random", 1, 30, 50), _row("random", 2, 20, 50)])
    cell = s["cells"]["random"]
    assert cell["win"] == pytest.approx(0.5)
    assert cell["by_side"]["1"] == {"wins": 30, "decided": 50}
    assert cell["by_side"]["2"] == {"wins": 20, "decided": 50}


def test_static_mean_is_unweighted_and_min_names_the_opponent():
    rows = []
    for i, (a, d) in enumerate(t1.STATIC_OPPONENTS):
        label = f"{a}/{d}"
        decided = 10 if i else 1000
        wins = decided if i else 0
        rows += [_row(label, 1, wins, decided), _row(label, 2, 0, 0)]
    s = t1.summarize(rows)
    assert s["static_mean"] == pytest.approx(14 / 15)
    assert s["static_min"] == 0.0
    assert s["static_min_opponent"] == "left/left"


def test_truncation_rate_counts_attempts():
    s = t1.summarize([_row("ibr", 1, 5, 8, truncated=2)])
    assert s["cells"]["ibr"]["trunc_rate"] == pytest.approx(0.2)


def test_wilson_interval_brackets_the_estimate():
    lo, hi = t1.wilson(30, 60)
    assert lo < 0.5 < hi


def test_model_dir_from_the_old_encoder_is_refused(tmp_path):
    import json
    (tmp_path / "heldout_attack_defend.json").write_text(json.dumps({"seed": 0}))
    with pytest.raises(ValueError, match="encoder"):
        t1.check_model_encoder(str(tmp_path))


def test_model_dir_with_the_current_encoder_passes(tmp_path):
    import json
    from nash_skills.v2.state_encoder_attack_defend import STATE_ENCODER_NAME
    (tmp_path / "heldout_attack_defend.json").write_text(json.dumps({"encoder": STATE_ENCODER_NAME}))
    t1.check_model_encoder(str(tmp_path))
