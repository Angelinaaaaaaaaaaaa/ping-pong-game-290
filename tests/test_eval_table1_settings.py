"""scripts/eval_table1_attack_defend.py: settings check, task splitting, timing summary."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

import eval_table1_attack_defend as t1
from nash_skills.v2.state_encoder_attack_defend import STATE_ENCODER_NAME


def _model_dir(tmp_path, **meta):
    report = {"encoder": STATE_ENCODER_NAME, "defend_mechanism": "preparation_cal",
              "decision_timing": "contact", "skill_profile": "moderate_aggressive", **meta}
    (tmp_path / "heldout_attack_defend.json").write_text(json.dumps(report))
    return str(tmp_path)


SETTINGS = {"defend_mechanism": "preparation_cal", "decision_timing": "contact",
            "skill_profile": "moderate_aggressive"}


def test_matching_settings_pass(tmp_path):
    t1.check_model_settings(_model_dir(tmp_path), SETTINGS)


@pytest.mark.parametrize("key,value", [("decision_timing", "crossing"), ("skill_profile", "aggressive"),
                                       ("defend_mechanism", "preparation")])
def test_a_model_trained_under_other_settings_is_refused(tmp_path, key, value):
    with pytest.raises(ValueError, match=key):
        t1.check_model_settings(_model_dir(tmp_path, **{key: value}), SETTINGS)


def test_selfplay_checkpoint_settings_are_checked_too(tmp_path):
    path = tmp_path / "sp.pth"
    (tmp_path / "sp.pth.json").write_text(json.dumps({"per_player": True, "decision_timing": "contact",
                                                      "skill_profile": "aggressive",
                                                      "defend_mechanism": "preparation_cal"}))
    with pytest.raises(ValueError, match="skill_profile"):
        t1.check_selfplay_settings(f"selfplay:{path}", SETTINGS)
    t1.check_selfplay_settings("random", SETTINGS)  # non-selfplay specs are ignored


def test_split_gives_chunks_that_add_up_with_distinct_seeds():
    tasks = t1.build_tasks("m", ["random", "ibr-sim"], episodes=10, split={"ibr-sim": 4}, seed=3)
    ibr = [t for t in tasks if t["opponent"] == "ibr-sim"]
    assert len(ibr) == 2 * 4
    for side in (1, 2):
        assert sum(t["episodes"] for t in ibr if t["method_player"] == side) == 10
    assert len({t["seed"] for t in tasks}) == len(tasks)


def test_build_tasks_caps_attempts_per_task():
    # default: no cap recorded, so run_matchup_joint keeps its 10x n_episodes budget
    assert all("max_attempts" not in t for t in t1.build_tasks("m", ["random"], 10, {}, seed=0))
    tasks = t1.build_tasks("m", ["random", "ibr-sim"], episodes=10, split={"ibr-sim": 4}, seed=0,
                           max_attempt_factor=2.0)
    assert {t["max_attempts"] for t in tasks if t["opponent"] == "random"} == {20}
    assert sorted({(t["episodes"], t["max_attempts"]) for t in tasks if t["opponent"] == "ibr-sim"}) == [(2, 4), (3, 6)]
    # a factor below 1 still allows one attempt per requested rally
    assert {t["max_attempts"] for t in t1.build_tasks("m", ["random"], 10, {}, 0, max_attempt_factor=0.5)} == {10}


def test_timing_summary_averages_decision_seconds_per_side():
    rows = [{"opponent": "ibr-sim", "method_player": 1, "method_seconds": [0.1, 0.3], "opponent_seconds": [2.0]},
            {"opponent": "ibr-sim", "method_player": 2, "method_seconds": [0.2], "opponent_seconds": [4.0, 6.0]}]
    s = t1.timing_summary(rows)
    assert s["method_mean_s"] == pytest.approx(0.2)
    assert s["by_opponent"]["ibr-sim"]["opponent_mean_s"] == pytest.approx(4.0)
