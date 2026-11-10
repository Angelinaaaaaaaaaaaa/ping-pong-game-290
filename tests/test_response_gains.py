"""Unilateral response gains (paper Eq. 15) via exact simulated continuations: nash_skills.response_gains."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from nash_skills.response_gains import STATIC_RESPONSES, gain_from_values


def test_fifteen_static_responses():
    assert len(STATIC_RESPONSES) == 15 and len(set(STATIC_RESPONSES)) == 15


def test_gain_counts_the_original_among_the_candidates():
    out = gain_from_values(0.2, {3: -0.5, 7: 0.9, 11: 0.1})
    assert out["gain"] == pytest.approx(0.7)
    assert out["best_response"] == 7
    out = gain_from_values(0.4, {3: -0.5, 7: 0.1})
    assert out["gain"] == 0.0 and out["best_response"] is None


def test_decision_timing_is_validated_before_any_rally_is_played():
    from nash_skills.response_gains import ResponseGainEvaluator, collect_decision_points

    class Untouchable:
        def __getattr__(self, name):
            raise AssertionError(f"env.{name} used before the timing was checked")

    with pytest.raises(ValueError):
        collect_decision_points(Untouchable(), None, "preparation", lambda: None, 1, np.random.default_rng(0), 10,
                                decision_timing="bounce")
    with pytest.raises(ValueError):
        ResponseGainEvaluator(Untouchable(), None, "preparation", 0.99, 10, lambda: None, decision_timing="bounce")
    assert ResponseGainEvaluator(Untouchable(), None, "preparation", 0.99, 10, lambda: None).timing == "contact"


mujoco_available = True
try:
    import mujoco  # noqa: F401
except ImportError:
    mujoco_available = False


class _TanhPPO:
    def predict(self, obs, deterministic=True):
        return np.tanh(obs[18:27]).astype(np.float32), None


@pytest.mark.skipif(not mujoco_available, reason="needs MuJoCo")
def test_replaying_the_method_itself_reproduces_its_value_and_bounds_hold():
    os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
    from nash_skills.env_wrapper import SkillEnv
    from nash_skills.eval_matchup_attack_defend import make_picker_joint
    from nash_skills.response_gains import ResponseGainEvaluator, collect_decision_points

    def make():
        return SkillEnv(proc_id=1, history=4, reset_mode="ready", skill_profile="aggressive", reset_targets=True)

    method = lambda: make_picker_joint(("left", "center"), None)   # a deterministic "method"
    ppo = _TanhPPO()
    points = collect_decision_points(make(), ppo, "preparation_cal", method, n_points=2,
                                     rng=np.random.default_rng(0), max_steps=600, take_prob=1.0)
    assert len(points) == 2
    ev = ResponseGainEvaluator(make(), ppo, "preparation_cal", gamma=0.99, max_steps=600,
                               make_method_picker=method, repeats=1)
    p = points[0]
    v_orig = ev.value(p, player=1, response=None)
    same_pair = STATIC_RESPONSES[0 * 3 + 2]   # (left, center) is joint index 2
    assert ev.value(p, player=1, response=same_pair) == v_orig
    assert -1.0 <= v_orig <= 1.0
    out = ev.gains(p, player=2)
    assert out["gain"] >= 0.0 and len(out["values"]) == 15


def test_policy_level_gain_scores_each_candidate_by_its_mean_over_points():
    from nash_skills.response_gains import policy_level_gain
    # candidate 3 wins point A only, candidate 5 wins point B only: neither is better on average
    vals = [{"orig": 0.0, 3: 1.0, 5: -1.0}, {"orig": 0.0, 3: -1.0, 5: 1.0}]
    assert policy_level_gain(vals)["gain"] == 0.0
    vals = [{"orig": -0.2, 3: 0.4, 5: 0.0}, {"orig": 0.2, 3: 0.4, 5: 0.0}]
    out = policy_level_gain(vals)
    assert out["gain"] == pytest.approx(0.4) and out["best_response"] == 3 and out["original_mean"] == 0.0


def test_summary_reports_both_players_and_the_profile_regret():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from eval_response_gains import summarize
    pts = [{"p1": {"orig": 0.0, "3": 0.5}, "p2": {"orig": 0.0, "3": -0.5}},
           {"p1": {"orig": 0.0, "3": 0.3}, "p2": {"orig": 0.0, "3": -0.1}}]
    s = summarize(pts)
    assert s["player1"]["gain"] == pytest.approx(0.4) and s["player2"]["gain"] == 0.0
    assert s["profile_regret"] == pytest.approx(0.4)
