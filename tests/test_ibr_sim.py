"""Tests for nash_skills.ibr_sim (simulation-based iterated best response, paper Sec. 5.1)."""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from nash_skills.ibr_sim import iterated_best_response

# Player 1's payoff M[c1, c2]; zero-sum, so player 2 gets -M.
M = np.array([[0.0, 0.9, -0.5],
              [0.4, -0.2, 0.3],
              [-0.6, 0.1, 0.8]])


def _evaluate(calls):
    def evaluate(p, own, other):
        calls.append((p, own, other))
        return M[own, other] if p == 1 else -M[other, own]
    return evaluate


def test_alternates_best_responses_starting_and_ending_with_the_planner():
    calls = []
    best, trace = iterated_best_response(1, _evaluate(calls), n_actions=3, rounds=2, init=0,
                                         rng=np.random.default_rng(0))
    # P1 vs 0 -> 1 (0.4); P2 vs 1 -> 1 (-0.2 is P1's min); P1 vs 1 -> 0 (0.9)
    assert [t["player"] for t in trace] == [1, 2, 1]
    assert [t["best"] for t in trace] == [1, 1, 0]
    assert best == 0
    assert len(calls) == 3 * 3


def test_planner_as_player_2():
    best, trace = iterated_best_response(2, _evaluate([]), n_actions=3, rounds=1, init=0,
                                         rng=np.random.default_rng(0))
    # one round: P2 best-responds to P1 playing 0 -> column minimizing M[0, :] -> 2
    assert [t["player"] for t in trace] == [2]
    assert best == 2


def test_ties_are_broken_at_random_among_the_maxima():
    flat = lambda p, own, other: 0.0
    picks = {iterated_best_response(1, flat, 3, 1, 0, np.random.default_rng(s))[0] for s in range(30)}
    assert picks == {0, 1, 2}


def test_rounds_must_be_positive():
    with pytest.raises(ValueError):
        iterated_best_response(1, _evaluate([]), 3, 0, 0, np.random.default_rng(0))


mujoco_available = True
try:
    import mujoco  # noqa: F401
except ImportError:
    mujoco_available = False


class _TanhPPO:
    """Deterministic stand-in controller (no checkpoint needed)."""

    def predict(self, obs, deterministic=True):
        return np.tanh(obs[18:27]).astype(np.float32), None


@pytest.mark.skipif(not mujoco_available, reason="needs MuJoCo")
def test_rollout_value_matches_actually_playing_the_fixed_pair():
    os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
    from nash_skills.env_wrapper import SkillEnv
    from nash_skills.defend_mechanisms import set_defend_choices
    from nash_skills.ibr_sim import SimRolloutEvaluator, apply_joint_pair
    from nash_skills.v2.selfplay_rollout import _ppo_step

    def make():
        return SkillEnv(proc_id=1, history=4, reset_mode="ready", skill_profile="aggressive", reset_targets=True)

    ppo = _TanhPPO()
    real = make()
    np.random.seed(2)
    real.set_skills("left", "right")
    set_defend_choices(real, "preparation_cal", "left", "center")
    obs, info = real.reset()
    for _ in range(40):
        obs, _r, _d, _t, info = _ppo_step(real, ppo, obs, info)
    ev = SimRolloutEvaluator(make(), ppo, "preparation_cal", horizon_exchanges=50, max_steps=600, gamma=1.0)
    ev.bind(real, obs, info)
    v = ev(1, 4, 7)
    assert ev(2, 7, 4) == -v  # same rollout, zero-sum
    assert ev.rollouts == 1   # cached
    # play it for real
    apply_joint_pair(real, "preparation_cal", 4, 7)
    done = False
    for _ in range(600):
        obs, _r, done, _t, info = _ppo_step(real, ppo, obs, info)
        if done:
            break
    from nash_skills.winner_inference import infer_terminal_winner
    expected = 0.0 if not done else (1.0 if infer_terminal_winner(obs, info, fallback="position") == "ego" else -1.0)
    assert v == expected
