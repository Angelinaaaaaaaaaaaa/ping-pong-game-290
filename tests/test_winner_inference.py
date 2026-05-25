import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nash_skills.winner_inference import infer_terminal_winner, winner_to_label
from nash_skills.v2.labeling import detect_winner


def _obs(ball_x=1.5, ball_vel_x=0.0):
    obs = np.zeros(116, dtype=np.float32)
    obs[36] = ball_x
    obs[39] = ball_vel_x
    return obs


def _info_for_rackets(ego_x=-1.0, opp_x=2.0):
    mirrored_opp_x = 2 * 1.5 - opp_x
    return {
        "target": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "diff_pos": np.array([0.0 - ego_x, 0.0, 0.0], dtype=np.float32),
        "target_opp": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "diff_pos_opp": np.array([0.0 - mirrored_opp_x, 0.0, 0.0], dtype=np.float32),
    }


def test_explicit_info_winner_wins_first():
    obs = _obs(ball_x=0.0, ball_vel_x=-5.0)
    assert infer_terminal_winner(obs, {"winner": "ego"}) == "ego"


def test_racket_boundary_handles_zero_velocity_ego_win():
    obs = _obs(ball_x=2.31, ball_vel_x=0.0)
    assert infer_terminal_winner(obs, _info_for_rackets(), fallback=None) == "ego"


def test_racket_boundary_handles_zero_velocity_opp_win():
    obs = _obs(ball_x=-1.31, ball_vel_x=0.0)
    assert infer_terminal_winner(obs, _info_for_rackets(), fallback=None) == "opp"


def test_velocity_still_overrides_position_when_no_racket_info():
    obs = _obs(ball_x=2.5, ball_vel_x=-3.0)
    assert infer_terminal_winner(obs, {}, fallback="position") == "opp"


def test_ambiguous_without_fallback_returns_none_and_zero_label():
    obs = _obs(ball_x=1.5, ball_vel_x=0.0)
    winner = infer_terminal_winner(obs, {}, fallback=None)
    assert winner is None
    assert winner_to_label(winner) == 0


def test_detect_winner_uses_terminal_info():
    obs = _obs(ball_x=1.5, ball_vel_x=0.0)
    assert detect_winner([obs], done=True, info={"winner": "opp"}) == 2
