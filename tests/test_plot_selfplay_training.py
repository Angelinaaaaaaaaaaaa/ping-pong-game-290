"""scripts/plot_selfplay_training.py: per-iteration self-play logs -> smoothed curves."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

import plot_selfplay_training as pt


def test_rolling_mean_uses_trailing_window():
    out = pt.rolling_mean(np.array([0.0, 2.0, 4.0, 6.0]), window=2)
    assert out.tolist() == [0.0, 1.0, 3.0, 5.0]


def test_read_log_extracts_curves(tmp_path):
    f = tmp_path / "s0.csv"
    f.write_text("iter,learner,win_rate,trunc_rate,mean_exchanges,mean_steps,n_decisions,attack_usage,defend_usage,"
                 "policy_loss,value_loss,entropy,pool_size,wall_time\n"
                 "1,1,0.5,0.0,6.0,400,100,1 1 1 1 1,1 1 2,0,0,1.2,0,30\n"
                 "2,1,0.5,0.0,8.0,500,100,5 0 0 0 0,0 0 4,0,0,1.0,0,30\n")
    c = pt.read_log(str(f))
    assert c["iter"].tolist() == [1, 2] and c["exchanges"].tolist() == [6.0, 8.0]
    assert c["prep_center"].tolist() == pytest.approx([0.5, 1.0]) and c["entropy"].tolist() == [1.2, 1.0]
    assert c["target_share"].shape == (2, 5) and c["target_share"][1, 0] == 1.0
