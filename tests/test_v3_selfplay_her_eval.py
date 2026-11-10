"""v3_selfplay.her_eval: self-play policies inside Kexin's own evaluator (v3/eval_q_policy.run_episode)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

import v3.eval_policy_fixed_grid as grid
import v3.eval_q_policy as her
from v3_selfplay import her_eval
from v3_selfplay import policy as sp


def _net(shot, position):
    net = sp.RolePolicy()
    with torch.no_grad():
        net.shot_head.bias.zero_()
        net.shot_head.bias[shot] = 50.0
        net.position_head.bias.zero_()
        net.position_head.bias[position] = 50.0
    return net


def _entries():
    return {"sp": her_eval.Entry(policies={1: _net(1, 0), 2: _net(3, 2)}, argmax=True)}


@pytest.fixture
def fake_state(monkeypatch):
    monkeypatch.setattr(her, "state_for_attacker", lambda obs, info, attacker: np.full(74, float(attacker)))


def _ask(policy, role):
    return her._distribution_for_role(policy, role, np.zeros((5, 3)), None, fixed_shot="left",
                                      fixed_defense="center", greedy_iterations=4)


def test_her_evaluator_asks_our_networks_for_registered_names(fake_state):
    with her_eval.installed(_entries()):
        her.state_for_attacker(None, None, 2)                     # her run_episode encodes P2 attacking
        assert _ask("sp", "attacker").tolist() == [0, 0, 0, 1, 0]   # P2's own network attacks
        assert _ask("sp", "defender").tolist() == [1, 0, 0]         # P1's network defends
        her.state_for_attacker(None, None, 1)
        assert _ask("sp", "attacker").tolist() == [0, 1, 0, 0, 0]


def test_other_policies_keep_her_behaviour_and_everything_is_restored(fake_state):
    original = her._distribution_for_role
    with her_eval.installed(_entries()):
        assert _ask("random", "defender") == pytest.approx(np.full(3, 1 / 3))
        assert "sp" in her.POLICY_NAMES and "sp" in grid.POLICY_NAMES
        assert grid.build_parser().parse_args(["--policies", "sp", "nash_q"]).policies == ["sp", "nash_q"]
    assert her._distribution_for_role is original and "sp" not in her.POLICY_NAMES + grid.POLICY_NAMES


def test_entries_parse_name_checkpoint_and_argmax(tmp_path):
    path = str(tmp_path / "s0.pth")
    sp.save_policies({1: sp.RolePolicy(), 2: sp.RolePolicy()}, path, {})
    name, entry = her_eval.parse_entry(f"sp_argmax={path}:argmax")
    assert name == "sp_argmax" and entry.argmax and set(entry.policies) == {1, 2}
    assert her_eval.parse_entry(f"sp={path}")[1].argmax is False
    with pytest.raises(ValueError):
        her_eval.parse_entry(path)


def test_head_to_head_plays_both_sides_per_seed_and_scores_like_her(monkeypatch):
    runs = []

    def fake_episode(*, episode_index, policies, match_action_counts, episode_seed, max_steps, **_kw):
        runs.append((dict(policies), match_action_counts, max_steps))          # kept alive: ids stay unique
        tested = 1 if policies[1]["attack"] == "sp" else 2
        winner = tested if episode_index == 0 else (3 - tested if episode_index == 1 else None)
        return {"winner_player": winner, "steps": 100}, []

    monkeypatch.setattr(her, "run_episode", fake_episode)
    rows = her_eval.head_to_head(tested="sp", opponents=["potential_minimax"], seeds=[0, 1], episodes=3,
                                 both_sides=True, env=None, ppo=None, q_model=None, potential_model=None, steps=600)
    assert len(rows) == 4 and len(runs) == 12 and {r[2] for r in runs} == {600}
    assert len({id(r[1]) for r in runs}) == 4                                # match history reset per run
    summary = her_eval.summarize_head_to_head(rows)["potential_minimax"]
    assert summary["episodes"] == 12 and summary["wins"] == 4 and summary["losses"] == 4
    assert summary["win_rate"] == pytest.approx(4 / 12)                     # her denominator: all rallies
    assert summary["decided_win_rate"] == pytest.approx(0.5)
