"""nash_skills.kexin_v3_player: Kexin's V3 alpha-potential rules playing inside the Design B match runner."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from nash_skills import kexin_v3_player as kp
from nash_skills.defend_positions import defend_position_index
from nash_skills.skills import skill_index
from nash_skills.v2.joint_action import joint_action_from_index, joint_action_index


class _PhiTable(torch.nn.Module):
    """Stub for her Phi network: reads the 5 + 3 action one-hots at the end of her 82-dim input."""

    def __init__(self, table):
        super().__init__()
        self.table = torch.tensor(table, dtype=torch.float32)

    def forward(self, x):
        a, d = x[:, 74:79].argmax(dim=1), x[:, 79:82].argmax(dim=1)
        return self.table[a, d].unsqueeze(1)


def _picker(rule, table, monkeypatch):
    monkeypatch.setattr(kp, "state_for_attacker", lambda obs, info, attacker: np.zeros(74, dtype=np.float32))
    return kp.make_kexin_v3_picker(rule, potential_model=_PhiTable(table))


def _ours(attack, defend):
    return joint_action_index(skill_index(attack), defend_position_index(defend))


def _info(attacker, k, current):
    return {"attacker": attacker, "decision_index": k, "current_joints": current}


def test_names_translate_between_her_convention_and_ours():
    assert [kp.ours_attack_from_v3(a) for a in range(5)] == [4, 3, 2, 1, 0]   # her left is our right
    assert kp.v3_defense_from_ours(defend_position_index("center")) == 1      # her order is left, center, right
    assert kp.ours_defense_from_v3(2) == defend_position_index("right")


def test_minimax_attacker_returns_her_shot_in_our_names_and_keeps_its_preparation(monkeypatch):
    table = np.zeros((5, 3))
    table[0, :] = 1.0                       # her "left" is the best worst case
    pick = _picker("potential_minimax", table, monkeypatch)
    own = _ours("center_safe", "right")
    out = pick(1, None, 0, _info(attacker=1, k=0, current=(own, 0)))
    assert joint_action_from_index(out) == (skill_index("right"), defend_position_index("right"))


def test_br_defender_answers_the_opponents_last_shot_and_defaults_to_centre(monkeypatch):
    table = np.zeros((5, 3))
    table[2, 0] = 1.0                       # vs her center_safe, her best pre-position is left
    table[0, 2] = 1.0                       # vs her "left" (our "right"), her best is right
    pick = _picker("potential_br", table, monkeypatch)
    own = _ours("left", "center")
    first = pick(2, None, 0, _info(attacker=1, k=0, current=(0, own)))        # nothing seen yet: centre shot assumed
    assert joint_action_from_index(first)[1] == defend_position_index("left")
    opp = _ours("right", "center")          # the opponent's latest shot, our name "right" = her "left"
    pick(2, None, opp, _info(attacker=2, k=1, current=(opp, first)))         # she attacks: sees the opponent's preparation
    later = pick(2, None, opp, _info(attacker=1, k=2, current=(opp, first)))
    assert joint_action_from_index(later)[1] == defend_position_index("right")


def test_adaptive_match_counts_the_opponents_fresh_choices_across_rallies(monkeypatch):
    pick = _picker("potential_adaptive_match", np.zeros((5, 3)), monkeypatch)
    opp = _ours("right", "left")
    pick(1, None, 0, _info(attacker=1, k=0, current=(0, 0)))                 # rally start: the placeholder is not counted
    pick(1, None, opp, _info(attacker=2, k=1, current=(0, opp)))             # opponent prepared last: its position counts
    pick(1, None, opp, _info(attacker=1, k=2, current=(0, opp)))             # opponent attacked last: its shot counts
    pick(1, None, opp, _info(attacker=1, k=0, current=(0, opp)))             # next rally starts
    counts = pick.opponent_counts[1]
    assert counts["attack"].tolist() == [1, 0, 0, 0, 0]                       # our "right" = her "left"
    assert counts["defense"].tolist() == [1, 0, 0]


def test_unknown_rule_is_refused(monkeypatch):
    with pytest.raises(ValueError):
        _picker("potential_hard", np.zeros((5, 3)), monkeypatch)


def test_match_runner_builds_her_player_from_a_spec(monkeypatch):
    from nash_skills import eval_matchup_attack_defend as emad
    loaded = []
    monkeypatch.setattr(kp, "load_potential_model", lambda path, device="cpu": loaded.append(path) or _PhiTable(np.zeros((5, 3))))
    monkeypatch.setattr(kp, "state_for_attacker", lambda obs, info, attacker: np.zeros(74, dtype=np.float32))
    pick = emad.make_picker_joint("kexin-v3:potential_minimax:/models/v3/potential_50k_aggressive.pth", None)
    assert loaded == ["/models/v3/potential_50k_aggressive.pth"]
    assert 0 <= pick(1, None, 0, _info(attacker=1, k=0, current=(0, 0))) < 15
