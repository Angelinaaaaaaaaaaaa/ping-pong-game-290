import numpy as np
import torch

from nash_skills.eval_matchup import _build_phi_table, make_picker
from nash_skills.skills import N_SKILLS


class TablePhi(torch.nn.Module):
    def __init__(self, table):
        super().__init__()
        self.table = torch.tensor(table, dtype=torch.float32)

    def forward(self, x):
        p1 = torch.round(x[:, -2] * (N_SKILLS - 1)).long()
        p2 = torch.round(x[:, -1] * (N_SKILLS - 1)).long()
        return self.table[p1, p2].unsqueeze(1)


class SkillQ(torch.nn.Module):
    def __init__(self, values, skill_slot):
        super().__init__()
        self.values = torch.tensor(values, dtype=torch.float32)
        self.skill_slot = skill_slot

    def forward(self, x):
        idx = torch.round(x[:, self.skill_slot] * (N_SKILLS - 1)).long()
        return self.values[idx].unsqueeze(1)


def _obs():
    return np.zeros(76, dtype=np.float32)


def test_phi_table_uses_p1_rows_and_p2_columns_with_global_skill_slots():
    values = np.arange(N_SKILLS * N_SKILLS, dtype=np.float32).reshape(N_SKILLS, N_SKILLS)
    phi = _build_phi_table(_obs(), None, player=1, model_p=TablePhi(values), state_encoder_fn=None)

    assert phi.shape == (N_SKILLS, N_SKILLS)
    assert phi[0, 0].item() == values[0, 0]
    assert phi[1, 3].item() == values[1, 3]
    assert phi[4, 2].item() == values[4, 2]


def test_ibr_player1_uses_argmax_q1():
    q1 = SkillQ([0.0, 1.0, 10.0, 3.0, 4.0], skill_slot=-2)
    q2 = SkillQ([0.0, 7.0, 2.0, 4.0, 3.0], skill_slot=-1)
    pick = make_picker("ibr", model_p=None, model1=q1, model2=q2)

    assert pick(player=1, obs_vec=_obs(), other_skill_idx=0) == 2


def test_ibr_player2_uses_argmax_q2():
    q1 = SkillQ([0.0, 1.0, 10.0, 3.0, 4.0], skill_slot=-2)
    q2 = SkillQ([0.0, 7.0, 2.0, 4.0, 3.0], skill_slot=-1)
    pick = make_picker("ibr", model_p=None, model1=q1, model2=q2)

    assert pick(player=2, obs_vec=_obs(), other_skill_idx=0) == 1


def test_nash_p_br_selects_best_p1_row_and_best_p2_column():
    table = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0, 0.0],
            [1.0, 5.0, 3.0, 8.0, 2.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 9.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    pick = make_picker("nash-p-br", TablePhi(table), tau=0.0, confidence_margin=0.0)

    assert pick(1, _obs(), 1) == 4
    assert pick(2, _obs(), 2) == 3


def test_nash_p_hard_returns_row_for_p1_and_column_for_p2():
    table = np.zeros((N_SKILLS, N_SKILLS), dtype=np.float32)
    table[2, 3] = 10.0
    table[4, 1] = 9.0
    pick = make_picker("nash-p-hard", TablePhi(table), tau=0.0, confidence_margin=0.0)

    assert pick(1, _obs(), 0) == 2
    assert pick(2, _obs(), 0) == 3
