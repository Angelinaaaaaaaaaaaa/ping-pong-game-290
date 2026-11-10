"""Tests for nash_skills.eval_matchup_attack_defend.

TDD: written before implementation.

Generalizes eval_matchup.py's Phi-based and IBR/Q-based picker
strategies from a 5-way single-skill action space to the 15-way joint
(attack,defend) action space (nash_skills.v2.joint_action). Legacy
eval_matchup.py is completely untouched -- this is a parallel module.

Fake-model pattern mirrors tests/test_eval_matchup.py's
TestMakePickerNoLeftBias (an nn.Module whose forward deterministically
prefers a specific input pattern, so the dispatch logic -- not a real
trained model -- is what's under test).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch
import torch.nn as nn

from nash_skills import eval_matchup_attack_defend as emad
from nash_skills.v2.joint_action import N_JOINT_ACTIONS, joint_action_index, joint_action_from_index
from nash_skills.v2.state_encoder_attack_defend import STATE_DIM_ATTACK_DEFEND
from nash_skills.skills import N_SKILLS
from nash_skills.defend_positions import N_DEFEND_POSITIONS


def _base_vec():
    return np.full(STATE_DIM_ATTACK_DEFEND, 7.0, dtype=np.float32)


class _PrefersAttackIndex(nn.Module):
    """+1.0 when the EGO attack one-hot slot [74+idx] is 1.0, else 0.0 --
    flat across defend-position and opponent choices."""
    def __init__(self, idx: int):
        super().__init__()
        self.idx = idx

    def forward(self, x):
        return (x[:, 74 + self.idx] == 1.0).float().unsqueeze(1)


class _FlatModel(nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 1)


class TestFixedAndRandomStrategies:
    def test_fixed_tuple_strategy_always_returns_its_joint_index(self):
        from nash_skills.skills import skill_index
        from nash_skills.defend_positions import defend_position_index

        pick = emad.make_picker_joint(("left", "center"), model_p=None)
        expected = joint_action_index(skill_index("left"), defend_position_index("center"))
        assert pick(1, _base_vec(), 0) == expected
        assert pick(2, _base_vec(), 5) == expected

    def test_unknown_attack_name_in_fixed_tuple_raises(self):
        with pytest.raises(ValueError):
            emad.make_picker_joint(("not-a-skill", "center"), model_p=None)

    def test_unknown_defend_name_in_fixed_tuple_raises(self):
        with pytest.raises(ValueError):
            emad.make_picker_joint(("left", "not-a-position"), model_p=None)

    def test_random_strategy_returns_valid_joint_indices(self):
        pick = emad.make_picker_joint("random", model_p=None)
        for _ in range(50):
            idx = pick(1, _base_vec(), 0)
            assert 0 <= idx < N_JOINT_ACTIONS

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            emad.make_picker_joint("not-a-real-strategy", model_p=None)


class TestPhiStrategiesRequireModel:
    @pytest.mark.parametrize("strategy", ["nash-p-hard", "nash-p-br", "nash-p-minimax", "nash-p-adaptive",
                                          "nash-p-pga"])
    def test_raises_without_model_p(self, strategy):
        with pytest.raises(ValueError):
            emad.make_picker_joint(strategy, model_p=None)


class _LinearInSlot(nn.Module):
    """Phi(x) = x[:, slot] -- linear (so differentiable) in one action-slot
    probability. Maximized by putting all mass on that slot's index. A
    non-differentiable fake (e.g. an == 1.0 test) would give PGA a zero
    gradient at every interior point, so these must be linear."""

    def __init__(self, slot: int):
        super().__init__()
        self.slot = slot

    def forward(self, x):
        return x[:, self.slot:self.slot + 1]


class TestNashPPGA:
    """Paper Eq. 18 online phase: projected gradient ascent on Phi(x, theta)
    over theta in Delta_5 x Delta_3 per player, warm-started from the
    preceding solution, with extra random restarts."""

    def test_player1_ascends_its_own_ego_slots(self):
        preferred_attack = 3
        pick = emad.make_picker_joint("nash-p-pga", _LinearInSlot(74 + preferred_attack),
                                      pga_steps=80, pga_lr=0.3, pga_restarts=1, pga_seed=0)
        attack, _defend = joint_action_from_index(pick(1, _base_vec(), 0))
        assert attack == preferred_attack

    def test_player2_ascends_the_opp_slots_not_the_ego_ones(self):
        # Same ego-perspective base state convention as nash-p-hard's
        # row/column split: player 2 reads the OPP blocks.
        preferred_attack = 1
        pick = emad.make_picker_joint("nash-p-pga", _LinearInSlot(74 + N_SKILLS + preferred_attack),
                                      pga_steps=80, pga_lr=0.3, pga_restarts=1, pga_seed=0)
        attack, _defend = joint_action_from_index(pick(2, _base_vec(), 0))
        assert attack == preferred_attack

    def test_defend_block_is_optimized_too(self):
        preferred_defend = 2
        pick = emad.make_picker_joint("nash-p-pga", _LinearInSlot(84 + preferred_defend),
                                      pga_steps=80, pga_lr=0.3, pga_restarts=1, pga_seed=0)
        _attack, defend = joint_action_from_index(pick(1, _base_vec(), 0))
        assert defend == preferred_defend

    def test_warm_start_is_remembered_across_calls(self):
        pick = emad.make_picker_joint("nash-p-pga", _LinearInSlot(74), pga_steps=10,
                                      pga_lr=0.3, pga_restarts=1, pga_seed=0)
        assert pick.pga_state["theta"] is None
        pick(1, _base_vec(), 0)
        assert pick.pga_state["theta"] is not None
        assert len(pick.pga_state["last_result"].start_values) == 1  # 1st call: no warm start yet
        pick(1, _base_vec(), 0)
        assert len(pick.pga_state["last_result"].start_values) == 2  # warm start + 1 restart

    def test_returns_a_valid_joint_index(self):
        pick = emad.make_picker_joint("nash-p-pga", _FlatModel(), pga_steps=5, pga_restarts=1)
        assert 0 <= pick(1, _base_vec(), 7) < N_JOINT_ACTIONS


class TestNashPHardPrefersGloballyBestJointAction:
    def test_picks_the_joint_action_the_model_prefers_regardless_of_other(self):
        preferred_attack = N_SKILLS - 1  # last attack skill
        model = _PrefersAttackIndex(preferred_attack)
        pick = emad.make_picker_joint("nash-p-hard", model, tau=0.0)
        chosen = pick(1, _base_vec(), 0)
        chosen_attack, _chosen_defend = joint_action_from_index(chosen)
        assert chosen_attack == preferred_attack

    def test_returns_valid_joint_index_for_flat_model(self):
        model = _FlatModel()
        pick = emad.make_picker_joint("nash-p-hard", model, tau=0.0)
        chosen = pick(1, _base_vec(), 3)
        assert 0 <= chosen < N_JOINT_ACTIONS


class TestNashPBrConditionsOnOtherJointIndex:
    def test_player1_uses_the_column_matching_other_joint_idx(self):
        # A model that prefers ego_attack == other_joint's attack component
        # (i.e. "mirror the opponent's current attack") -- verifies pick_br
        # actually indexes phi[:, other_joint_idx] / phi[other_joint_idx, :]
        # rather than ignoring other_joint_idx.
        class PrefersMatchingAttack(nn.Module):
            def forward(self, x):
                ego_attack = x[:, 74:74 + N_SKILLS].argmax(dim=1)
                opp_attack = x[:, 74 + N_SKILLS:74 + 2 * N_SKILLS].argmax(dim=1)
                return (ego_attack == opp_attack).float().unsqueeze(1)

        model = PrefersMatchingAttack()
        pick = emad.make_picker_joint("nash-p-br", model, tau=0.0)
        other_joint = joint_action_index(2, 1)  # opponent currently playing attack=2
        chosen = pick(1, _base_vec(), other_joint)
        chosen_attack, _ = joint_action_from_index(chosen)
        assert chosen_attack == 2


class TestNashPMinimaxPicksMaximin:
    def test_player1_maximizes_worst_case_over_opponent_joint_actions(self):
        # Model: ego_attack==0 always scores 0.5 (safe); ego_attack==1
        # scores 1.0 against most opponents but 0.0 against one specific
        # opponent joint action -- minimax should prefer the SAFE one.
        class RiskyVsSafe(nn.Module):
            def forward(self, x):
                ego_attack = x[:, 74:74 + N_SKILLS].argmax(dim=1)
                opp_attack = x[:, 74 + N_SKILLS:74 + 2 * N_SKILLS].argmax(dim=1)
                opp_defend = x[:, 84 + N_DEFEND_POSITIONS:84 + 2 * N_DEFEND_POSITIONS].argmax(dim=1)
                is_risky_opp = (opp_attack == 0) & (opp_defend == 0)
                ego_is_risky_choice = (ego_attack == 1)
                risky_choice_value = torch.where(is_risky_opp, torch.tensor(0.0), torch.tensor(1.0))
                base = torch.where(ego_is_risky_choice, risky_choice_value, torch.tensor(0.5))
                return base.unsqueeze(1)

        model = RiskyVsSafe()
        pick = emad.make_picker_joint("nash-p-minimax", model, tau=0.0)
        chosen = pick(1, _base_vec(), 0)
        chosen_attack, _ = joint_action_from_index(chosen)
        assert chosen_attack == 0  # the safe choice, not the risky-but-usually-better one


class TestNashPAdaptiveTracksOpponentMix:
    def test_updates_internal_opponent_counts_across_calls(self):
        model = _FlatModel()
        pick = emad.make_picker_joint("nash-p-adaptive", model, tau=0.0)
        # Should not raise, and should return valid indices across repeated
        # calls that each update the tracked opponent-joint-action mixture.
        for other_joint in [0, 5, 5, 5, 10]:
            chosen = pick(1, _base_vec(), other_joint)
            assert 0 <= chosen < N_JOINT_ACTIONS


class TestIbrRequiresBothQModels:
    def test_raises_without_model1_or_model2(self):
        with pytest.raises(ValueError):
            emad.make_picker_joint("ibr", model_p=None, model1=_FlatModel(), model2=None)
        with pytest.raises(ValueError):
            emad.make_picker_joint("ibr", model_p=None, model1=None, model2=_FlatModel())


class TestIbrConvergesToSharedBestResponse:
    def test_both_players_converge_to_the_jointly_preferred_action(self):
        preferred = joint_action_index(1, 2)

        class PrefersOneJointAction(nn.Module):
            def forward(self, x):
                ego_attack = x[:, 74:74 + N_SKILLS].argmax(dim=1)
                ego_defend = x[:, 84:84 + N_DEFEND_POSITIONS].argmax(dim=1)
                pa, pd = joint_action_from_index(preferred)
                return ((ego_attack == pa) & (ego_defend == pd)).float().unsqueeze(1)

        model1 = PrefersOneJointAction()
        model2 = PrefersOneJointAction()
        pick1 = emad.make_picker_joint("ibr", model_p=None, model1=model1, model2=model2)
        chosen = pick1(1, _base_vec(), 0)
        assert chosen == preferred


class TestIbrQUsesEmpiricalOpponentMix:
    def test_raises_without_model1_or_model2(self):
        with pytest.raises(ValueError):
            emad.make_picker_joint("ibr-q", model_p=None, model1=_FlatModel(), model2=None)

    def test_returns_valid_joint_index(self):
        model1 = _FlatModel()
        model2 = _FlatModel()
        pick = emad.make_picker_joint("ibr-q", model_p=None, model1=model1, model2=model2)
        chosen = pick(1, _base_vec(), 7)
        assert 0 <= chosen < N_JOINT_ACTIONS


class TestBuildPhiTableJoint:
    def test_table_shape_is_15_by_15(self):
        model = _FlatModel()
        table = emad.build_phi_table_joint(_base_vec(), model)
        assert table.shape == (N_JOINT_ACTIONS, N_JOINT_ACTIONS)

    def test_table_values_reflect_the_model(self):
        preferred_attack = 3
        model = _PrefersAttackIndex(preferred_attack)
        table = emad.build_phi_table_joint(_base_vec(), model)
        for ego_j in range(N_JOINT_ACTIONS):
            ego_attack, _ = joint_action_from_index(ego_j)
            expected = 1.0 if ego_attack == preferred_attack else 0.0
            assert table[ego_j, :].eq(expected).all()


class TestSelfPlayStrategy:
    """'selfplay:<path>' samples the trained two-headed actor, 'selfplay-argmax:<path>'
    takes its modes; the state is built exactly as in self-play training."""

    def _checkpoint(self, tmp_path, attack_bias=None, defend_bias=None):
        from nash_skills.v2.selfplay_attack_defend import MetaPolicy
        policy = MetaPolicy()
        with torch.no_grad():
            for p in policy.parameters():
                p.zero_()
            if attack_bias is not None:
                policy.attack_head.bias[attack_bias] = 50.0
            if defend_bias is not None:
                policy.defend_head.bias[defend_bias] = 50.0
        from nash_skills.v2.selfplay_attack_defend import save_checkpoint
        path = str(tmp_path / "sp.pth")
        save_checkpoint(policy, path)
        return path

    def _obs_info(self):
        obs = np.zeros(116, dtype=np.float32)
        info = {"diff_pos": np.zeros(3), "diff_quat": np.zeros(4), "target": np.zeros(7),
                "diff_pos_opp": np.zeros(3), "diff_quat_opp": np.zeros(4), "target_opp": np.zeros(7)}
        return obs, info

    @pytest.mark.parametrize("kind", ["selfplay", "selfplay-argmax"])
    @pytest.mark.parametrize("player", [1, 2])
    def test_returns_the_policy_mode_when_it_is_certain(self, tmp_path, kind, player):
        path = self._checkpoint(tmp_path, attack_bias=3, defend_bias=2)
        pick = emad.make_picker_joint(f"{kind}:{path}", model_p=None)
        obs, info = self._obs_info()
        assert pick(player, obs, 0, info) == joint_action_index(3, 2)

    def test_argmax_is_deterministic_and_sampling_covers_the_support(self, tmp_path):
        path = self._checkpoint(tmp_path)
        obs, info = self._obs_info()
        argmax = emad.make_picker_joint(f"selfplay-argmax:{path}", model_p=None)
        assert len({argmax(1, obs, 0, info) for _ in range(20)}) == 1
        sampled = emad.make_picker_joint(f"selfplay:{path}", model_p=None)
        assert len({sampled(1, obs, 0, info) for _ in range(200)}) > 5


class TestSelfPlayCheckpointGuard:
    def test_checkpoint_from_the_old_encoder_is_refused(self, tmp_path):
        from nash_skills.v2.selfplay_attack_defend import MetaPolicy
        path = tmp_path / "old.pth"
        torch.save(MetaPolicy().state_dict(), path)
        with pytest.raises(ValueError, match="encoder"):
            emad.make_picker_joint(f"selfplay:{path}", model_p=None)


class TestSelfPlayPickerUsesTheActiveChoices:
    def test_own_preparation_comes_from_the_runner_not_the_last_pick(self, tmp_path, monkeypatch):
        import nash_skills.v2.state_encoder_attack_defend as enc
        from nash_skills.v2.selfplay_attack_defend import MetaPolicy, save_checkpoint
        path = str(tmp_path / "sp.pth")
        save_checkpoint(MetaPolicy(), path)
        calls = []
        real = enc.encode_ego
        monkeypatch.setattr(enc, "encode_ego", lambda o, i, d1, d2: (calls.append((d1, d2)), real(o, i, d1, d2))[1])
        pick = emad.make_picker_joint(f"selfplay:{path}", model_p=None)
        obs, info = TestSelfPlayStrategy()._obs_info()
        own = joint_action_index(0, 1)
        other = joint_action_index(2, 2)
        pick(1, obs, other, {**info, "current_joints": (own, other)})
        from nash_skills.defend_positions import defend_position_from_index
        assert calls[-1] == (defend_position_from_index(1), defend_position_from_index(2))
