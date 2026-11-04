"""Tests for nash_skills.v2.selfplay_attack_defend's pure, testable pieces.

TDD: written before implementation.

The full training loop (play_one_rally, train) requires a real MuJoCo
env and a real PPO checkpoint, like selfplay_5skill.py's own training
loop -- untested by pytest here, same precedent as that script (zero
existing pytest coverage), verified instead by a direct smoke run
(scripts/smoke_selfplay_attack_defend.sh or an equivalent manual
invocation). This file covers what IS cheaply, meaningfully testable
without MuJoCo: the two-headed policy's architecture/shapes, and the
fixed-joint-opponent's sampling behavior.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from nash_skills.v2.selfplay_attack_defend import MetaPolicy, FixedJointPolicy
from nash_skills.v2.state_encoder_attack_defend import STATE_DIM_ATTACK_DEFEND
from nash_skills.skills import N_SKILLS, SKILL_NAMES
from nash_skills.defend_positions import N_DEFEND_POSITIONS, DEFEND_POSITION_NAMES


class TestMetaPolicyShapes:
    """
    UPDATE: MetaPolicy gained a value_head this round (REINFORCE ->
    actor-critic -> PPO). forward() now returns a 3-tuple
    (attack_logits, defend_logits, value) instead of 2 -- the value is
    a per-state baseline replacing the old global-scalar EMA, and is
    also what PPO's GAE computation and critic loss need.
    """

    def test_forward_returns_logits_and_value_with_expected_widths(self):
        policy = MetaPolicy()
        x = torch.zeros(1, STATE_DIM_ATTACK_DEFEND)
        attack_logits, defend_logits, value = policy(x)
        assert attack_logits.shape == (1, N_SKILLS)
        assert defend_logits.shape == (1, N_DEFEND_POSITIONS)
        assert value.shape == (1,)

    def test_forward_batches_correctly(self):
        policy = MetaPolicy()
        x = torch.zeros(5, STATE_DIM_ATTACK_DEFEND)
        attack_logits, defend_logits, value = policy(x)
        assert attack_logits.shape == (5, N_SKILLS)
        assert defend_logits.shape == (5, N_DEFEND_POSITIONS)
        assert value.shape == (5,)

    def test_the_two_heads_share_a_trunk_not_independent_networks(self):
        # Changing only the defend head's weights must not change the
        # attack head's output for the same input -- but both must flow
        # through the same trunk (sanity: trunk params exist and are
        # shared, not duplicated per head).
        policy = MetaPolicy()
        trunk_params = list(policy.trunk.parameters())
        assert len(trunk_params) > 0
        assert policy.attack_head.in_features == policy.defend_head.in_features
        assert policy.value_head.in_features == policy.attack_head.in_features
        assert policy.value_head.out_features == 1

    def test_sample_returns_valid_indices_and_probability_vectors(self):
        policy = MetaPolicy()
        state = np.zeros(STATE_DIM_ATTACK_DEFEND, dtype=np.float32)
        attack_idx, defend_idx, attack_probs, defend_probs = policy.sample(state, torch.device("cpu"))
        assert 0 <= attack_idx < N_SKILLS
        assert 0 <= defend_idx < N_DEFEND_POSITIONS
        assert attack_probs.shape == (N_SKILLS,)
        assert defend_probs.shape == (N_DEFEND_POSITIONS,)
        assert attack_probs.sum() == pytest.approx(1.0, abs=1e-5)
        assert defend_probs.sum() == pytest.approx(1.0, abs=1e-5)


class TestFixedJointPolicy:
    def test_fixed_attack_and_fixed_defend_always_returns_same_pair(self):
        policy = FixedJointPolicy("left", "center")
        for _ in range(20):
            attack_idx, defend_idx, _ap, _dp = policy.sample(None, None)
            assert attack_idx == SKILL_NAMES.index("left")
            assert defend_idx == DEFEND_POSITION_NAMES.index("center")

    def test_random_attack_varies_fixed_defend_does_not(self):
        policy = FixedJointPolicy("random", "right")
        attack_draws = set()
        for _ in range(50):
            attack_idx, defend_idx, _ap, _dp = policy.sample(None, None)
            attack_draws.add(attack_idx)
            assert defend_idx == DEFEND_POSITION_NAMES.index("right")
        assert len(attack_draws) > 1

    def test_random_defend_varies_fixed_attack_does_not(self):
        policy = FixedJointPolicy("right", "random")
        defend_draws = set()
        for _ in range(50):
            attack_idx, defend_idx, _ap, _dp = policy.sample(None, None)
            assert attack_idx == SKILL_NAMES.index("right")
            defend_draws.add(defend_idx)
        assert len(defend_draws) > 1

    def test_both_random_produces_valid_indices(self):
        policy = FixedJointPolicy("random", "random")
        for _ in range(20):
            attack_idx, defend_idx, _ap, _dp = policy.sample(None, None)
            assert 0 <= attack_idx < N_SKILLS
            assert 0 <= defend_idx < N_DEFEND_POSITIONS

    def test_unknown_attack_mode_raises(self):
        with pytest.raises(ValueError):
            FixedJointPolicy("not-a-skill", "center")

    def test_unknown_defend_mode_raises(self):
        with pytest.raises(ValueError):
            FixedJointPolicy("left", "not-a-position")

    def test_uniform_probability_vectors_have_correct_shape_and_sum(self):
        policy = FixedJointPolicy("left", "center")
        _ai, _di, attack_probs, defend_probs = policy.sample(None, None)
        assert attack_probs.shape == (N_SKILLS,)
        assert defend_probs.shape == (N_DEFEND_POSITIONS,)
        assert attack_probs.sum() == pytest.approx(1.0, abs=1e-5)
        assert defend_probs.sum() == pytest.approx(1.0, abs=1e-5)


class TestComputeGAE:
    """
    compute_gae(rewards, values, gamma, gae_lambda) -> (advantages, returns).
    rewards[t]=0 for every non-terminal decision in a rally, the
    terminal reward only at the last index (this game has no
    intermediate per-decision reward) -- values[t] is the critic's
    OWN estimate at that decision's state, from collection time
    (always treated as fixed here, per standard GAE/PPO practice: the
    advantage estimator never uses the policy currently being updated).
    """

    def test_single_decision_reduces_to_reward_minus_value(self):
        from nash_skills.v2.selfplay_attack_defend import compute_gae
        advantages, returns = compute_gae(rewards=[1.0], values=[0.3], gamma=0.99, gae_lambda=0.95)
        assert advantages == pytest.approx([0.7])
        assert returns == pytest.approx([1.0])

    def test_two_decisions_matches_hand_computed_recursion(self):
        from nash_skills.v2.selfplay_attack_defend import compute_gae
        rewards = [0.0, 1.0]
        values = [0.2, 0.5]
        gamma, lam = 0.99, 0.95
        # t=1 (terminal): delta1 = r1 + gamma*0 - v1 = 1.0 - 0.5 = 0.5; gae1 = 0.5
        delta1 = rewards[1] - values[1]
        # t=0: delta0 = r0 + gamma*v1 - v0; gae0 = delta0 + gamma*lam*gae1
        delta0 = rewards[0] + gamma * values[1] - values[0]
        expected_adv1 = delta1
        expected_adv0 = delta0 + gamma * lam * expected_adv1
        advantages, returns = compute_gae(rewards, values, gamma, lam)
        assert advantages == pytest.approx([expected_adv0, expected_adv1])
        assert returns == pytest.approx([expected_adv0 + values[0], expected_adv1 + values[1]])

    def test_zero_lambda_reduces_to_one_step_td_error(self):
        # gae_lambda=0 means no multi-step lookahead -- each advantage
        # IS just that step's own TD residual.
        from nash_skills.v2.selfplay_attack_defend import compute_gae
        rewards = [0.0, 0.0, 2.0]
        values = [0.1, 0.4, 0.6]
        gamma = 0.9
        advantages, _returns = compute_gae(rewards, values, gamma, gae_lambda=0.0)
        expected = [
            rewards[0] + gamma * values[1] - values[0],
            rewards[1] + gamma * values[2] - values[1],
            rewards[2] - values[2],
        ]
        assert advantages == pytest.approx(expected)

    def test_returns_equal_advantages_plus_values_elementwise(self):
        from nash_skills.v2.selfplay_attack_defend import compute_gae
        rewards = [0.0, 0.0, -1.0]
        values = [0.5, -0.2, 0.1]
        advantages, returns = compute_gae(rewards, values, gamma=0.95, gae_lambda=0.9)
        for a, v, r in zip(advantages, values, returns):
            assert r == pytest.approx(a + v)

    def test_nonzero_bootstrap_value_replaces_zero_at_the_final_step(self):
        # Truncation case: paper's own framing ("bootstrapped from its
        # critic rather than treated as a lost point") -- the terminal
        # reward is 0 (no artificial penalty), but the "value beyond
        # this point" used at the last step is the critic's own
        # estimate at the cutoff state, not 0 (which would implicitly
        # claim the game is definitely over with nothing more to lose
        # or gain, exactly what a REAL done=True terminal means and a
        # truncation does not).
        from nash_skills.v2.selfplay_attack_defend import compute_gae
        rewards = [0.0]
        values = [0.3]
        adv_terminal, _ret_terminal = compute_gae(rewards, values, gamma=0.99, gae_lambda=0.95,
                                                    bootstrap_value=0.0)
        adv_bootstrap, _ret_bootstrap = compute_gae(rewards, values, gamma=0.99, gae_lambda=0.95,
                                                      bootstrap_value=0.6)
        # With bootstrap=0: delta = 0 + 0.99*0 - 0.3 = -0.3 (reads as a
        # pure loss in expectation). With bootstrap=0.6: delta = 0 +
        # 0.99*0.6 - 0.3 = +0.294 (the game was expected to continue
        # favorably, so truncating it is not treated as a loss at all).
        assert adv_terminal[0] == pytest.approx(0.0 + 0.99 * 0.0 - 0.3)
        assert adv_bootstrap[0] == pytest.approx(0.0 + 0.99 * 0.6 - 0.3)
        assert adv_bootstrap[0] != pytest.approx(adv_terminal[0])

    def test_bootstrap_value_defaults_to_zero_backward_compatible(self):
        from nash_skills.v2.selfplay_attack_defend import compute_gae
        a1, r1 = compute_gae([1.0], [0.3], gamma=0.99, gae_lambda=0.95)
        a2, r2 = compute_gae([1.0], [0.3], gamma=0.99, gae_lambda=0.95, bootstrap_value=0.0)
        assert a1 == pytest.approx(a2)
        assert r1 == pytest.approx(r2)


class TestPPOClippedLoss:
    def test_unclipped_when_ratio_is_one_matches_plain_policy_gradient(self):
        from nash_skills.v2.selfplay_attack_defend import ppo_clipped_loss
        new_lp = torch.tensor([0.0, 0.0])
        old_lp = torch.tensor([0.0, 0.0])  # ratio = exp(0) = 1 exactly
        adv = torch.tensor([2.0, -3.0])
        loss = ppo_clipped_loss(new_lp, old_lp, adv, clip_epsilon=0.2)
        assert loss.item() == pytest.approx(-adv.mean().item())

    def test_large_favorable_ratio_gets_clipped_for_positive_advantage(self):
        from nash_skills.v2.selfplay_attack_defend import ppo_clipped_loss
        # new >> old -> ratio far above 1+eps; with positive advantage,
        # the clipped branch (smaller) must be the one selected (min).
        new_lp = torch.tensor([5.0])
        old_lp = torch.tensor([0.0])
        adv = torch.tensor([1.0])
        loss = ppo_clipped_loss(new_lp, old_lp, adv, clip_epsilon=0.2)
        assert loss.item() == pytest.approx(-1.2)  # -(1+eps)*adv

    def test_large_unfavorable_ratio_is_not_clipped_for_positive_advantage(self):
        # PPO's clip only removes the INCENTIVE to move further in an
        # already-exploited direction -- it does not clip away the
        # penalty for a bad (low-ratio) move with positive advantage.
        from nash_skills.v2.selfplay_attack_defend import ppo_clipped_loss
        new_lp = torch.tensor([-5.0])
        old_lp = torch.tensor([0.0])
        adv = torch.tensor([1.0])
        loss = ppo_clipped_loss(new_lp, old_lp, adv, clip_epsilon=0.2)
        ratio = torch.exp(torch.tensor(-5.0)).item()
        assert loss.item() == pytest.approx(-ratio * 1.0, abs=1e-4)


class TestParseBaselinePoolEntry:
    def test_parses_attack_colon_defend(self):
        from nash_skills.v2.selfplay_attack_defend import parse_baseline_pool_entry
        assert parse_baseline_pool_entry("left:center") == ("left", "center")
        assert parse_baseline_pool_entry("random:random") == ("random", "random")

    def test_missing_colon_raises(self):
        from nash_skills.v2.selfplay_attack_defend import parse_baseline_pool_entry
        with pytest.raises(ValueError):
            parse_baseline_pool_entry("left")


class _ScriptedRallyEnv:
    """Ball crosses the net once, then ego wins; records defend-setter calls."""

    BALL_XS = [0.5, 1.0, 1.6, 2.4, 2.9]

    def __init__(self):
        self.reactive, self.prep = [], []
        self._i = 0

    def set_skills(self, _s1, _s2):
        pass

    def set_defend_reactive_positions(self, p1, p2):
        self.reactive.append((p1, p2))

    def set_defend_preparation_positions(self, p1, p2):
        self.prep.append((p1, p2))

    @staticmethod
    def _obs(ball_x):
        obs = np.zeros(116, dtype=np.float32)
        obs[36] = ball_x
        return obs

    @staticmethod
    def _info():
        return {
            "diff_pos": np.zeros(3), "diff_quat": np.zeros(4), "target": np.zeros(7),
            "diff_pos_opp": np.zeros(3), "diff_quat_opp": np.zeros(4), "target_opp": np.zeros(7),
        }

    def reset(self):
        self._i = 0
        return self._obs(0.2), self._info()

    def step(self, _action):
        ball_x = self.BALL_XS[self._i]
        self._i += 1
        done = self._i == len(self.BALL_XS)
        info = self._info()
        if done:
            info["winner"] = "ego"
        return self._obs(ball_x), 0.0, done, False, info


class _ZeroPPO:
    def predict(self, _obs, deterministic=True):
        return np.zeros(9, dtype=np.float32), None


class TestPlayOneRallyDefendMechanism:
    def _play(self, **kwargs):
        from nash_skills.v2.selfplay_attack_defend import play_one_rally
        env = _ScriptedRallyEnv()
        policy = MetaPolicy()
        rally = play_one_rally(env, _ZeroPPO(), policy, policy, torch.device("cpu"), **kwargs)
        return env, rally

    def test_preparation_routes_every_call_to_the_preparation_setter(self):
        env, rally = self._play(defend_mechanism="preparation")
        assert env.reactive == []
        assert len(env.prep) == 1 + rally["crossings"] == 2
        assert all(p in DEFEND_POSITION_NAMES for pair in env.prep for p in pair)

    def test_default_stays_reactive(self):
        env, _rally = self._play()
        assert env.prep == []
        assert len(env.reactive) == 2

    def test_unknown_mechanism_raises(self):
        with pytest.raises(ValueError):
            self._play(defend_mechanism="hard_override")


class TestCheckpointEncoderTag:
    def test_saved_checkpoint_passes_the_check(self, tmp_path):
        from nash_skills.v2.selfplay_attack_defend import save_checkpoint, check_checkpoint_encoder
        path = str(tmp_path / "p.pth")
        save_checkpoint(MetaPolicy(), path)
        check_checkpoint_encoder(path)

    def test_checkpoint_without_a_tag_is_rejected(self, tmp_path):
        from nash_skills.v2.selfplay_attack_defend import check_checkpoint_encoder
        path = str(tmp_path / "old.pth")
        torch.save(MetaPolicy().state_dict(), path)
        with pytest.raises(ValueError, match="encoder"):
            check_checkpoint_encoder(path)
