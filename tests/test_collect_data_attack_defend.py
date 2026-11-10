"""Tests for nash_skills.v2.collect_data_attack_defend.

No MuJoCo needed -- a FakeEnv drives a REAL DefenderRoleTracker (not a
stub) from a scripted ball_x/ball_vel_x sequence, exactly mirroring how
the production SkillEnv._apply_defend_steering drives its own
env._role_tracker internally. This verifies the collector's integration
with the real role-detection logic, not just a mock substitute.

Covers exactly what the plan asked to verify: saved rally schema;
attack/defend labels round-trip; defending_player matches
DefenderRoleTracker's own resolution for a scripted sequence; states are
90-dim and tagged encoder="attack_defend"; no regression to the legacy
collect_data.py collector (run side by side in the same test session).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import Counter

import numpy as np
import pytest

from nash_skills.v2 import collect_data_attack_defend as cad
from nash_skills.v2 import collect_data as legacy_collect
from nash_skills.defend_role import DefenderRoleTracker
from nash_skills.skills import SKILL_NAMES, skill_index, N_SKILLS
from nash_skills.defend_positions import DEFEND_POSITION_NAMES


def _skill_value(skill: str) -> float:
    return skill_index(skill) / (N_SKILLS - 1)


def _info():
    return {
        "diff_pos": np.zeros(3), "diff_quat": np.zeros(4), "target": np.zeros(7),
        "diff_pos_opp": np.zeros(3), "diff_quat_opp": np.zeros(4), "target_opp": np.zeros(7),
    }


def _obs(ball_x: float, skill1: str, skill2: str) -> np.ndarray:
    obs = np.zeros(116, dtype=np.float32)
    obs[36] = ball_x
    obs[-2] = _skill_value(skill1)
    obs[-1] = _skill_value(skill2)
    return obs


class FakeEnv:
    """
    Drives a REAL DefenderRoleTracker from a scripted
    (ball_x, ball_vel_x, done, winner_info) sequence, one entry consumed
    per step() call -- exactly mirroring how SkillEnv._apply_defend_steering
    drives its own internal _role_tracker every step once
    set_defend_reactive_positions has been given real values for both
    players.

    Deliberately has NO set_defend_positions method (the hard-override
    mechanism) and NO set_defend_target_positions method (the static
    pose-redirect "Option B" mechanism) -- both confirmed net-harmful
    this session (mechanism_validation_matched.py,
    mechanism_validation_target_pose.py). If the collector under test
    ever regresses to calling either instead of
    set_defend_reactive_positions (the velocity/trajectory-informed
    mechanism that replaced them both), every test here fails loudly
    with AttributeError rather than silently training on harmful-
    mechanism data.
    """

    def __init__(self, script):
        self._script = list(script)
        self._step_idx = 0
        self._role_tracker = DefenderRoleTracker()
        self.skill1 = "left"
        self.skill2 = "right"
        self.defend_target1 = None
        self.defend_target2 = None

    def set_skills(self, skill1, skill2):
        self.skill1 = skill1
        self.skill2 = skill2

    def set_defend_reactive_positions(self, pos1, pos2):
        self.defend_target1 = pos1
        self.defend_target2 = pos2

    def reset(self):
        self._step_idx = 0
        self._role_tracker.reset()
        return _obs(0.0, self.skill1, self.skill2), _info()

    def step(self, _action):
        ball_x, ball_vel_x, done, winner = self._script[self._step_idx]
        self._step_idx += 1
        self._role_tracker.update(ball_x, ball_vel_x)
        info = _info()
        if winner is not None:
            info["winner"] = winner
        return _obs(ball_x, self.skill1, self.skill2), 0.0, done, False, info

    def close(self):
        pass


class FakeModel:
    def predict(self, _obs, deterministic=True):
        assert deterministic is True
        return np.zeros(9, dtype=np.float32), None


class FakePPO:
    @staticmethod
    def load(_path):
        return FakeModel()


def _patch(monkeypatch, script):
    env = FakeEnv(script)
    monkeypatch.setattr(cad, "SkillEnv", lambda **_kw: env)
    monkeypatch.setattr(cad, "PPO", FakePPO)
    return env


# A script with two decisions: ball heads to P2 (vel>threshold) for a
# few steps (latches defending_player=2), then crosses the net (ball_x
# crosses 1.5) heading back to P1 (latches/flips to defending_player=1)
# on a clean winning hit for ego at the end.
TWO_DECISION_SCRIPT = [
    (0.1, 4.5, False, None),
    (0.8, 4.5, False, None),
    (1.6, 4.5, False, None),   # crossed net -- still role=2 (heading to 2's side)
    (2.5, -4.0, False, None),  # hit back -- still role=2 until it crosses back
    (1.4, -4.0, False, None),  # crossed back -- role flips to 1
    (0.5, -4.0, False, None),
    (0.05, -4.0, True, "ego"),  # terminal, ego wins
]


class TestSchema:
    def test_saved_rally_has_exactly_the_expected_fields(self, tmp_path, monkeypatch):
        _patch(monkeypatch, TWO_DECISION_SCRIPT)
        rallies = cad.collect(
            target_decisions=1, output_path=str(tmp_path / "x.pkl"),
            ppo_path="fake", max_rallies=1, progress_every=0,
        )
        assert len(rallies) == 1
        expected_keys = {
            "states", "ego_attack_choices", "ego_defend_choices",
            "opp_attack_choices", "opp_defend_choices", "defending_player", "attacker",
            "winner", "encoder", "defend_mechanism", "decision_timing", "skill_profile",
            "steps", "ego_contacts", "opp_contacts",
            "ego_successes", "opp_successes",
        }
        assert set(rallies[0]) == expected_keys

    def test_encoder_tag_is_the_current_encoder_version(self, tmp_path, monkeypatch):
        from nash_skills.v2.state_encoder_attack_defend import STATE_ENCODER_NAME
        _patch(monkeypatch, TWO_DECISION_SCRIPT)
        rallies = cad.collect(
            target_decisions=1, output_path=str(tmp_path / "x.pkl"),
            ppo_path="fake", max_rallies=1, progress_every=0,
        )
        assert rallies[0]["encoder"] == STATE_ENCODER_NAME

    def test_states_are_90_dim(self, tmp_path, monkeypatch):
        _patch(monkeypatch, TWO_DECISION_SCRIPT)
        rallies = cad.collect(
            target_decisions=1, output_path=str(tmp_path / "x.pkl"),
            ppo_path="fake", max_rallies=1, progress_every=0,
        )
        for state in rallies[0]["states"]:
            assert state.shape == (cad.STATE_DIM_ATTACK_DEFEND,)
            assert cad.STATE_DIM_ATTACK_DEFEND == 90

    def test_winner_is_decoded_to_1_for_ego(self, tmp_path, monkeypatch):
        _patch(monkeypatch, TWO_DECISION_SCRIPT)
        rallies = cad.collect(
            target_decisions=1, output_path=str(tmp_path / "x.pkl"),
            ppo_path="fake", max_rallies=1, progress_every=0,
        )
        assert rallies[0]["winner"] == 1


class TestChoicesRoundTrip:
    def test_all_four_choice_lists_are_valid_names_and_equal_length(self, tmp_path, monkeypatch):
        _patch(monkeypatch, TWO_DECISION_SCRIPT)
        rallies = cad.collect(
            target_decisions=1, output_path=str(tmp_path / "x.pkl"),
            ppo_path="fake", max_rallies=1, progress_every=0, seed=3,
        )
        r = rallies[0]
        n = len(r["states"])
        assert len(r["ego_attack_choices"]) == len(r["ego_defend_choices"]) == n
        assert len(r["opp_attack_choices"]) == len(r["opp_defend_choices"]) == n
        assert len(r["defending_player"]) == n
        assert all(s in SKILL_NAMES for s in r["ego_attack_choices"])
        assert all(s in SKILL_NAMES for s in r["opp_attack_choices"])
        assert all(d in DEFEND_POSITION_NAMES for d in r["ego_defend_choices"])
        assert all(d in DEFEND_POSITION_NAMES for d in r["opp_defend_choices"])

    def test_two_decisions_recorded_for_the_scripted_sequence(self, tmp_path, monkeypatch):
        # The script has exactly 2 net crossings after the initial latch
        # (serve -> P2 defends, hit back -> P1 defends) -- so exactly 2
        # decisions should be recorded (the state right after each role
        # change), matching collect_data.py's "record at the crossing"
        # convention.
        _patch(monkeypatch, TWO_DECISION_SCRIPT)
        rallies = cad.collect(
            target_decisions=1, output_path=str(tmp_path / "x.pkl"),
            ppo_path="fake", max_rallies=1, progress_every=0,
        )
        assert len(rallies[0]["states"]) == 2


class TestFixedPerRallySampling:
    """
    Regression guard: per-decision resampling (the original design)
    was found to collapse collection throughput to near zero under the
    reactive mechanism (scripts/diagnose_collector_stall.py -- 10/10
    traced rallies hit the full step cap in a sustained, non-
    terminating pattern). The fix samples ONCE per rally; this must not
    regress back to resampling at each decision.
    """

    def test_all_decisions_in_one_rally_share_the_same_choices(self, tmp_path, monkeypatch):
        _patch(monkeypatch, TWO_DECISION_SCRIPT)
        rallies = cad.collect(
            target_decisions=1, output_path=str(tmp_path / "x.pkl"),
            ppo_path="fake", max_rallies=1, progress_every=0, seed=5,
        )
        r = rallies[0]
        assert len(set(r["ego_attack_choices"])) == 1
        assert len(set(r["ego_defend_choices"])) == 1
        assert len(set(r["opp_attack_choices"])) == 1
        assert len(set(r["opp_defend_choices"])) == 1

    def test_env_set_defend_reactive_positions_called_exactly_once_per_rally(self, tmp_path, monkeypatch):
        env = _patch(monkeypatch, TWO_DECISION_SCRIPT)
        calls = []
        original = env.set_defend_reactive_positions
        env.set_defend_reactive_positions = lambda p1, p2: (calls.append((p1, p2)), original(p1, p2))
        cad.collect(
            target_decisions=1, output_path=str(tmp_path / "x.pkl"),
            ppo_path="fake", max_rallies=1, progress_every=0, seed=5,
        )
        assert len(calls) == 1


class TestDefendingPlayerMatchesRoleTracker:
    def test_credited_roles_match_an_independent_role_tracker_replay(self, tmp_path, monkeypatch):
        # Replay the same script through a fresh, independent
        # DefenderRoleTracker and confirm the collector's credited
        # defending_player at each decision matches which role was
        # active during the segment that just concluded.
        tracker = DefenderRoleTracker()
        prev = tracker.defending_player
        expected_credits = []
        for ball_x, ball_vel_x, _done, _winner in TWO_DECISION_SCRIPT:
            role = tracker.update(ball_x, ball_vel_x)
            if role is not None and role != prev:
                expected_credits.append(prev if prev is not None else role)
            prev = role

        _patch(monkeypatch, TWO_DECISION_SCRIPT)
        rallies = cad.collect(
            target_decisions=1, output_path=str(tmp_path / "x.pkl"),
            ppo_path="fake", max_rallies=1, progress_every=0,
        )
        assert rallies[0]["defending_player"] == expected_credits
        assert expected_credits == [2, 2]  # first decision: P2 was defending
                                            # (serve->P2's side); second
                                            # decision: still credited to 2
                                            # since prev_defending_player was
                                            # 2 right before the flip to 1


class TestNoRegressionToLegacyCollector:
    def test_legacy_collect_data_still_produces_76_dim_legacy_tagged_states(self, tmp_path, monkeypatch):
        class LegacyFakeEnv:
            def __init__(self, proc_id, history, reset_mode, skill_profile, gantry_speed_scale):
                self.skill1 = "left"
                self.skill2 = "right"

            def set_skills(self, skill1, skill2):
                self.skill1 = skill1
                self.skill2 = skill2

            def reset(self):
                return _obs(1.0, self.skill1, self.skill2), _info()

            def step(self, _action):
                info = _info()
                info["winner"] = "ego"
                return _obs(2.0, self.skill1, self.skill2), 0.0, True, False, info

            def close(self):
                pass

        monkeypatch.setattr(legacy_collect, "SkillEnv", LegacyFakeEnv)
        monkeypatch.setattr(legacy_collect, "PPO", FakePPO)
        monkeypatch.setattr(legacy_collect, "detect_winner", lambda _raw, done, info=None: 1 if done else 0)

        rallies = legacy_collect.collect(
            target_rallies=1, output_path=str(tmp_path / "legacy.pkl"), ppo_path="fake",
            max_steps_per_episode=5, max_attempts_per_pair=1, progress_every=0,
            mode="fixed_random", fixed_player=1, fixed_skill="left",
        )
        assert rallies[0]["encoder"] == "legacy"
        assert rallies[0]["states"][0].shape == (76,)


class TestUsesFixedReactivePoseMechanism:
    """
    Design B's own mechanism-validation track found both the hard-override
    mechanism (SkillEnv.set_defend_positions, mechanism_validation_matched.py)
    and the static pose-redirect "Option B" mechanism
    (set_defend_target_positions, mechanism_validation_target_pose.py)
    net-harmful, and built set_defend_reactive_positions (the velocity/
    trajectory-informed channel, mechanism_validation_reactive_pose.py)
    specifically to replace them both. This collector must use the fixed
    mechanism -- training data collected under either harmful one would
    bake the wrong dynamics into every downstream model. FakeEnv
    deliberately has neither older method at all, so a regression here
    fails loudly (AttributeError) rather than silently.
    """

    def test_collector_calls_set_defend_reactive_positions_with_sampled_names(self, tmp_path, monkeypatch):
        env = _patch(monkeypatch, TWO_DECISION_SCRIPT)
        cad.collect(
            target_decisions=1, output_path=str(tmp_path / "x.pkl"),
            ppo_path="fake", max_rallies=1, progress_every=0, seed=7,
        )
        assert env.defend_target1 in DEFEND_POSITION_NAMES
        assert env.defend_target2 in DEFEND_POSITION_NAMES


class TestDefendMechanismSwitch:
    """--defend-mechanism picks the receiver-bias (reactive) or the paper's
    hitter-preparation mechanism; every setter call must go to the chosen one."""

    def _record_both(self, env):
        reactive, prep = [], []
        original = env.set_defend_reactive_positions
        env.set_defend_reactive_positions = lambda p1, p2: (reactive.append((p1, p2)), original(p1, p2))
        env.set_defend_preparation_positions = lambda p1, p2: prep.append((p1, p2))
        return reactive, prep

    def test_preparation_routes_every_call_to_the_preparation_setter(self, tmp_path, monkeypatch):
        env = _patch(monkeypatch, TWO_DECISION_SCRIPT)
        reactive, prep = self._record_both(env)
        rallies = cad.collect(
            target_decisions=1, output_path=str(tmp_path / "x.pkl"), ppo_path="fake",
            max_rallies=1, progress_every=0, seed=3, sampling="mixed",
            defend_mechanism="preparation",
        )
        assert reactive == []
        assert len(prep) == 1 + len(rallies[0]["states"])
        assert all(p in DEFEND_POSITION_NAMES for pair in prep for p in pair)
        assert rallies[0]["defend_mechanism"] == "preparation"

    def test_default_is_reactive_and_is_recorded_on_each_rally(self, tmp_path, monkeypatch):
        _patch(monkeypatch, TWO_DECISION_SCRIPT)
        rallies = cad.collect(
            target_decisions=1, output_path=str(tmp_path / "x.pkl"), ppo_path="fake",
            max_rallies=1, progress_every=0, seed=3,
        )
        assert rallies[0]["defend_mechanism"] == "reactive"

    def test_unknown_mechanism_raises(self, tmp_path, monkeypatch):
        _patch(monkeypatch, TWO_DECISION_SCRIPT)
        with pytest.raises(ValueError):
            cad.collect(
                target_decisions=1, output_path=str(tmp_path / "x.pkl"), ppo_path="fake",
                max_rallies=1, progress_every=0, defend_mechanism="hard_override",
            )


class TestGridJointActionSampler:
    """
    _sample_joint_action's pure-random draw left the pilot run's coverage
    uneven (116/225 combos hit, mean 1.3/combo -- see module docstring).
    GridJointActionSampler instead round-robins the full 225-combo space,
    reshuffling only once exhausted, so a collection run targeting N
    decisions gets every combination hit floor(N/225) or ceil(N/225)
    times instead of Poisson-ish random variance.
    """

    def _n_total(self):
        return len(SKILL_NAMES) * len(DEFEND_POSITION_NAMES) * len(SKILL_NAMES) * len(DEFEND_POSITION_NAMES)

    def test_covers_all_combinations_before_any_repeat(self):
        sampler = cad.GridJointActionSampler(np.random.default_rng(0))
        n_total = self._n_total()
        assert n_total == 225
        drawn = [sampler.sample() for _ in range(n_total)]
        assert len(set(drawn)) == n_total

    def test_reshuffles_and_continues_seamlessly_after_exhaustion(self):
        sampler = cad.GridJointActionSampler(np.random.default_rng(1))
        n_total = self._n_total()
        for _ in range(n_total):
            sampler.sample()
        extra = sampler.sample()
        assert extra in sampler.all_combos

    def test_balanced_over_multiple_full_cycles(self):
        sampler = cad.GridJointActionSampler(np.random.default_rng(2))
        n_total = self._n_total()
        n_cycles = 3
        counts = Counter(sampler.sample() for _ in range(n_total * n_cycles))
        assert len(counts) == n_total
        assert set(counts.values()) == {n_cycles}

    def test_different_seeds_give_different_orderings(self):
        s1 = cad.GridJointActionSampler(np.random.default_rng(10))
        s2 = cad.GridJointActionSampler(np.random.default_rng(20))
        seq1 = [s1.sample() for _ in range(10)]
        seq2 = [s2.sample() for _ in range(10)]
        assert seq1 != seq2

    def test_every_combo_is_a_valid_name_tuple(self):
        sampler = cad.GridJointActionSampler(np.random.default_rng(3))
        for combo in sampler.all_combos:
            ea, ed, oa, od = combo
            assert ea in SKILL_NAMES and oa in SKILL_NAMES
            assert ed in DEFEND_POSITION_NAMES and od in DEFEND_POSITION_NAMES


class TestSamplingModeSelection:
    def test_default_sampling_is_random_backward_compatible(self, tmp_path, monkeypatch):
        _patch(monkeypatch, TWO_DECISION_SCRIPT)
        rallies = cad.collect(
            target_decisions=1, output_path=str(tmp_path / "x.pkl"),
            ppo_path="fake", max_rallies=1, progress_every=0,
        )
        assert len(rallies) == 1

    def test_grid_sampling_mode_runs_without_crashing(self, tmp_path, monkeypatch):
        _patch(monkeypatch, TWO_DECISION_SCRIPT)
        rallies = cad.collect(
            target_decisions=1, output_path=str(tmp_path / "x.pkl"),
            ppo_path="fake", max_rallies=1, progress_every=0, sampling="grid",
        )
        assert len(rallies) == 1

    def test_unknown_sampling_mode_raises(self, tmp_path, monkeypatch):
        _patch(monkeypatch, TWO_DECISION_SCRIPT)
        with pytest.raises(ValueError):
            cad.collect(
                target_decisions=1, output_path=str(tmp_path / "x.pkl"),
                ppo_path="fake", max_rallies=1, progress_every=0, sampling="bogus",
            )


class TestMixedSampling:
    """
    Paper Sec 4.1: Phi(x, theta) is learned from continuations played under
    a MIXED theta held fixed for the rally ("include deterministic choices
    as well as mixed strategies so that the data cover both the fifteen
    fixed target-preparation pairings and randomized play"). In
    sampling="mixed", each rally gets one theta per player; actions are
    DRAWN from theta at every decision; the recorded state carries theta
    itself in the four action slots, not the realized action's one-hot.
    """

    def _collect(self, tmp_path, monkeypatch, **kw):
        _patch(monkeypatch, TWO_DECISION_SCRIPT)
        return cad.collect(target_decisions=1, output_path=str(tmp_path / "x.pkl"),
                           ppo_path="fake", max_rallies=1, progress_every=0,
                           sampling="mixed", **kw)

    def test_runs_and_records_theta_per_rally(self, tmp_path, monkeypatch):
        r = self._collect(tmp_path, monkeypatch, seed=3)[0]
        for key in ("ego_theta", "opp_theta"):
            wA, wD = r[key]
            assert len(wA) == N_SKILLS and len(wD) == 3
            assert sum(wA) == pytest.approx(1.0) and sum(wD) == pytest.approx(1.0)
        assert r["theta_mode"] == "mixed"

    def test_state_action_slots_hold_theta_not_the_realized_action(self, tmp_path, monkeypatch):
        r = self._collect(tmp_path, monkeypatch, seed=3, p_vertex=0.0)[0]
        ego_wA, ego_wD = r["ego_theta"]
        opp_wA, opp_wD = r["opp_theta"]
        for state in r["states"]:
            np.testing.assert_allclose(state[74:79], ego_wA, atol=1e-6)
            np.testing.assert_allclose(state[79:84], opp_wA, atol=1e-6)
            np.testing.assert_allclose(state[84:87], ego_wD, atol=1e-6)
            np.testing.assert_allclose(state[87:90], opp_wD, atol=1e-6)
        assert max(ego_wA) < 1.0  # p_vertex=0 -> interior, so not a one-hot

    def test_p_vertex_one_gives_pure_pairings(self, tmp_path, monkeypatch):
        r = self._collect(tmp_path, monkeypatch, seed=4, p_vertex=1.0)[0]
        assert max(r["ego_theta"][0]) == 1.0 and max(r["opp_theta"][1]) == 1.0

    def test_realized_actions_are_still_valid_names(self, tmp_path, monkeypatch):
        r = self._collect(tmp_path, monkeypatch, seed=5)[0]
        assert all(s in SKILL_NAMES for s in r["ego_attack_choices"] + r["opp_attack_choices"])
        assert all(d in DEFEND_POSITION_NAMES for d in r["ego_defend_choices"] + r["opp_defend_choices"])

    def test_default_modes_schema_is_unchanged(self, tmp_path, monkeypatch):
        # Mixed-only keys must NOT leak into the default random mode's
        # schema (TestSchema pins that exact key set).
        _patch(monkeypatch, TWO_DECISION_SCRIPT)
        r = cad.collect(target_decisions=1, output_path=str(tmp_path / "x.pkl"),
                        ppo_path="fake", max_rallies=1, progress_every=0)[0]
        assert "ego_theta" not in r and "theta_mode" not in r


class TestResetTargetsFlag:
    def _patch_capturing(self, monkeypatch):
        env = FakeEnv(TWO_DECISION_SCRIPT)
        captured = {}
        monkeypatch.setattr(cad, "SkillEnv", lambda **kw: (captured.update(kw), env)[1])
        monkeypatch.setattr(cad, "PPO", FakePPO)
        return captured

    @pytest.mark.parametrize("flag", [True, False])
    def test_reset_targets_is_passed_to_the_env(self, tmp_path, monkeypatch, flag):
        captured = self._patch_capturing(monkeypatch)
        cad.collect(target_decisions=1, output_path=str(tmp_path / "x.pkl"), ppo_path="fake",
                    max_rallies=1, progress_every=0, reset_targets=flag)
        assert captured["reset_targets"] is flag
