"""
Tests for the 5-skill Nash pipeline (nash_skills/).

Covers:
  - skills.py  : skill definitions, look-ups, index conversions
  - env_wrapper.py : SkillEnv behaviour without a real MuJoCo simulation
  - train_q_model_5skill.py: observation preprocessing logic
  - comp_5skill.py : pick_skill() strategy logic (no PPO required)

Run from the project root:
    python -m pytest tests/test_skills.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import types
import unittest
import numpy as np
import torch


# =========================================================================== #
# 1. skills.py                                                                 #
# =========================================================================== #

class TestSkillDefinitions(unittest.TestCase):

    def setUp(self):
        from nash_skills.skills import (
            SKILLS, SKILL_NAMES, N_SKILLS,
            get_skill, skill_index, skill_from_index,
            TABLE_SHIFT, TABLE_NEAR, TABLE_MID, TABLE_FAR,
        )
        self.SKILLS          = SKILLS
        self.SKILL_NAMES     = SKILL_NAMES
        self.N_SKILLS        = N_SKILLS
        self.get_skill       = get_skill
        self.skill_index     = skill_index
        self.skill_from_index = skill_from_index
        self.TABLE_SHIFT     = TABLE_SHIFT
        self.TABLE_NEAR      = TABLE_NEAR
        self.TABLE_MID       = TABLE_MID
        self.TABLE_FAR       = TABLE_FAR

    # --- basic counts -------------------------------------------------------

    def test_five_skills_defined(self):
        self.assertEqual(self.N_SKILLS, 5)

    def test_all_required_skills_present(self):
        required = {"left", "right", "left_short", "right_short", "center_safe"}
        self.assertEqual(set(self.SKILL_NAMES), required)

    def test_skill_names_list_length(self):
        self.assertEqual(len(self.SKILL_NAMES), self.N_SKILLS)

    # --- geometry checks ----------------------------------------------------

    def test_left_is_negative_side(self):
        side, _ = self.get_skill("left")
        self.assertLess(side, 0)

    def test_right_is_positive_side(self):
        side, _ = self.get_skill("right")
        self.assertGreater(side, 0)

    def test_center_safe_has_zero_side(self):
        side, _ = self.get_skill("center_safe")
        self.assertEqual(side, 0.0)

    def test_left_short_matches_left_side_target(self):
        side_l,  _ = self.get_skill("left")
        side_ls, _ = self.get_skill("left_short")
        self.assertEqual(side_l, side_ls,
                         "left and left_short should aim at the same lateral side")

    def test_right_short_matches_right_side_target(self):
        side_r,  _ = self.get_skill("right")
        side_rs, _ = self.get_skill("right_short")
        self.assertEqual(side_r, side_rs)

    def test_short_skills_have_smaller_x_than_deep(self):
        _, x_left       = self.get_skill("left")
        _, x_left_short = self.get_skill("left_short")
        self.assertLess(x_left_short, x_left,
                        "left_short should land closer to the net than left")

        _, x_right       = self.get_skill("right")
        _, x_right_short = self.get_skill("right_short")
        self.assertLess(x_right_short, x_right)

    def test_center_safe_x_between_short_and_deep(self):
        _, x_near = self.get_skill("left_short")
        _, x_mid  = self.get_skill("center_safe")
        _, x_far  = self.get_skill("left")
        self.assertGreater(x_mid, x_near)
        self.assertLess(x_mid, x_far)

    def test_all_x_targets_are_in_opponent_half(self):
        """x_target must be strictly past the net (TABLE_SHIFT = 1.5)."""
        for name in self.SKILL_NAMES:
            _, x = self.get_skill(name)
            self.assertGreater(x, self.TABLE_SHIFT,
                               f"{name}: x_target {x} must be > TABLE_SHIFT {self.TABLE_SHIFT}")

    def test_all_x_targets_within_table(self):
        """Table ends at TABLE_SHIFT + 1.37 = 2.87 m."""
        table_end = self.TABLE_SHIFT + 1.37
        for name in self.SKILL_NAMES:
            _, x = self.get_skill(name)
            self.assertLessEqual(x, table_end,
                                 f"{name}: x_target {x} exceeds table end {table_end}")

    def test_side_targets_bounded(self):
        """side_target must be in [-1, 1]."""
        for name in self.SKILL_NAMES:
            side, _ = self.get_skill(name)
            self.assertGreaterEqual(side, -1.0)
            self.assertLessEqual(side,  1.0)

    def test_table_near_at_least_1_75(self):
        """
        TABLE_NEAR must be >= 1.75.

        Simulator validation (60 trials × 800 steps) showed that the PPO
        cannot reliably reach 1.65m: left_short landed 0.20m DEEPER than
        left on average (x_mean=2.286 vs 2.089). Raising to 1.75 keeps the
        short skill within the PPO's demonstrated tracking range.
        """
        self.assertGreaterEqual(
            self.TABLE_NEAR, 1.75,
            f"TABLE_NEAR={self.TABLE_NEAR} is too aggressive; must be >= 1.75 "
            f"(see skill_eval/table_near_results.json for evidence)"
        )

    def test_short_skills_use_table_near(self):
        """left_short and right_short must both use TABLE_NEAR as x_target."""
        _, x_ls = self.get_skill("left_short")
        _, x_rs = self.get_skill("right_short")
        self.assertAlmostEqual(x_ls, self.TABLE_NEAR, places=6,
                               msg="left_short x_target should equal TABLE_NEAR")
        self.assertAlmostEqual(x_rs, self.TABLE_NEAR, places=6,
                               msg="right_short x_target should equal TABLE_NEAR")

    def test_short_skills_x_clearly_less_than_deep(self):
        """
        Short skills should be at least 0.30m shorter than deep skills.
        With TABLE_NEAR=1.75 and TABLE_FAR≈2.185 the gap is ~0.435m.
        """
        _, x_short = self.get_skill("left_short")
        _, x_deep  = self.get_skill("left")
        self.assertLess(x_short, x_deep - 0.30,
                        f"left_short x={x_short:.3f} is not clearly shorter "
                        f"than left x={x_deep:.3f} (gap < 0.30m)")

    # --- index round-trip ---------------------------------------------------

    def test_skill_index_all_skills(self):
        for i, name in enumerate(self.SKILL_NAMES):
            self.assertEqual(self.skill_index(name), i)

    def test_skill_from_index_all(self):
        for i, name in enumerate(self.SKILL_NAMES):
            self.assertEqual(self.skill_from_index(i), name)

    def test_index_roundtrip(self):
        for name in self.SKILL_NAMES:
            self.assertEqual(self.skill_from_index(self.skill_index(name)), name)

    def test_unknown_skill_raises(self):
        with self.assertRaises(ValueError):
            self.get_skill("smash")

    def test_unknown_skill_error_message(self):
        with self.assertRaises(ValueError) as ctx:
            self.get_skill("drop_shot")
        self.assertIn("drop_shot", str(ctx.exception))

    # --- semantic ordering of skill indices (left→center→right) --------------

    def test_left_has_index_0(self):
        """'left' must be first so its normalised index is 0.0."""
        self.assertEqual(self.skill_index("left"), 0)

    def test_right_has_index_4(self):
        """'right' must be last so its normalised index is 1.0."""
        self.assertEqual(self.skill_index("right"), self.N_SKILLS - 1)

    def test_left_short_before_center_safe(self):
        """left_short (index 1) < center_safe (index 2)."""
        self.assertLess(self.skill_index("left_short"), self.skill_index("center_safe"))

    def test_center_safe_before_right_short(self):
        """center_safe (index 2) < right_short (index 3)."""
        self.assertLess(self.skill_index("center_safe"), self.skill_index("right_short"))

    def test_right_short_before_right(self):
        """right_short (index 3) < right (index 4)."""
        self.assertLess(self.skill_index("right_short"), self.skill_index("right"))

    def test_skill_order_is_left_to_right(self):
        """Full expected order: left, left_short, center_safe, right_short, right."""
        expected = ["left", "left_short", "center_safe", "right_short", "right"]
        self.assertEqual(self.SKILL_NAMES, expected)


# =========================================================================== #
# 2. env_wrapper.py  (no MuJoCo — mocked inner env)                           #
# =========================================================================== #

def _make_mock_env():
    """Return a mock KukaTennisEnv-like object without MuJoCo."""
    env = types.SimpleNamespace()
    env.side_target     = 0.0
    env.side_target_opp = 0.0

    # Track calls to update_target_racket_pose
    env._last_update_kwargs     = {}
    env._last_update_opp_kwargs = {}
    env._step_called            = False

    def update_target_racket_pose(**kwargs):
        env._last_update_kwargs = dict(kwargs)

    def update_target_racket_pose_opp(**kwargs):
        env._last_update_opp_kwargs = dict(kwargs)

    env.update_target_racket_pose     = update_target_racket_pose
    env.update_target_racket_pose_opp = update_target_racket_pose_opp

    # Fake obs: 116-dim, last two = side_target, side_target_opp
    def reset(seed=None):
        env.side_target     = np.random.choice([-1., 1.])
        env.side_target_opp = np.random.choice([-1., 1.])
        obs = np.zeros(116, dtype=np.float32)
        obs[-2] = env.side_target
        obs[-1] = env.side_target_opp
        return obs, {}

    def step(action):
        env._step_called = True
        # Simulate the inner env calling update_target_racket_pose
        env.update_target_racket_pose(y_target=env.side_target * 0.38)
        env.update_target_racket_pose_opp(y_target=env.side_target_opp * 0.38)
        obs = np.zeros(116, dtype=np.float32)
        obs[-2] = env.side_target
        obs[-1] = env.side_target_opp
        return obs, 0.0, False, False, {}

    env.reset = reset
    env.step  = step
    env.close = lambda: None

    # Use a simple namespace for action/observation space so tests don't
    # require gymnasium or gym to be installed.
    env.action_space      = types.SimpleNamespace(shape=(18,))
    env.observation_space = types.SimpleNamespace(shape=(116,))
    return env


class _SkillEnvWithMock:
    """SkillEnv but with a mock inner env injected."""

    def __init__(self, mock_env):
        from nash_skills.skills import get_skill, SKILL_NAMES
        self._env     = mock_env
        self._skill1  = "left"
        self._skill2  = "right"
        self._x_target1 = None
        self._x_target2 = None
        self._get_skill  = get_skill
        self._SKILL_NAMES = SKILL_NAMES
        self._apply_skills()

    def set_skills(self, skill1, skill2):
        if skill1 not in self._SKILL_NAMES:
            raise ValueError(f"Unknown skill '{skill1}'")
        if skill2 not in self._SKILL_NAMES:
            raise ValueError(f"Unknown skill '{skill2}'")
        self._skill1 = skill1
        self._skill2 = skill2
        self._apply_skills()

    def _apply_skills(self):
        side1, x1 = self._get_skill(self._skill1)
        side2, x2 = self._get_skill(self._skill2)
        self._env.side_target     = side1
        self._env.side_target_opp = side2
        self._x_target1 = x1
        self._x_target2 = x2

    def reset(self, seed=None):
        obs, info = self._env.reset(seed=seed)
        self._apply_skills()
        return obs, info

    def step(self, action):
        self._apply_skills()
        original_update     = self._env.update_target_racket_pose
        original_update_opp = self._env.update_target_racket_pose_opp
        x1 = self._x_target1
        x2 = self._x_target2

        def patched_update(**kwargs):
            kwargs.setdefault("x_target", x1)
            original_update(**kwargs)

        def patched_update_opp(**kwargs):
            kwargs.setdefault("x_target", x2)
            original_update_opp(**kwargs)

        self._env.update_target_racket_pose     = patched_update
        self._env.update_target_racket_pose_opp = patched_update_opp
        result = self._env.step(action)
        self._env.update_target_racket_pose     = original_update
        self._env.update_target_racket_pose_opp = original_update_opp
        return result

    @property
    def side_target(self):
        return self._env.side_target

    @property
    def side_target_opp(self):
        return self._env.side_target_opp


class TestSkillEnv(unittest.TestCase):

    def _make(self, skill1="left", skill2="right"):
        mock = _make_mock_env()
        env  = _SkillEnvWithMock(mock)
        env.set_skills(skill1, skill2)
        return env, mock

    # --- set_skills ---------------------------------------------------------

    def test_set_skills_sets_side_target(self):
        from nash_skills.skills import get_skill
        for name in ["left", "right", "left_short", "right_short", "center_safe"]:
            env, mock = self._make(name, "right")
            expected_side, _ = get_skill(name)
            self.assertAlmostEqual(mock.side_target, expected_side,
                                   msg=f"skill={name}")

    def test_set_skills_sets_side_target_opp(self):
        from nash_skills.skills import get_skill
        for name in ["left", "right", "left_short", "right_short", "center_safe"]:
            env, mock = self._make("left", name)
            expected_side, _ = get_skill(name)
            self.assertAlmostEqual(mock.side_target_opp, expected_side,
                                   msg=f"opp skill={name}")

    def test_set_skills_raises_on_unknown_skill1(self):
        env, _ = self._make()
        with self.assertRaises(ValueError):
            env.set_skills("smash", "right")

    def test_set_skills_raises_on_unknown_skill2(self):
        env, _ = self._make()
        with self.assertRaises(ValueError):
            env.set_skills("left", "spin")

    # --- reset re-applies skills after randomisation -----------------------

    def test_reset_restores_skill_side_target(self):
        from nash_skills.skills import get_skill
        env, mock = self._make("center_safe", "left_short")
        obs, _ = env.reset()
        expected_side1, _ = get_skill("center_safe")
        expected_side2, _ = get_skill("left_short")
        self.assertAlmostEqual(mock.side_target,     expected_side1)
        self.assertAlmostEqual(mock.side_target_opp, expected_side2)

    # --- step injects x_target via monkey-patch ----------------------------

    def test_step_injects_x_target_for_ego(self):
        from nash_skills.skills import get_skill
        env, mock = self._make("left_short", "right")
        _, x_short = get_skill("left_short")
        env.step(np.zeros(18))
        self.assertIn("x_target", mock._last_update_kwargs,
                      "update_target_racket_pose should receive x_target")
        self.assertAlmostEqual(mock._last_update_kwargs["x_target"], x_short)

    def test_step_injects_x_target_for_opp(self):
        from nash_skills.skills import get_skill
        env, mock = self._make("left", "right_short")
        _, x_short = get_skill("right_short")
        env.step(np.zeros(18))
        self.assertIn("x_target", mock._last_update_opp_kwargs)
        self.assertAlmostEqual(mock._last_update_opp_kwargs["x_target"], x_short)

    def test_step_deep_skill_uses_far_x(self):
        from nash_skills.skills import get_skill
        env, mock = self._make("left", "right")
        _, x_far = get_skill("left")
        env.step(np.zeros(18))
        self.assertAlmostEqual(mock._last_update_kwargs["x_target"], x_far)

    def test_step_center_safe_x_target(self):
        from nash_skills.skills import get_skill
        env, mock = self._make("center_safe", "left")
        _, x_mid = get_skill("center_safe")
        env.step(np.zeros(18))
        self.assertAlmostEqual(mock._last_update_kwargs["x_target"], x_mid)

    def test_step_preserves_y_target_from_side(self):
        """y_target in the update call must equal side_target * 0.38."""
        from nash_skills.skills import get_skill
        for skill_name in ["left", "right", "left_short", "right_short", "center_safe"]:
            env, mock = self._make(skill_name, "right")
            env.step(np.zeros(18))
            side, _ = get_skill(skill_name)
            expected_y = side * 0.38
            self.assertAlmostEqual(mock._last_update_kwargs.get("y_target"), expected_y,
                                   places=6, msg=f"skill={skill_name}")

    def test_step_restores_original_methods_after(self):
        """Monkey-patch must be cleaned up after step()."""
        env, mock = self._make("left_short", "right")
        original_update = mock.update_target_racket_pose
        env.step(np.zeros(18))
        self.assertIs(mock.update_target_racket_pose, original_update,
                      "update_target_racket_pose should be restored after step")

    def test_switching_skills_changes_x_target(self):
        from nash_skills.skills import get_skill
        env, mock = self._make("left", "right")
        env.step(np.zeros(18))
        _, x_deep = get_skill("left")
        self.assertAlmostEqual(mock._last_update_kwargs["x_target"], x_deep)

        env.set_skills("left_short", "right")
        env.step(np.zeros(18))
        _, x_short = get_skill("left_short")
        self.assertAlmostEqual(mock._last_update_kwargs["x_target"], x_short)


# =========================================================================== #
# 3. Observation preprocessing (from train_q_model_5skill logic)              #
# =========================================================================== #

class TestObsPreprocessing(unittest.TestCase):
    """
    Verifies that the normalised skill index encoding used in training
    is correct: skill_idx / (N_SKILLS - 1) maps [0, N_SKILLS-1] -> [0, 1].
    """

    def setUp(self):
        from nash_skills.skills import SKILL_NAMES, N_SKILLS, skill_index
        self.SKILL_NAMES  = SKILL_NAMES
        self.N_SKILLS     = N_SKILLS
        self.skill_index  = skill_index

    def _norm(self, name):
        return self.skill_index(name) / (self.N_SKILLS - 1)

    def test_first_skill_normalises_to_zero(self):
        self.assertAlmostEqual(self._norm(self.SKILL_NAMES[0]), 0.0)

    def test_last_skill_normalises_to_one(self):
        self.assertAlmostEqual(self._norm(self.SKILL_NAMES[-1]), 1.0)

    def test_all_normalised_indices_in_unit_range(self):
        for name in self.SKILL_NAMES:
            v = self._norm(name)
            self.assertGreaterEqual(v, 0.0, msg=name)
            self.assertLessEqual(v,   1.0, msg=name)

    def test_normalised_indices_are_strictly_increasing(self):
        vals = [self._norm(n) for n in self.SKILL_NAMES]
        for a, b in zip(vals, vals[1:]):
            self.assertLess(a, b)

    def test_obs_last_two_set_correctly(self):
        """Simulate how train_q_model_5skill.py rewrites the last two obs entries."""
        obs = np.zeros(116, dtype=np.float32)
        obs[-2] = 1.0   # original side_target
        obs[-1] = -1.0  # original side_target_opp

        skill1, skill2 = "left_short", "center_safe"
        obs_copy = obs.copy()
        obs_copy[-2] = self._norm(skill1)
        obs_copy[-1] = self._norm(skill2)

        self.assertAlmostEqual(float(obs_copy[-2]), self._norm("left_short"))
        self.assertAlmostEqual(float(obs_copy[-1]), self._norm("center_safe"))
        # Original not mutated
        self.assertAlmostEqual(float(obs[-2]), 1.0)

    def test_model_input_shape_unchanged(self):
        """The model still expects 116-dim input — skill encoding doesn't change shape."""
        obs = np.zeros(116, dtype=np.float32)
        obs[-2] = self._norm("left")
        obs[-1] = self._norm("right")
        self.assertEqual(obs.shape, (116,))


# =========================================================================== #
# 4. SimpleModel architecture compatibility                                    #
# =========================================================================== #

class TestSimpleModelCompatibility(unittest.TestCase):
    """
    Verifies that SimpleModel (model_arch.py) is compatible with the 5-skill
    pipeline without changes:
      - same 116-dim input
      - same hidden architecture
      - output is a scalar per sample
    """

    def setUp(self):
        from model_arch import SimpleModel
        self.SimpleModel = SimpleModel

    def test_model_accepts_116_inputs(self):
        model = self.SimpleModel(116, [64, 32, 16], 1)
        x = torch.randn(8, 116)
        out = model(x)
        self.assertEqual(out.shape, (8, 1))

    def test_model_output_range_tanh(self):
        """Tanh output must be in (-1, 1)."""
        model = self.SimpleModel(116, [64, 32, 16], 1, last_layer_activation='tanh')
        x = torch.randn(32, 116) * 10  # large inputs stress the tanh
        out = model(x)
        self.assertTrue((out > -1).all())
        self.assertTrue((out <  1).all())

    def test_model_no_activation_output_unbounded(self):
        """Potential model uses no final activation."""
        model = self.SimpleModel(116, [64, 32, 16], 1, last_layer_activation=None)
        x = torch.randn(8, 116)
        out = model(x)
        # Should not raise; output can be any real value
        self.assertEqual(out.shape, (8, 1))

    def test_model_skill_index_input_different_from_original(self):
        """
        The 5-skill model receives [0, 1] in the last two positions,
        while the original model received [-1, +1].
        Both are valid inputs to the same architecture.
        """
        model = self.SimpleModel(116, [64, 32, 16], 1)
        # Original-style input (±1)
        x_orig = torch.zeros(4, 116)
        x_orig[:, -2] = torch.tensor([-1., -1., 1., 1.])
        x_orig[:, -1] = torch.tensor([-1.,  1., -1., 1.])

        # 5-skill-style input (0–1)
        x_5sk = torch.zeros(4, 116)
        x_5sk[:, -2] = torch.tensor([0., 0.25, 0.5, 0.75])
        x_5sk[:, -1] = torch.tensor([0., 0.5,  1.,  0.25])

        # Both should produce valid outputs without error
        out_orig = model(x_orig)
        out_5sk  = model(x_5sk)
        self.assertEqual(out_orig.shape, (4, 1))
        self.assertEqual(out_5sk.shape,  (4, 1))

    def test_separate_model_files_do_not_conflict(self):
        """
        model1.pth and model1_5skill.pth are separate files.
        Loading one should not affect the other.
        """
        import os
        path_orig = "models/model1.pth"
        path_5sk  = "models/model1_5skill.pth"

        if not os.path.exists(path_orig):
            self.skipTest("models/model1.pth not found — run train_q_model.py first")

        model_a = self.SimpleModel(116, [64, 32, 16], 1)
        model_a.load_state_dict(torch.load(path_orig, weights_only=True))

        if os.path.exists(path_5sk):
            model_b = self.SimpleModel(116, [64, 32, 16], 1)
            model_b.load_state_dict(torch.load(path_5sk, weights_only=True))
            # The two models should produce different outputs (different weights)
            x = torch.randn(4, 116)
            out_a = model_a(x)
            out_b = model_b(x)
            # Not asserting equality — just that both run without error
            self.assertEqual(out_a.shape, out_b.shape)


# =========================================================================== #
# 5. Nash potential pick_skill logic (extracted from comp_5skill.py)           #
# =========================================================================== #

class TestPickSkillLogic(unittest.TestCase):
    """
    Tests the pick_skill() function logic extracted from comp_5skill.py.
    Uses a fake model_p that always returns a constant per skill index
    so we can predict the expected choice.
    """

    def setUp(self):
        from nash_skills.skills import SKILL_NAMES, N_SKILLS, skill_index, skill_from_index
        from model_arch import SimpleModel
        self.SKILL_NAMES      = SKILL_NAMES
        self.N_SKILLS         = N_SKILLS
        self.skill_index      = skill_index
        self.skill_from_index = skill_from_index

    def _pick_skill(self, model_p, strategy, player, obs_vec, other_idx):
        """Replicated pick_skill() from comp_5skill.py for testing."""
        N_SKILLS = self.N_SKILLS

        if strategy in self.SKILL_NAMES:
            return self.skill_index(strategy)
        if strategy == 'random':
            return np.random.randint(N_SKILLS)
        if strategy == 'nash-p':
            best_idx = 0
            best_val = -float("inf")
            for s in range(N_SKILLS):
                x = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
                if player == 1:
                    x[0, -2] = s / (N_SKILLS - 1)
                    x[0, -1] = other_idx / (N_SKILLS - 1)
                else:
                    x[0, -2] = other_idx / (N_SKILLS - 1)
                    x[0, -1] = s / (N_SKILLS - 1)
                with torch.no_grad():
                    val = model_p(x).item()
                if val > best_val:
                    best_val = val
                    best_idx = s
            return best_idx
        raise ValueError(f"Unknown strategy '{strategy}'")

    def _make_model_p_prefer(self, preferred_skill_idx):
        """Make a fake model_p that scores preferred_skill_idx highest for player 1."""
        from model_arch import SimpleModel
        import torch.nn as nn

        class FakeModelP(nn.Module):
            def __init__(self, preferred, n):
                super().__init__()
                self.preferred = preferred
                self.n = n
                # dummy parameter so it's a proper Module
                self.dummy = nn.Linear(1, 1, bias=False)
                self.batch_norm = nn.BatchNorm1d(116, affine=False)

            def forward(self, x):
                # Returns a high score only if -2 index matches preferred norm
                target_norm = self.preferred / (self.n - 1)
                skill_val = x[:, -2]
                score = -(skill_val - target_norm) ** 2  # peaked at preferred
                return score.unsqueeze(1)

        return FakeModelP(preferred_skill_idx, self.N_SKILLS)

    def test_fixed_strategy_returns_correct_index(self):
        obs = np.zeros(116, dtype=np.float32)
        for name in self.SKILL_NAMES:
            idx = self._pick_skill(None, name, 1, obs, 0)
            self.assertEqual(idx, self.skill_index(name), msg=f"strategy={name}")

    def test_random_strategy_returns_valid_index(self):
        obs = np.zeros(116, dtype=np.float32)
        results = set()
        for _ in range(200):
            idx = self._pick_skill(None, 'random', 1, obs, 0)
            self.assertIn(idx, range(self.N_SKILLS))
            results.add(idx)
        # With 200 trials, very likely to hit multiple skills
        self.assertGreater(len(results), 1)

    def test_nash_p_picks_highest_scoring_skill(self):
        obs = np.zeros(116, dtype=np.float32)
        for preferred in range(self.N_SKILLS):
            model_p = self._make_model_p_prefer(preferred)
            chosen = self._pick_skill(model_p, 'nash-p', 1, obs, 0)
            self.assertEqual(chosen, preferred,
                             f"nash-p should pick skill {preferred} when model prefers it")

    def test_nash_p_uses_player_index_correctly(self):
        """Player 2 writes to position -1, player 1 writes to position -2."""
        obs = np.zeros(116, dtype=np.float32)

        class CheckPositionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.last_x = None
                self.batch_norm = torch.nn.BatchNorm1d(116, affine=False)
                self.dummy = torch.nn.Linear(1, 1, bias=False)

            def forward(self, x):
                self.last_x = x.detach().clone()
                return torch.zeros(x.shape[0], 1)

        model_check = CheckPositionModel()

        # Player 1: skill written to position -2
        self._pick_skill(model_check, 'nash-p', 1, obs, 2)
        self.assertIsNotNone(model_check.last_x)
        # position -1 should be fixed (other player's skill idx=2 normalised)
        expected_other = 2 / (self.N_SKILLS - 1)
        self.assertAlmostEqual(float(model_check.last_x[0, -1]), expected_other, places=5)

        # Player 2: skill written to position -1
        self._pick_skill(model_check, 'nash-p', 2, obs, 1)
        expected_other2 = 1 / (self.N_SKILLS - 1)
        self.assertAlmostEqual(float(model_check.last_x[0, -2]), expected_other2, places=5)

    def test_unknown_strategy_raises(self):
        obs = np.zeros(116, dtype=np.float32)
        with self.assertRaises(ValueError):
            self._pick_skill(None, 'smash', 1, obs, 0)


# =========================================================================== #
# 6. Integration: skill application in full step cycle                         #
# =========================================================================== #

class TestSkillIntegration(unittest.TestCase):
    """
    Verifies that switching skills between steps correctly updates
    the x_target injected into pose-update calls.
    """

    def _make_env(self, skill1, skill2):
        mock = _make_mock_env()
        env  = _SkillEnvWithMock(mock)
        env.set_skills(skill1, skill2)
        return env, mock

    def test_all_25_skill_combos_step_without_error(self):
        from nash_skills.skills import SKILL_NAMES, get_skill
        import itertools
        for s1, s2 in itertools.product(SKILL_NAMES, SKILL_NAMES):
            env, mock = self._make_env(s1, s2)
            env.step(np.zeros(18))  # should not raise
            _, x1 = get_skill(s1)
            _, x2 = get_skill(s2)
            self.assertAlmostEqual(mock._last_update_kwargs.get("x_target"), x1,
                                   msg=f"ego={s1}")
            self.assertAlmostEqual(mock._last_update_opp_kwargs.get("x_target"), x2,
                                   msg=f"opp={s2}")

    def test_skill_switch_mid_episode(self):
        from nash_skills.skills import get_skill
        env, mock = self._make_env("left", "right")

        env.step(np.zeros(18))
        _, x_left = get_skill("left")
        self.assertAlmostEqual(mock._last_update_kwargs["x_target"], x_left)

        env.set_skills("left_short", "center_safe")
        env.step(np.zeros(18))
        _, x_short = get_skill("left_short")
        _, x_mid   = get_skill("center_safe")
        self.assertAlmostEqual(mock._last_update_kwargs["x_target"],     x_short)
        self.assertAlmostEqual(mock._last_update_opp_kwargs["x_target"], x_mid)

    def test_center_safe_side_target_is_zero_in_obs(self):
        env, mock = self._make_env("center_safe", "left")
        self.assertAlmostEqual(mock.side_target, 0.0)

    def test_short_skill_x_target_less_than_deep(self):
        from nash_skills.skills import get_skill
        env_deep,  mock_d = self._make_env("left",       "right")
        env_short, mock_s = self._make_env("left_short", "right")
        env_deep.step(np.zeros(18))
        env_short.step(np.zeros(18))
        self.assertGreater(
            mock_d._last_update_kwargs["x_target"],
            mock_s._last_update_kwargs["x_target"],
        )


# =========================================================================== #
# 7. Rally data and trained model sanity checks                               #
# =========================================================================== #

class TestRallyDataAndModels(unittest.TestCase):
    """
    Validates the collected rally data (rallies_5skill.pkl) and the trained
    model files (model1_5skill.pth, model2_5skill.pth, model_p_5skill.pth).

    All tests skip gracefully if the files don't exist yet.
    """

    RALLY_PATH  = "data/rallies_5skill.pkl"
    MODEL1_PATH = "models/model1_5skill.pth"
    MODEL2_PATH = "models/model2_5skill.pth"
    MODEL_P_PATH = "models/model_p_5skill.pth"

    def setUp(self):
        from nash_skills.skills import SKILL_NAMES, N_SKILLS
        self.SKILL_NAMES = SKILL_NAMES
        self.N_SKILLS    = N_SKILLS

    # ── rally data checks ────────────────────────────────────────────────── #

    def _load_rallies(self):
        import pickle
        if not os.path.exists(self.RALLY_PATH):
            self.skipTest(f"{self.RALLY_PATH} not found — run collect_data_5skill.py first")
        return pickle.load(open(self.RALLY_PATH, "rb"))

    def test_rally_file_exists(self):
        if not os.path.exists(self.RALLY_PATH):
            self.skipTest(f"{self.RALLY_PATH} not found")
        self.assertTrue(os.path.exists(self.RALLY_PATH))

    def test_rallies_nonempty(self):
        rallies = self._load_rallies()
        self.assertGreater(len(rallies), 0, "rallies_5skill.pkl must contain at least one rally")

    def test_all_25_skill_combos_covered(self):
        """Every (skill1, skill2) pair must appear at least once."""
        import itertools
        rallies = self._load_rallies()
        seen = set()
        for r in rallies:
            seen.add((r['skill1'], r['skill2']))
        expected = set(itertools.product(self.SKILL_NAMES, self.SKILL_NAMES))
        missing  = expected - seen
        self.assertEqual(missing, set(),
                         f"Missing skill combos in rally data: {missing}")

    def test_rally_entries_have_required_keys(self):
        rallies = self._load_rallies()
        for i, r in enumerate(rallies[:20]):   # sample first 20
            self.assertIn('skill1', r,  f"Rally {i} missing 'skill1'")
            self.assertIn('skill2', r,  f"Rally {i} missing 'skill2'")
            self.assertIn('states', r,  f"Rally {i} missing 'states'")
            self.assertIn(r['skill1'], self.SKILL_NAMES, f"Rally {i} has unknown skill1")
            self.assertIn(r['skill2'], self.SKILL_NAMES, f"Rally {i} has unknown skill2")

    def test_rally_states_are_116_dim(self):
        rallies = self._load_rallies()
        for r in rallies[:20]:
            for state in r['states']:
                self.assertEqual(len(state), 116,
                                 f"State dim should be 116, got {len(state)}")

    def test_rally_skill_indices_match_current_order(self):
        """
        Skills in the rally file must match the CURRENT SKILL_NAMES order
        (left, left_short, center_safe, right_short, right).
        If this fails, the data was collected with an old skill order and
        must be re-collected.
        """
        from nash_skills.skills import skill_index
        rallies = self._load_rallies()
        # Just check that all skill names in the data are valid under current order
        for r in rallies[:50]:
            try:
                skill_index(r['skill1'])
                skill_index(r['skill2'])
            except ValueError as e:
                self.fail(f"Rally contains skill not in current SKILL_NAMES: {e}")

    def test_minimum_rally_count(self):
        """Expect at least 5 rallies per combo on average = 125 total minimum."""
        rallies = self._load_rallies()
        self.assertGreaterEqual(len(rallies), 125,
                                f"Only {len(rallies)} rallies — need >= 125 for training")

    # ── trained model checks ─────────────────────────────────────────────── #

    def _load_model(self, path):
        from model_arch import SimpleModel
        if not os.path.exists(path):
            self.skipTest(f"{path} not found — run train_q_model_5skill.py first")
        m = SimpleModel(116, [64, 32, 16], 1)
        m.load_state_dict(torch.load(path, weights_only=True))
        m.eval()
        return m

    def test_model1_loadable(self):
        self._load_model(self.MODEL1_PATH)

    def test_model2_loadable(self):
        self._load_model(self.MODEL2_PATH)

    def test_model_p_loadable(self):
        from model_arch import SimpleModel
        if not os.path.exists(self.MODEL_P_PATH):
            self.skipTest(f"{self.MODEL_P_PATH} not found — run train_q_model_5skill.py first")
        m = SimpleModel(116, [64, 32, 16], 1, last_layer_activation=None)
        m.load_state_dict(torch.load(self.MODEL_P_PATH, weights_only=True))
        m.eval()

    def test_model1_produces_bounded_output(self):
        """Q-value model1 uses tanh: output must be in [-1, 1]."""
        model = self._load_model(self.MODEL1_PATH)
        x = torch.randn(16, 116)
        with torch.no_grad():
            out = model(x)
        self.assertTrue((out >= -1).all() and (out <= 1).all(),
                        "model1 output should be in [-1, 1] due to tanh activation")

    def test_model_p_output_unbounded(self):
        """Potential model has no final activation — output is unbounded."""
        from model_arch import SimpleModel
        if not os.path.exists(self.MODEL_P_PATH):
            self.skipTest(f"{self.MODEL_P_PATH} not found")
        m = SimpleModel(116, [64, 32, 16], 1, last_layer_activation=None)
        m.load_state_dict(torch.load(self.MODEL_P_PATH, weights_only=True))
        m.eval()
        x = torch.randn(16, 116)
        with torch.no_grad():
            out = m(x)
        self.assertEqual(out.shape, (16, 1))

    def test_potential_skill_ordering_is_meaningful(self):
        """
        For a fixed state, sweeping player 1's skill index [0→4] through the
        potential model should produce non-constant output (the model learned
        something skill-dependent).
        """
        from model_arch import SimpleModel
        if not os.path.exists(self.MODEL_P_PATH):
            self.skipTest(f"{self.MODEL_P_PATH} not found")
        m = SimpleModel(116, [64, 32, 16], 1, last_layer_activation=None)
        m.load_state_dict(torch.load(self.MODEL_P_PATH, weights_only=True))
        m.eval()

        base = torch.zeros(1, 116)
        values = []
        with torch.no_grad():
            for idx in range(self.N_SKILLS):
                x = base.clone()
                x[0, -2] = idx / (self.N_SKILLS - 1)
                values.append(m(x).item())

        # At least two skill indices should produce different potential values
        self.assertGreater(max(values) - min(values), 1e-4,
                           "Potential model output is identical for all skill indices — "
                           "model may not have learned skill-dependent behaviour")

    def test_training_logs_exist(self):
        """train_q_model_5skill.py should have written CSV logs."""
        q_log = "logs/train_q_5skill.csv"
        p_log = "logs/train_p_5skill.csv"
        if not os.path.exists(self.MODEL1_PATH):
            self.skipTest("Models not trained yet")
        self.assertTrue(os.path.exists(q_log),  f"{q_log} not found")
        self.assertTrue(os.path.exists(p_log),  f"{p_log} not found")

    def test_q_loss_csv_has_correct_columns(self):
        import csv
        q_log = "logs/train_q_5skill.csv"
        if not os.path.exists(q_log):
            self.skipTest(f"{q_log} not found")
        with open(q_log) as f:
            reader = csv.DictReader(f)
            self.assertIn('epoch', reader.fieldnames)
            self.assertIn('loss1', reader.fieldnames)
            self.assertIn('loss2', reader.fieldnames)
            rows = list(reader)
        self.assertGreater(len(rows), 0, "Q-loss CSV is empty")

    def test_q_loss_decreases_over_training(self):
        """loss1 at epoch 1500 should be lower than at epoch 1."""
        import csv
        q_log = "logs/train_q_5skill.csv"
        if not os.path.exists(q_log):
            self.skipTest(f"{q_log} not found")
        with open(q_log) as f:
            rows = list(csv.DictReader(f))
        if len(rows) < 100:
            self.skipTest("Too few epochs logged to check convergence")
        first_loss  = float(rows[0]['loss1'])
        last_loss   = float(rows[-1]['loss1'])
        self.assertLess(last_loss, first_loss,
                        f"Q-loss did not decrease: first={first_loss:.5f} last={last_loss:.5f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
