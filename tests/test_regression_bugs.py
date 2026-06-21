"""
Regression tests for bugs fixed in the 2026-06 review.

Bugs covered:
  1. train_q_model_5skill.py: model_p.eval() not called before training loop
     → BatchNorm in train mode zeros skill-diff gradients → degenerate potential
  2. eval_matchup.py: _infer_winner body entirely commented out → returns None
     → every episode scored as opp win → 0% ego win rate
  3. comp.py: duplicate 'nash-ibr' branch for STRATEGY2 shadows real IBR loop
  4. comp.py: model1/model2/model_p not set to eval() after loading
  5. mujoco_env_comp.py: curr_target[3:7] used instead of curr_target_opp[3:7]
     for opponent quaternion computation at reset
  6. mujoco_env_comp.py: ball_pos = self.data.body('ball').xpos (live view)
     mutated in update_target_racket_pose_opp → corrupts MuJoCo state mid-step
  7. mujoco_env_comp.py: ep_no % 200 == -1 dead branch → randomization never runs

Run (targeted):
    venv/bin/python -m pytest tests/test_regression_bugs.py -v
"""

import sys
import os
import re
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _read(relpath: str) -> str:
    path = os.path.join(ROOT, relpath)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return f.read()


# =========================================================================== #
# 1. train_q_model_5skill.py — model_p.eval() must precede training loop      #
# =========================================================================== #

class TestModelPEvalIn5SkillTrainer(unittest.TestCase):
    """
    model_p uses BatchNorm.  In train() mode, BN normalises each batch using
    that batch's own mean/std.  The potential training batches Xij and Xi2j
    differ only in skill columns, which are constant within a batch → BN maps
    them to 0 → phi(Xij) == phi(Xi2j) → gradient = 0 → potential never learns.

    Fix: call model_p.eval() after setting BN momentum=0 so that BN uses
    frozen running stats (inherited from model1), preserving skill differences.
    """

    SRC_PATH = "nash_skills/train_q_model_5skill.py"

    def _src(self):
        s = _read(self.SRC_PATH)
        if s is None:
            self.skipTest(f"{self.SRC_PATH} not found")
        return s

    def test_model_p_eval_called_before_training_loop(self):
        """model_p.eval() must appear after momentum=0 and before the epoch loop."""
        src = self._src()
        m = re.search(
            r'model_p\.batch_norm\.momentum\s*=\s*0\.0(.+?)for .+ in range\(',
            src, re.DOTALL
        )
        self.assertIsNotNone(
            m,
            "Could not find 'model_p.batch_norm.momentum = 0.0 ... for ... in range(' "
            "pattern in train_q_model_5skill.py"
        )
        setup_block = m.group(1)
        self.assertIn(
            "model_p.eval()",
            setup_block,
            "train_q_model_5skill.py must call model_p.eval() after setting "
            "batch_norm.momentum=0.0 and BEFORE the potential training loop. "
            "Without it, BatchNorm runs in train mode and zeroes skill-diff gradients."
        )

    def test_model_p_not_in_train_mode_for_potential(self):
        """Source must NOT call model_p.train() anywhere after the BN setup block."""
        src = self._src()
        # Find everything after the BN setup
        idx_bn = src.find("model_p.batch_norm.momentum")
        self.assertGreater(idx_bn, 0, "Could not find model_p.batch_norm.momentum line")
        tail = src[idx_bn:]
        # model_p.train() after setup would revert eval() — that's the bug pattern
        self.assertNotIn(
            "model_p.train()",
            tail,
            "model_p.train() found after BN setup. This would revert the eval() fix."
        )

    def test_phi_eval_mode_nonzero_gradient(self):
        """
        Functional: a phi model in eval() mode must produce nonzero gradients
        when Xij and Xi2j differ only in skill columns.
        """
        import torch
        import torch.nn as nn
        try:
            from model_arch import SimpleModel
        except ImportError:
            self.skipTest("model_arch not importable")

        torch.manual_seed(0)
        mp = SimpleModel(76, [64, 32, 16], 1, last_layer_activation=None)
        mp.eval()

        X = torch.zeros(50, 76)
        Xij  = X.clone(); Xij[:, -2]  =  1.0; Xij[:, -1]  = 1.0
        Xi2j = X.clone(); Xi2j[:, -2] = -1.0; Xi2j[:, -1] = 1.0

        diff = mp(Xij).mean() - mp(Xi2j).mean()
        nn.MSELoss()(diff, torch.tensor(0.01)).backward()

        total_grad = sum(
            p.grad.norm().item()
            for p in mp.parameters() if p.grad is not None
        )
        self.assertGreater(
            total_grad, 0.0,
            "phi in eval() mode has zero gradients — BN still zeroing skill diff."
        )

    def test_phi_train_mode_zero_gradient(self):
        """
        Documents the bug: phi in train() mode produces zero gradient because
        per-batch BN collapses the constant skill column.  This test must PASS
        (proving the bug exists in train mode, so eval mode is the right fix).
        """
        import torch
        import torch.nn as nn
        try:
            from model_arch import SimpleModel
        except ImportError:
            self.skipTest("model_arch not importable")

        torch.manual_seed(0)
        mp = SimpleModel(76, [64, 32, 16], 1, last_layer_activation=None)
        mp.train()

        X = torch.zeros(50, 76)
        Xij  = X.clone(); Xij[:, -2]  =  1.0; Xij[:, -1]  = 1.0
        Xi2j = X.clone(); Xi2j[:, -2] = -1.0; Xi2j[:, -1] = 1.0

        diff = mp(Xij).mean() - mp(Xi2j).mean()
        nn.MSELoss()(diff, torch.tensor(0.01)).backward()

        total_grad = sum(
            p.grad.norm().item()
            for p in mp.parameters() if p.grad is not None
        )
        self.assertEqual(
            total_grad, 0.0,
            f"Expected zero gradients in train() mode (the bug) but got {total_grad:.6f}."
        )


# =========================================================================== #
# 2. eval_matchup.py — _infer_winner must return a non-None string             #
# =========================================================================== #

class TestInferWinner5SkillNotNone(unittest.TestCase):
    """
    _infer_winner in eval_matchup.py had its entire body commented out,
    causing it to return None implicitly.  The caller in run_matchup casts
    None → 0 → every episode is counted as neither ego nor opp win → 0% ego win.

    Fix: uncomment the return infer_terminal_winner(obs, info, ...) line.
    """

    def _get_fn(self):
        try:
            from nash_skills.eval_matchup import _infer_winner
            return _infer_winner
        except ImportError:
            self.skipTest("nash_skills.eval_matchup not importable")

    def _obs(self, vel_x: float):
        import numpy as np
        obs = np.zeros(76, dtype=np.float32)
        obs[39] = vel_x
        return obs

    def test_returns_string_not_none_positive_vel(self):
        fn = self._get_fn()
        import numpy as np
        obs = np.zeros(116, dtype=np.float32)
        obs[39] = 3.0
        result = fn(obs, {})
        self.assertIsNotNone(result, "_infer_winner returned None — body is commented out")
        self.assertIn(result, ("ego", "opp"), f"Expected 'ego' or 'opp', got {result!r}")

    def test_returns_string_not_none_negative_vel(self):
        fn = self._get_fn()
        import numpy as np
        obs = np.zeros(116, dtype=np.float32)
        obs[39] = -3.0
        result = fn(obs, {})
        self.assertIsNotNone(result, "_infer_winner returned None — body is commented out")
        self.assertIn(result, ("ego", "opp"), f"Expected 'ego' or 'opp', got {result!r}")

    def test_positive_vel_means_ego_wins(self):
        """ball_vel_x > 0 → ball heading toward opp → opp missed → ego wins."""
        fn = self._get_fn()
        import numpy as np
        obs = np.zeros(116, dtype=np.float32)
        obs[39] = 4.0
        self.assertEqual(fn(obs, {}), "ego")

    def test_negative_vel_means_opp_wins(self):
        """ball_vel_x < 0 → ball heading toward ego → ego missed → opp wins."""
        fn = self._get_fn()
        import numpy as np
        obs = np.zeros(116, dtype=np.float32)
        obs[39] = -4.0
        self.assertEqual(fn(obs, {}), "opp")


# =========================================================================== #
# 3. comp.py — duplicate nash-ibr branch removed                               #
# =========================================================================== #

class TestCompNashIbrNoDuplicate(unittest.TestCase):
    """
    comp.py had two consecutive `elif STRATEGY2 == 'nash-ibr':` blocks.
    Python enters the first and skips the second entirely, so the real IBR
    loop was dead code and only the naive sign-flip ran.

    Fix: keep only the real IBR loop.
    """

    SRC_PATH = "comp.py"

    def _src(self):
        s = _read(self.SRC_PATH)
        if s is None:
            self.skipTest("comp.py not found")
        return s

    def test_single_nash_ibr_strategy2_branch(self):
        """STRATEGY2 == 'nash-ibr' must appear exactly once in the elif chain."""
        src = self._src()
        count = src.count("STRATEGY2 == 'nash-ibr'")
        self.assertEqual(
            count, 1,
            f"Found {count} occurrences of \"STRATEGY2 == 'nash-ibr'\" in comp.py. "
            "Must be exactly 1 — the real IBR loop. "
            "The duplicate naive sign-flip branch must be removed."
        )

    def test_ibr_loop_present(self):
        """The real IBR loop (for i in range(4)) must still exist in comp.py."""
        src = self._src()
        self.assertIn(
            "for i in range(4)",
            src,
            "The IBR loop 'for i in range(4)' is missing from comp.py. "
            "Make sure the real IBR loop was kept, not the naive sign-flip."
        )

    def test_naive_sign_flip_removed(self):
        """
        The naive sign-flip `env.side_target_opp = -env.side_target_opp` must
        NOT appear directly as the STRATEGY2='nash-ibr' handler.  It was the
        body of the first duplicate branch that shadowed the real IBR.
        The real IBR sets side_target_opp via the IBR iteration, not a simple negation.
        """
        src = self._src()
        # Find the nash-ibr block and check its immediate body
        m = re.search(
            r"STRATEGY2 == 'nash-ibr':\s*\n(\s+env\.side_target_opp\s*=\s*-env\.side_target_opp)",
            src
        )
        self.assertIsNone(
            m,
            "The naive sign-flip 'env.side_target_opp = -env.side_target_opp' is still "
            "the direct body of the nash-ibr branch. This was the dead-code bug. "
            "The real IBR loop must be the handler instead."
        )


# =========================================================================== #
# 4. comp.py — all three models set to eval() after loading                   #
# =========================================================================== #

class TestCompModelsInEvalMode(unittest.TestCase):
    """
    comp.py loaded model1, model2, model_p but never called .eval() on them.
    BatchNorm in inference mode sees one row at a time (batch_size=1–4) and
    normalises within that tiny batch, erasing the ±1 skill encoding.
    Result: nash / nash-ibr / nash-p skill selection is noisy/wrong.
    """

    SRC_PATH = "comp.py"

    def _src(self):
        s = _read(self.SRC_PATH)
        if s is None:
            self.skipTest("comp.py not found")
        return s

    def test_model1_eval_called(self):
        src = self._src()
        self.assertIn(
            "model1.eval()",
            src,
            "comp.py must call model1.eval() after loading weights."
        )

    def test_model2_eval_called(self):
        src = self._src()
        self.assertIn(
            "model2.eval()",
            src,
            "comp.py must call model2.eval() after loading weights."
        )

    def test_model_p_eval_called(self):
        src = self._src()
        self.assertIn(
            "model_p.eval()",
            src,
            "comp.py must call model_p.eval() after loading weights."
        )

    def test_eval_calls_after_load_state_dict(self):
        """All .eval() calls must appear after the last load_state_dict call."""
        src = self._src()
        last_load = src.rfind("load_state_dict")
        self.assertGreater(last_load, 0, "load_state_dict not found in comp.py")
        tail = src[last_load:]
        for name in ("model1.eval()", "model2.eval()", "model_p.eval()"):
            self.assertIn(
                name, tail,
                f"{name} must appear after the last load_state_dict call in comp.py"
            )


# =========================================================================== #
# 5. mujoco_env_comp.py — opponent quaternion uses curr_target_opp             #
# =========================================================================== #

class TestMujocoEnvOppQuaternion(unittest.TestCase):
    """
    At reset, the opponent observation's orientation error was computed using
    self.curr_target[3:7] (player-1's target quaternion) instead of
    self.curr_target_opp[3:7] (the opponent's own target quaternion).
    This puts wrong orientation error into the opponent's first-frame obs.
    """

    SRC_PATH = "mujoco_env_comp.py"

    def _src(self):
        s = _read(self.SRC_PATH)
        if s is None:
            self.skipTest("mujoco_env_comp.py not found")
        return s

    def test_opp_quat_uses_curr_target_opp(self):
        """
        R.from_quat(self.curr_target_opp[3:7]) must appear in the opponent
        observation block (where diff_quat_opp is computed).
        """
        src = self._src()
        # Check the diff_quat_opp computation block
        m = re.search(
            r'diff_quat_opp\s*=\s*r_target\s*\*\s*r_current\.inv\(\)',
            src
        )
        self.assertIsNotNone(m, "diff_quat_opp block not found in mujoco_env_comp.py")

        # The r_target for the opp block must use curr_target_opp
        block_start = src.rfind("r_target = R.from_quat(", 0, m.start())
        self.assertGreater(block_start, 0, "r_target assignment before diff_quat_opp not found")
        r_target_line = src[block_start: block_start + 60]
        self.assertIn(
            "curr_target_opp",
            r_target_line,
            f"Opponent quaternion must use self.curr_target_opp[3:7], "
            f"not self.curr_target[3:7]. Found: {r_target_line.strip()!r}"
        )

    def test_opp_quat_does_not_use_ego_target(self):
        """
        Within the 300 characters after diff_pos_opp = ..., the r_target
        assignment must NOT reference self.curr_target[3:7] (the ego target).
        """
        src = self._src()
        m = re.search(r'diff_pos_opp\s*=', src)
        if m is None:
            self.skipTest("diff_pos_opp not found — block structure may have changed")
        # Only check the immediate block (next ~300 chars) — not the whole file
        window = src[m.start(): m.start() + 300]
        bad_m = re.search(
            r'r_target\s*=\s*R\.from_quat\(self\.curr_target\[',
            window
        )
        self.assertIsNone(
            bad_m,
            "r_target uses self.curr_target[3:7] (ego) instead of "
            "self.curr_target_opp[3:7] in the opponent obs block (~300 chars "
            "after diff_pos_opp). Should be self.curr_target_opp[3:7]."
        )


# =========================================================================== #
# 6. mujoco_env_comp.py — opponent ball_pos must be a copy, not a live view   #
# =========================================================================== #

class TestMujocoEnvBallPosCopy(unittest.TestCase):
    """
    update_target_racket_pose_opp did:
        ball_pos = self.data.body('ball').xpos   # live view into MuJoCo!
        ball_pos[1] = -ball_pos[1]               # mutates MuJoCo state
    This corrupts the ball's y-coordinate inside the physics engine until the
    next mj_step overwrites it.  Ball-contact and success checks in the same
    timestep see the wrong position.

    Fix: ball_pos = self.data.body('ball').xpos.copy()
    """

    SRC_PATH = "mujoco_env_comp.py"

    def _src(self):
        s = _read(self.SRC_PATH)
        if s is None:
            self.skipTest("mujoco_env_comp.py not found")
        return s

    def test_ball_pos_opp_uses_copy(self):
        """
        In update_target_racket_pose_opp, the ball position must be obtained
        via .xpos.copy() to avoid mutating the MuJoCo live array.
        """
        src = self._src()
        # Find the method definition
        method_m = re.search(r'def update_target_racket_pose_opp\b', src)
        self.assertIsNotNone(method_m, "update_target_racket_pose_opp not found")
        method_body = src[method_m.start(): method_m.start() + 600]

        self.assertIn(
            "xpos.copy()",
            method_body,
            "update_target_racket_pose_opp must use .xpos.copy() for ball_pos "
            "to avoid mutating the MuJoCo live view array."
        )

    def test_ball_pos_opp_not_bare_xpos(self):
        """
        The bare `.xpos` assignment (without .copy()) must NOT appear as a
        standalone assignment in update_target_racket_pose_opp.
        """
        src = self._src()
        method_m = re.search(r'def update_target_racket_pose_opp\b', src)
        if method_m is None:
            self.skipTest("update_target_racket_pose_opp not found")
        method_body = src[method_m.start(): method_m.start() + 600]

        # Pattern: ball_pos = self.data.body('ball').xpos  (not followed by .copy())
        bare_m = re.search(
            r"ball_pos\s*=\s*self\.data\.body\('ball'\)\.xpos(?!\.copy\(\))",
            method_body
        )
        self.assertIsNone(
            bare_m,
            "Found bare .xpos assignment in update_target_racket_pose_opp. "
            "Must be .xpos.copy() to avoid mutating the MuJoCo live view."
        )


# =========================================================================== #
# 7. mujoco_env_comp.py — dead randomization branch ep_no % 200 == 0          #
# =========================================================================== #

class TestMujocoEnvRandomizationBranch(unittest.TestCase):
    """
    The joint randomization block used `if self.ep_no % 200 == -1:`.
    Python's modulo always returns 0..199 for positive ep_no, so -1 is
    impossible — the branch was dead code and randomization never ran.

    Fix: change to `== 0` so it fires every 200th episode.
    """

    SRC_PATH = "mujoco_env_comp.py"

    def _src(self):
        s = _read(self.SRC_PATH)
        if s is None:
            self.skipTest("mujoco_env_comp.py not found")
        return s

    def test_dead_branch_removed(self):
        """ep_no % 200 == -1 must not appear anywhere in the source."""
        src = self._src()
        self.assertNotIn(
            "ep_no%200 == -1",
            src,
            "Dead branch 'ep_no%200 == -1' still present. "
            "Python modulo is always >= 0, so this never fires. Change to == 0."
        )
        self.assertNotIn(
            "ep_no % 200 == -1",
            src,
            "Dead branch 'ep_no % 200 == -1' still present. Change to == 0."
        )

    def test_randomization_fires_on_modulo_zero(self):
        """ep_no % 200 == 0 must be the active condition."""
        src = self._src()
        has_zero = (
            "ep_no%200 == 0" in src or
            "ep_no % 200 == 0" in src
        )
        self.assertTrue(
            has_zero,
            "ep_no % 200 == 0 condition not found. "
            "Randomization will never fire if the dead -1 branch is used."
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
