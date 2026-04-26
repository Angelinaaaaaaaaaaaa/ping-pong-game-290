"""
Regression tests for low win-rate bugs in the 2-skill evaluation pipeline.

Bugs diagnosed:
  1. _infer_winner uses ball POSITION (ball_x > TABLE_SHIFT=1.5) to decide the winner.
     The correct signal is ball VELOCITY: ball_vel_x > 0 → ball moving toward opp →
     opp missed → ego wins.  Using position gives wrong answers when the ball's final
     position is ambiguous (e.g., still near the net at the moment done fires).

  2. eval_matchup_2skill.py loads the OLD buggy model_p.pth (sparse labels, LR=0.1,
     {0,1} encoding) instead of the corrected model_p_v2.pth.  The old model outputs
     near-constant values so nash-p effectively picks randomly.

TDD note: tests written BEFORE the fix.  Run:
    venv/bin/python -m pytest tests/test_winrate_bugs.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import numpy as np


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _make_obs(ball_x: float, ball_vel_x: float) -> np.ndarray:
    """
    Build a minimal 116-dim obs array with ball position and velocity set.

    In KukaTennisEnv.step():
      obs[36:39] = ball position (x, y, z)
      obs[39:42] = ball velocity (vx, vy, vz)
    """
    obs = np.zeros(116, dtype=np.float32)
    obs[36] = ball_x
    obs[39] = ball_vel_x
    return obs


TABLE_SHIFT = 1.5   # net position — same constant as the env


# --------------------------------------------------------------------------- #
# 1. _infer_winner must use ball velocity, not position                        #
# --------------------------------------------------------------------------- #

class TestInferWinner2SkillVelocityBased(unittest.TestCase):
    """
    _infer_winner in eval_matchup_2skill.py must use ball_vel_x, not ball_x.

    The environment fires done when:
      ball_pos[0] < racket_ego[0] - 0.3   → ego missed → OPP wins
      ball_pos[0] > racket_opp[0] + 0.3   → opp missed → EGO wins

    At the moment done fires, the ball's position can be anywhere — it does NOT
    reliably satisfy ball_x > TABLE_SHIFT for an ego win.  The ball velocity
    direction is the correct signal:
      ball_vel_x > 0  → ball moving toward opp side → opp missed → EGO wins
      ball_vel_x < 0  → ball moving toward ego side → ego missed → OPP wins
    """

    def _get_infer_fn(self):
        try:
            from nash_skills.eval_matchup_2skill import _infer_winner
            return _infer_winner
        except (ImportError, AttributeError):
            self.skipTest("_infer_winner not available in eval_matchup_2skill")

    def test_ego_wins_when_ball_moving_toward_opp(self):
        """
        ball_vel_x > 0 means ball heading to opp side → opp missed → ego wins.
        Ball position can be anywhere (even < TABLE_SHIFT).
        """
        infer = self._get_infer_fn()
        # Ball still on ego's side (x < TABLE_SHIFT) but moving toward opp
        obs = _make_obs(ball_x=1.0, ball_vel_x=3.0)
        result = infer(obs, {})
        self.assertEqual(result, "ego",
            "ball_vel_x > 0 should yield 'ego' win regardless of ball_x")

    def test_opp_wins_when_ball_moving_toward_ego(self):
        """
        ball_vel_x < 0 means ball heading to ego side → ego missed → opp wins.
        """
        infer = self._get_infer_fn()
        obs = _make_obs(ball_x=2.0, ball_vel_x=-3.0)
        result = infer(obs, {})
        self.assertEqual(result, "opp",
            "ball_vel_x < 0 should yield 'opp' win")

    def test_ego_wins_ball_on_ego_side_but_vel_positive(self):
        """
        Critical case the old position heuristic got WRONG:
        Ball at x=0.5 (deep on ego's side, < TABLE_SHIFT) but ball_vel_x > 0.
        Old code: ball_x=0.5 < 1.5 → returns 'opp' (WRONG).
        New code: ball_vel_x=5.0 > 0 → returns 'ego' (CORRECT).
        """
        infer = self._get_infer_fn()
        obs = _make_obs(ball_x=0.5, ball_vel_x=5.0)
        result = infer(obs, {})
        self.assertEqual(result, "ego",
            "Position heuristic bug: ball at x=0.5 but vel > 0 should be ego win")

    def test_opp_wins_ball_on_opp_side_but_vel_negative(self):
        """
        Critical case: ball at x=2.5 (on opp's side, > TABLE_SHIFT) but vel < 0.
        Old code: ball_x=2.5 > 1.5 → returns 'ego' (WRONG).
        New code: ball_vel_x=-3.0 < 0 → returns 'opp' (CORRECT).
        """
        infer = self._get_infer_fn()
        obs = _make_obs(ball_x=2.5, ball_vel_x=-3.0)
        result = infer(obs, {})
        self.assertEqual(result, "opp",
            "Position heuristic bug: ball at x=2.5 but vel < 0 should be opp win")

    def test_zero_velocity_falls_back_gracefully(self):
        """
        When ball_vel_x == 0 (stationary), return some consistent value.
        We don't prescribe which, just that it doesn't crash.
        """
        infer = self._get_infer_fn()
        obs = _make_obs(ball_x=TABLE_SHIFT, ball_vel_x=0.0)
        result = infer(obs, {})
        self.assertIn(result, ("ego", "opp"),
            "zero velocity should return 'ego' or 'opp' without crashing")


class TestInferWinnerEvalMatchupVelocityBased(unittest.TestCase):
    """Same tests for _infer_winner in eval_matchup.py (5-skill evaluator)."""

    def _get_infer_fn(self):
        try:
            from nash_skills.eval_matchup import _infer_winner
            return _infer_winner
        except (ImportError, AttributeError):
            self.skipTest("_infer_winner not available in nash_skills.eval_matchup")

    def test_ego_wins_when_ball_moving_toward_opp(self):
        infer = self._get_infer_fn()
        obs = _make_obs(ball_x=1.0, ball_vel_x=3.0)
        result = infer(obs, {})
        self.assertEqual(result, "ego")

    def test_opp_wins_when_ball_moving_toward_ego(self):
        infer = self._get_infer_fn()
        obs = _make_obs(ball_x=2.0, ball_vel_x=-3.0)
        result = infer(obs, {})
        self.assertEqual(result, "opp")

    def test_ego_wins_ball_on_ego_side_but_vel_positive(self):
        infer = self._get_infer_fn()
        obs = _make_obs(ball_x=0.5, ball_vel_x=5.0)
        result = infer(obs, {})
        self.assertEqual(result, "ego")

    def test_opp_wins_ball_on_opp_side_but_vel_negative(self):
        infer = self._get_infer_fn()
        obs = _make_obs(ball_x=2.5, ball_vel_x=-3.0)
        result = infer(obs, {})
        self.assertEqual(result, "opp")


# --------------------------------------------------------------------------- #
# 2. eval_matchup_2skill must have --v2 support to load model_p_v2.pth         #
# --------------------------------------------------------------------------- #

class TestEvalMatchup2SkillHasV2Flag(unittest.TestCase):
    """
    eval_matchup_2skill.py must expose MODEL_P_V2_PATH, MODEL_P_76DIM_PATH,
    and a --model CLI flag to select which potential model to load.

    The preferred model is model_p_76dim.pth (76-dim encoded state with joint
    angles), selected via --model 76dim (the default).
    """

    def _read_source(self):
        path = os.path.join(os.path.dirname(__file__), "..", "nash_skills", "eval_matchup_2skill.py")
        if not os.path.exists(path):
            self.skipTest("eval_matchup_2skill.py not found")
        with open(path) as f:
            return f.read()

    def test_model_p_v2_path_defined(self):
        """eval_matchup_2skill.py must define MODEL_P_V2_PATH constant."""
        src = self._read_source()
        self.assertIn("MODEL_P_V2_PATH", src,
            "eval_matchup_2skill.py must define MODEL_P_V2_PATH for the 116-dim model")

    def test_model_p_76dim_path_defined(self):
        """eval_matchup_2skill.py must define MODEL_P_76DIM_PATH for the new 76-dim model."""
        src = self._read_source()
        self.assertIn("MODEL_P_76DIM_PATH", src,
            "eval_matchup_2skill.py must define MODEL_P_76DIM_PATH for the 76-dim model")

    def test_argparse_has_model_flag(self):
        """The CLI must have a --model flag to select the potential model variant."""
        src = self._read_source()
        self.assertIn("--model", src,
            "eval_matchup_2skill.py must have --model CLI flag to select model variant")


# --------------------------------------------------------------------------- #
# 3. make_picker_2skill encoding must match train_q_model_v2 conventions      #
# --------------------------------------------------------------------------- #

class TestMakePicker2SkillEncoding(unittest.TestCase):
    """
    make_picker_2skill with strategy='nash-p-2skill' must use {-1.0, +1.0}
    skill encoding when building the observation vector — matching what
    train_q_model_v2.py was trained on.

    Left  → obs[-2] or obs[-1] = -1.0
    Right → obs[-2] or obs[-1] = +1.0
    """

    def _get_picker_fn(self):
        try:
            from nash_skills.eval_matchup_2skill import make_picker_2skill
            return make_picker_2skill
        except (ImportError, AttributeError):
            self.skipTest("make_picker_2skill not available")

    def _make_dummy_model_p(self, prefer_val):
        """
        Dummy potential model that returns a fixed tensor matching the batch size.
        prefer_val is applied to the last row — simulating a model that prefers
        the last combo in the batch.
        """
        import torch
        import torch.nn as nn

        class _DummyModel(nn.Module):
            def forward(self, x):
                # Return shape (batch, 1) — last row gets prefer_val
                out = torch.zeros(x.shape[0], 1)
                out[-1] = prefer_val
                return out

        return _DummyModel()

    def test_left_encodes_as_minus_one(self):
        """
        When pick_nash considers skill index 0 (left), the encoded feature vector
        entry for ego skill must be -1.0, not 0.0.
        Verified by inspecting the tensor batch built inside pick_nash.
        Uses model_state_dim=116 to bypass encode_ego (raw obs passthrough).
        """
        import torch

        captured_batches = []

        class _CapturingModel:
            def __call__(self, x):
                captured_batches.append(x.clone())
                # Return shape (batch, 1), all zeros → argmax picks first row
                return torch.zeros(x.shape[0], 1)
            def eval(self): return self

        from nash_skills.eval_matchup_2skill import make_picker_2skill
        model_p = _CapturingModel()
        pick = make_picker_2skill("nash-p-2skill", model_p, model_state_dim=116)

        obs = np.zeros(116, dtype=np.float32)
        pick(player=1, obs_vec=obs, info={}, other_idx=1)

        self.assertTrue(len(captured_batches) > 0, "model_p was never called")
        batch = captured_batches[-1]  # shape (4, 116)

        # Find the row where ego skill is set to -1.0 (left)
        left_rows = (batch[:, -2] == -1.0).nonzero(as_tuple=False).flatten().tolist()
        self.assertTrue(len(left_rows) > 0,
            f"No row in batch has obs[-2]=-1.0 for left ego. "
            f"Actual obs[-2] values: {batch[:, -2].tolist()}")

    def test_right_encodes_as_plus_one(self):
        """When pick_nash considers skill index 1 (right), obs[-2] must be +1.0."""
        import torch

        captured_batches = []

        class _CapturingModel:
            def __call__(self, x):
                captured_batches.append(x.clone())
                return torch.zeros(x.shape[0], 1)
            def eval(self): return self

        from nash_skills.eval_matchup_2skill import make_picker_2skill
        model_p = _CapturingModel()
        pick = make_picker_2skill("nash-p-2skill", model_p, model_state_dim=116)

        obs = np.zeros(116, dtype=np.float32)
        pick(player=1, obs_vec=obs, info={}, other_idx=0)

        self.assertTrue(len(captured_batches) > 0)
        batch = captured_batches[-1]

        right_rows = (batch[:, -2] == 1.0).nonzero(as_tuple=False).flatten().tolist()
        self.assertTrue(len(right_rows) > 0,
            f"No row in batch has obs[-2]=+1.0 for right ego. "
            f"Actual obs[-2] values: {batch[:, -2].tolist()}")


# --------------------------------------------------------------------------- #
# 4. 2-skill v2 output paths must not collide with 5-skill v2 paths           #
# --------------------------------------------------------------------------- #

class TestTwoSkillV2PathCollision(unittest.TestCase):
    """
    Both train_q_model_v2.py (2-skill, 76-dim encoded) and
    nash_skills/v2/train_models.py (5-skill, 76-dim) write different models.
    They must use distinct filenames to avoid clobbering each other.

    Fix: 2-skill trainer uses _76dim suffix; 5-skill trainer keeps _v2 suffix.
    eval_matchup_2skill.py uses MODEL_P_76DIM_PATH for the preferred model.
    """

    def _read_2skill_trainer(self):
        path = os.path.join(os.path.dirname(__file__), "..", "train_q_model_v2.py")
        if not os.path.exists(path):
            self.skipTest("train_q_model_v2.py not found")
        with open(path) as f:
            return f.read()

    def _read_2skill_evaluator(self):
        path = os.path.join(os.path.dirname(__file__), "..", "nash_skills", "eval_matchup_2skill.py")
        if not os.path.exists(path):
            self.skipTest("eval_matchup_2skill.py not found")
        with open(path) as f:
            return f.read()

    def test_train_saves_model1_76dim(self):
        """train_q_model_v2.py must save to model1_76dim.pth."""
        src = self._read_2skill_trainer()
        self.assertIn("model1_76dim.pth", src,
            "train_q_model_v2.py must write model1_76dim.pth (76-dim encoded state)")

    def test_train_saves_model2_76dim(self):
        """train_q_model_v2.py must save to model2_76dim.pth."""
        src = self._read_2skill_trainer()
        self.assertIn("model2_76dim.pth", src,
            "train_q_model_v2.py must write model2_76dim.pth")

    def test_train_saves_model_p_76dim(self):
        """train_q_model_v2.py must save to model_p_76dim.pth."""
        src = self._read_2skill_trainer()
        self.assertIn("model_p_76dim.pth", src,
            "train_q_model_v2.py must write model_p_76dim.pth")

    def test_train_does_not_write_to_shared_v2_path(self):
        """train_q_model_v2.py must NOT torch.save to model_p_v2.pth (5-skill path)."""
        import re
        src = self._read_2skill_trainer()
        saves = re.findall(r'torch\.save\([^,]+,\s*["\']([^"\']+)["\']', src)
        for p in saves:
            self.assertFalse(
                p.endswith("model_p_v2.pth") or
                p.endswith("model1_v2.pth") or
                p.endswith("model2_v2.pth"),
                f"train_q_model_v2.py saves to shared 5-skill path '{p}'. "
                f"Use _76dim suffix instead."
            )

    def test_eval_defines_model_p_76dim_path(self):
        """eval_matchup_2skill.py must define MODEL_P_76DIM_PATH."""
        src = self._read_2skill_evaluator()
        self.assertIn("MODEL_P_76DIM_PATH", src,
            "eval_matchup_2skill.py must define MODEL_P_76DIM_PATH for the preferred 76-dim model")


# --------------------------------------------------------------------------- #
# 5. collect_data_v2.py must store 76-dim encoded state in 'states'           #
# --------------------------------------------------------------------------- #

class TestCollectDataV2StatesDimension(unittest.TestCase):
    """
    collect_data_v2.py stores crossing states in rally['states'].

    NEW design (mentor guidance): 'states' stores a 76-dim encoded state via
    encode_ego(obs, info), which includes gantry position, full joint angles,
    ball pos/vel, and skill indicators.  This richer representation replaces
    the old 116-dim raw obs (which was redundant and lacked proper encoding).

    'raw_obs' still stores obs.copy() for detect_winner (reads obs[39]).

    These tests inspect collect_data_v2.py source to verify the new design statically.
    """

    def _read_source(self):
        path = os.path.join(os.path.dirname(__file__), "..", "collect_data_v2.py")
        if not os.path.exists(path):
            self.skipTest("collect_data_v2.py not found")
        with open(path) as f:
            return f.read()

    def test_states_appends_encode_ego(self):
        """
        curr_states.append(...) must use encode_ego(obs, info) for the 76-dim
        encoded state that includes joint angles, gantry position, and ball state.
        """
        src = self._read_source()
        self.assertIn(
            "curr_states.append(encode_ego(obs, info))",
            src,
            "collect_data_v2.py must store encode_ego(obs, info) in 'states' "
            "for the 76-dim encoded state (joint angles + ball + skill)."
        )

    def test_states_does_not_append_raw_obs(self):
        """
        'states' must NOT store raw 116-dim obs — only encoded 76-dim.
        Raw obs is kept in 'raw_obs' for detect_winner.
        """
        import re
        src = self._read_source()
        # Find the crossing block and verify curr_states does not get raw obs
        crossing_block = re.search(
            r'# Record.+?(?=# Episode ended)',
            src, re.DOTALL
        )
        if crossing_block is None:
            self.skipTest("Could not extract crossing block from source")
        block_text = crossing_block.group(0)
        # curr_raw.append(obs.copy()) is fine; curr_states.append(obs.copy()) is not
        self.assertNotIn(
            "curr_states.append(obs.copy())",
            block_text,
            "curr_states must NOT append raw obs.copy() — use encode_ego(obs, info) instead"
        )

    def test_raw_obs_still_stored_for_winner_detection(self):
        """
        curr_raw.append(obs.copy()) must still exist — detect_winner uses raw obs
        to read ball_vel_x at obs[39].
        """
        src = self._read_source()
        self.assertIn(
            "curr_raw.append(obs.copy())",
            src,
            "curr_raw must still store raw obs for detect_winner (reads obs[39])"
        )

    def test_encode_ego_imported(self):
        """
        encode_ego must be imported from nash_skills.v2.state_encoder.
        """
        src = self._read_source()
        self.assertIn(
            "from nash_skills.v2.state_encoder import encode_ego",
            src,
            "collect_data_v2.py must import encode_ego from nash_skills.v2.state_encoder"
        )


# --------------------------------------------------------------------------- #
# 6. collect_data_v2.py must only count done episodes toward target_rallies   #
# --------------------------------------------------------------------------- #

class TestCollectDataV2DoneOnlyCount(unittest.TestCase):
    """
    collect_data_v2.py was counting EVERY episode (done AND truncated) toward
    target_rallies.  This meant 80%+ of collected rallies were truncated
    (winner=0), giving all-zero discounted returns → ~97% zero labels →
    Q-models learned near-zero → potential targets were zero → potential model
    got stuck at epoch 1 with constant loss and never improved.

    FIX: only increment `completed` when done=True.  Truncated episodes are
    discarded.  This ensures every stored rally has a real winner (1 or 2) and
    a nonzero discounted return at every crossing.

    These tests inspect collect_data_v2.py source statically.
    """

    def _read_source(self):
        path = os.path.join(os.path.dirname(__file__), "..", "collect_data_v2.py")
        if not os.path.exists(path):
            self.skipTest("collect_data_v2.py not found")
        with open(path) as f:
            return f.read()

    def test_completed_incremented_only_on_done(self):
        """
        `completed += 1` must appear inside a `if done` branch, not in a
        `if done or steps_in_ep >= MAX_STEPS_PER_EPISODE` branch.
        The old pattern counted truncated episodes, diluting the dataset with
        all-zero labels.
        """
        import re
        src = self._read_source()
        # The buggy pattern: `if done or steps_in_ep >= MAX_STEPS_PER_EPISODE:`
        # followed by `completed += 1`
        bad_pattern = r'if done or steps_in_ep[^:]+:'
        self.assertNotRegex(
            src, bad_pattern,
            "collect_data_v2.py must NOT count truncated episodes as completed. "
            "Use `if done:` to gate `completed += 1`."
        )

    def test_truncated_episodes_not_saved(self):
        """
        Truncated episodes (step-cap hit without done=True) must not be
        appended to all_rallies.  Only done=True episodes should be saved.
        """
        import re
        src = self._read_source()
        # Looking for the key pattern: episode append inside done-only block
        # The code should NOT save when `done` is False (step-cap truncation).
        # We verify that the save block is gated on `done`.
        # Simple heuristic: `all_rallies.append` must NOT appear outside an `if done` scope.
        # Check that there is no append inside a `steps_in_ep >= MAX_STEPS_PER_EPISODE` branch.
        truncated_branch = re.search(
            r'steps_in_ep\s*>=\s*MAX_STEPS_PER_EPISODE[^\n]*\n(.*?)(?=\n\s*(?:if|while|for|def|class|\Z))',
            src, re.DOTALL
        )
        if truncated_branch:
            branch_text = truncated_branch.group(1)
            self.assertNotIn(
                "all_rallies.append",
                branch_text,
                "all_rallies.append must NOT appear in the truncation branch. "
                "Only done=True episodes should be stored."
            )


# --------------------------------------------------------------------------- #
# 7. Trained model files must be 116-dim, not 76-dim                          #
# --------------------------------------------------------------------------- #

class TestTwoSkillV2ModelDimensions(unittest.TestCase):
    """
    After retraining with the 76-dim encoded state, model_p_76dim.pth /
    model1_76dim.pth / model2_76dim.pth must have 76-dim input weights —
    matching the 76-dim state that collect_data_v2.py (new design) stores.

    These tests check the actual .pth files on disk.
    """

    MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

    def _load_sd(self, filename):
        import torch
        path = os.path.join(self.MODELS_DIR, filename)
        if not os.path.exists(path):
            self.skipTest(f"{filename} not found — run train_q_model_v2.py first")
        return torch.load(path, weights_only=True)

    def _check_input_dim(self, sd, expected_dim: int, model_name: str):
        """Assert that fc.0.weight has shape (hidden, expected_dim)."""
        self.assertIn("fc.0.weight", sd,
            f"{model_name}: fc.0.weight not found in state_dict")
        shape = tuple(sd["fc.0.weight"].shape)
        self.assertEqual(shape[1], expected_dim,
            f"{model_name}: fc.0.weight has input dim {shape[1]}, expected {expected_dim}. "
            f"Re-run train_q_model_v2.py after re-collecting with collect_data_v2.py.")

    def _check_bn_dim(self, sd, expected_dim: int, model_name: str):
        """Assert that batch_norm.running_mean has shape (expected_dim,)."""
        self.assertIn("batch_norm.running_mean", sd,
            f"{model_name}: batch_norm.running_mean not found in state_dict")
        shape = tuple(sd["batch_norm.running_mean"].shape)
        self.assertEqual(shape[0], expected_dim,
            f"{model_name}: batch_norm.running_mean has dim {shape[0]}, expected {expected_dim}. "
            f"Re-run train_q_model_v2.py after re-collecting with collect_data_v2.py.")

    def test_model_p_76dim_is_76_dim(self):
        """model_p_76dim.pth must have 76-dim input (fc.0.weight[:, 76])."""
        sd = self._load_sd("model_p_76dim.pth")
        self._check_input_dim(sd, 76, "model_p_76dim.pth")

    def test_model_p_76dim_bn_is_76_dim(self):
        """model_p_76dim.pth batch_norm must be 76-dim."""
        sd = self._load_sd("model_p_76dim.pth")
        self._check_bn_dim(sd, 76, "model_p_76dim.pth")

    def test_model1_76dim_is_76_dim(self):
        """model1_76dim.pth must have 76-dim input."""
        sd = self._load_sd("model1_76dim.pth")
        self._check_input_dim(sd, 76, "model1_76dim.pth")

    def test_model2_76dim_is_76_dim(self):
        """model2_76dim.pth must have 76-dim input."""
        sd = self._load_sd("model2_76dim.pth")
        self._check_input_dim(sd, 76, "model2_76dim.pth")


# --------------------------------------------------------------------------- #
# 8. run_matchup_2skill must count only DONE episodes toward n_episodes       #
# --------------------------------------------------------------------------- #

class TestEvalMatchup2SkillDoneOnlyCount(unittest.TestCase):
    """
    run_matchup_2skill used to count EVERY episode (done AND truncated) toward
    n_episodes.  When a fixed-skill opponent (e.g. always-left) causes the PPO
    to return stably, 93% of episodes hit the step cap without a ball exit.
    Result: 60 requested episodes → only 4 done → win_rate = 0/60 = 0%
    even though the true win rate should be ~52%.

    FIX: count only done=True episodes toward n_episodes.  Truncated episodes
    are discarded (reset, no win/loss recorded, counter NOT incremented).
    This guarantees exactly n_episodes statistically meaningful done episodes.

    Tests check the source of eval_matchup_2skill.py statically.
    """

    def _read_source(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "nash_skills", "eval_matchup_2skill.py"
        )
        if not os.path.exists(path):
            self.skipTest("eval_matchup_2skill.py not found")
        with open(path) as f:
            return f.read()

    def test_completed_episodes_not_incremented_on_truncation(self):
        """
        The step-cap truncation branch must NOT increment completed_episodes.
        In the old code: truncated episodes incremented completed_episodes,
        so 56 truncated episodes out of 60 left only 4 real outcomes.
        """
        import re
        src = self._read_source()
        # Locate the truncation block: the `elif steps_in_episode >= max_steps_per_episode:` branch
        # In old code this block contained `completed_episodes += 1`
        # In fixed code it must NOT.
        trunc_block = re.search(
            r'(?:Per-episode step cap|steps_in_episode\s*>=\s*max_steps_per_episode)[^\n]*\n'
            r'((?:[ \t]+[^\n]+\n)*)',
            src,
        )
        self.assertIsNotNone(
            trunc_block,
            "Could not find the truncation block in eval_matchup_2skill.py"
        )
        block_text = trunc_block.group(1)
        self.assertNotIn(
            "completed_episodes += 1",
            block_text,
            "Truncated episodes must NOT increment completed_episodes. "
            "Only done=True episodes should count toward n_episodes. "
            "The old code made 56/60 episodes truncated → 0/4 win rate "
            "even though the true rate was ~52%."
        )

    def test_done_episodes_increment_completed_episodes(self):
        """
        The done=True branch in the evaluation loop must increment
        completed_episodes so the loop terminates after exactly n_episodes
        meaningful (won/lost) episodes.
        """
        import re
        src = self._read_source()
        # The evaluation loop done block contains _infer_winner AND completed_episodes += 1
        # Use a multiline pattern that captures the block with _infer_winner
        done_block = re.search(
            r'if done:\s*\n(.*?_infer_winner.*?completed_episodes \+= 1)',
            src, re.DOTALL
        )
        self.assertIsNotNone(
            done_block,
            "The done=True evaluation block must contain both "
            "_infer_winner() and completed_episodes += 1. "
            "Only done episodes should count toward n_episodes."
        )

    def test_truncated_episodes_still_tracked(self):
        """
        Truncated episodes must still increment truncated_episodes so the
        result dataclass can report how many step-cap episodes occurred.
        """
        import re
        src = self._read_source()
        trunc_block = re.search(
            r'(?:Per-episode step cap|steps_in_episode\s*>=\s*max_steps_per_episode)[^\n]*\n'
            r'((?:[ \t]+[^\n]+\n)*)',
            src,
        )
        if trunc_block:
            block_text = trunc_block.group(1)
            self.assertIn(
                "truncated_episodes += 1",
                block_text,
                "Truncated episodes must still increment truncated_episodes counter"
            )

    def test_result_episodes_equals_done_count(self):
        """
        MatchupResult.episodes must equal the number of done (non-truncated)
        episodes.  win_rate = ego_wins / episodes, so episodes must not
        include truncated ones or win_rate is diluted to near zero.
        """
        src = self._read_source()
        # The return statement must use completed_episodes for episodes field.
        # After the fix, completed_episodes only counts done episodes.
        # Verify that MatchupResult is constructed with episodes=completed_episodes.
        self.assertIn(
            "episodes=completed_episodes",
            src,
            "MatchupResult must be created with episodes=completed_episodes, "
            "and completed_episodes must only count done (not truncated) episodes"
        )


# --------------------------------------------------------------------------- #
# 9. Potential model must get nonzero gradients at epoch 1                    #
# --------------------------------------------------------------------------- #

class TestPotentialModelGradients(unittest.TestCase):
    """
    BUG: train_q_model_v2.py trains the potential model (model_p) in the
    default train() mode.  BatchNorm1d in train mode normalizes EACH batch
    independently — subtracting the batch's own mean and dividing by its own
    std.  X11 and X01 only differ in obs[-2:] (skill encoding ±1), but within
    each batch that column is CONSTANT, so BN maps it to 0 in both batches.
    Result: phi(X11) == phi(X01) exactly → phi_diff = 0 → loss ≈ target² →
    gradient = 0 at every epoch → potential weights never update.

    FIX: call model_p.eval() before the potential training loop so that BN
    uses its frozen running_mean/running_var (inherited from model1) instead
    of per-batch stats.  In eval mode both X11 and X01 pass through the same
    BN transformation, preserving the ±1 skill difference.

    Tests check:
    1. The training source code calls model_p.eval() before the potential loop.
    2. A freshly initialised phi in eval mode produces nonzero gradients on
       the first backward pass (verifying the fix works).
    3. A phi in train mode produces zero gradients (confirming the bug).
    """

    def _read_trainer(self):
        path = os.path.join(os.path.dirname(__file__), "..", "train_q_model_v2.py")
        if not os.path.exists(path):
            self.skipTest("train_q_model_v2.py not found")
        with open(path) as f:
            return f.read()

    def test_trainer_calls_model_p_eval_before_potential_loop(self):
        """
        train_q_model_v2.py must call model_p.eval() after inheriting BN stats
        and BEFORE the potential training loop starts.
        Without this call BatchNorm runs in train mode and zeros out skill diffs.
        """
        import re
        src = self._read_trainer()

        # Find the block between BN stat copy and the potential training loop
        # We expect: model_p.batch_norm.momentum = 0.0  ... model_p.eval()
        # both appearing before "for epoch in range(n_epochs):" in the potential section
        potential_section = re.search(
            r'model_p\.batch_norm\.momentum\s*=\s*0\.0(.+?)for epoch in range\(n_epochs\)',
            src, re.DOTALL
        )
        self.assertIsNotNone(
            potential_section,
            "Could not locate the potential setup block in train_q_model_v2.py"
        )
        setup_block = potential_section.group(1)
        self.assertIn(
            "model_p.eval()",
            setup_block,
            "train_q_model_v2.py must call model_p.eval() after setting BN "
            "momentum=0.0 and BEFORE the potential training loop. "
            "Without this, BatchNorm runs in train mode and zeros out all "
            "skill-encoding differences, making gradients = 0."
        )

    def test_phi_in_eval_mode_has_nonzero_gradients(self):
        """
        A potential model in eval() mode (frozen BN stats) must have nonzero
        parameter gradients after one backward pass on the phi_diff loss.
        This is the corrected training mode. Uses 76-dim (new encoded state).
        """
        import torch
        import torch.nn as nn
        from model_arch import SimpleModel

        torch.manual_seed(0)
        mp = SimpleModel(76, [64, 32, 16], 1, last_layer_activation=None)
        mp.eval()  # frozen BN — the fix

        X = torch.zeros(50, 76)
        X11 = X.clone(); X11[:, -2] = 1.0;  X11[:, -1] = 1.0
        X01 = X.clone(); X01[:, -2] = -1.0; X01[:, -1] = 1.0

        target = torch.tensor(0.01)
        criterion = nn.MSELoss()

        diff = mp(X11).mean() - mp(X01).mean()
        loss = criterion(diff, target)
        loss.backward()

        grad_norms = [
            p.grad.norm().item()
            for p in mp.parameters()
            if p.grad is not None
        ]
        self.assertTrue(
            len(grad_norms) > 0,
            "No parameters received gradients"
        )
        total_grad_norm = sum(grad_norms)
        self.assertGreater(
            total_grad_norm, 0.0,
            f"All gradients are zero in eval mode. "
            f"Got grad norms: {grad_norms}. "
            f"phi_diff = {diff.item():.8f}. "
            "Eval mode should preserve skill encoding through BN."
        )

    def test_phi_in_train_mode_has_zero_gradients(self):
        """
        Confirm the bug: a potential model in train() mode produces zero
        gradients because per-batch BN collapses the skill encoding.
        This test documents the broken behavior and will PASS (verifying
        our understanding of the bug). Uses 76-dim (new encoded state).
        """
        import torch
        import torch.nn as nn
        from model_arch import SimpleModel

        torch.manual_seed(0)
        mp = SimpleModel(76, [64, 32, 16], 1, last_layer_activation=None)
        mp.train()  # BUG: per-batch BN zeros out skill diff

        X = torch.zeros(50, 76)
        X11 = X.clone(); X11[:, -2] = 1.0;  X11[:, -1] = 1.0
        X01 = X.clone(); X01[:, -2] = -1.0; X01[:, -1] = 1.0

        target = torch.tensor(0.01)
        criterion = nn.MSELoss()

        diff = mp(X11).mean() - mp(X01).mean()
        loss = criterion(diff, target)
        loss.backward()

        grad_norms = [
            p.grad.norm().item()
            for p in mp.parameters()
            if p.grad is not None
        ]
        total_grad_norm = sum(grad_norms)
        self.assertEqual(
            total_grad_norm, 0.0,
            f"Expected zero gradients in train mode (the bug), "
            f"but got total norm = {total_grad_norm:.6f}. "
            "If this fails, the train-mode bug no longer exists and "
            "test_phi_in_eval_mode_has_nonzero_gradients may need revisiting."
        )

    def test_potential_loss_improves_over_epochs_in_eval_mode(self):
        """
        With model_p in eval() mode, training for a few epochs on a simple
        synthetic dataset must reduce the potential loss (i.e., loss at epoch
        10 < loss at epoch 1).  Uses 76-dim (new encoded state).
        """
        import torch
        import torch.nn as nn
        from model_arch import SimpleModel

        torch.manual_seed(42)
        # Create a Q-model with known stats to inherit
        m1 = SimpleModel(76, [64, 32, 16], 1)
        # Run a few forward passes to populate BN running stats
        X_warm = torch.randn(200, 76)
        m1.eval()
        with torch.no_grad():
            _ = m1(X_warm)

        mp = SimpleModel(76, [64, 32, 16], 1, last_layer_activation=None)
        mp.batch_norm.running_mean = m1.batch_norm.running_mean.clone()
        mp.batch_norm.running_var  = m1.batch_norm.running_var.clone()
        mp.batch_norm.momentum     = 0.0
        mp.eval()  # THE FIX

        X = torch.randn(100, 76)
        X11 = X.clone(); X11[:, -2] = 1.0;  X11[:, -1] = 1.0
        X01 = X.clone(); X01[:, -2] = -1.0; X01[:, -1] = 1.0

        target = torch.tensor(0.05)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(mp.parameters(), lr=0.001)

        first_loss = None
        for epoch in range(10):
            optimizer.zero_grad()
            diff = mp(X11).mean() - mp(X01).mean()
            loss = criterion(diff, target)
            loss.backward()
            optimizer.step()
            if first_loss is None:
                first_loss = loss.item()
            last_loss = loss.item()

        self.assertLess(
            last_loss, first_loss,
            f"Loss did not improve: first={first_loss:.8f}, last={last_loss:.8f}. "
            "Potential model in eval mode should learn to match Q-model targets."
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
