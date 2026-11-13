import csv
import json
from pathlib import Path
import tempfile
import types
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from nash_skills.skills import SKILL_NAMES
from v3.constants import DEFENSE_POSITIONS
from v3 import eval_q_policy as evaluator
from v3 import eval_policy_fixed_grid as batch
from v3.potential_policy import (
    load_potential_model, potential_matrix, potential_max_distributions, potential_max_q_regrets,
)
from v3.potential_max_diagnostics import (
    potential_max_regret_rows, regret_summary, validate_regret_threshold, write_regret_diagnostics,
)


PHI = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 20], [11, 12, 13]], dtype=float)
Q = np.array([[.1, .2, .3], [.4, .5, .6], [.7, .8, .9], [-.3, .1, .2], [.2, .3, .4]])


class TableModel(torch.nn.Module):
    def __init__(self, matrix):
        super().__init__()
        self.matrix = matrix
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().cpu().numpy().copy())
        return torch.as_tensor(self.matrix.reshape(-1, 1), dtype=x.dtype, device=x.device)


class CrossingEnv:
    """Two crossings give both physical players both roles, then terminate."""
    skill_profile = "aggressive"

    def __init__(self):
        self.commits = []

    def set_skills(self, first, second):
        self._skill1, self._skill2 = first, second

    def clear_defense_position(self):
        pass

    def reset(self, seed=None):
        self.index = 0
        obs = np.zeros(116)
        obs[36] = 1.4
        return obs, {}

    def step(self, action):
        self.index += 1
        obs = np.zeros(116)
        obs[36] = 1.6 if self.index == 1 else 1.4
        return obs, 0., self.index == 3, False, {"winner": "ego", "termination_reason": "test_terminal"}

    def set_defense_position(self, player, position):
        self.commits.append((player, position))
        return SimpleNamespace(target=SimpleNamespace(x=0., y=0.), qpos_target=SimpleNamespace(x=0., y=0.))


class TestPotentialMax(unittest.TestCase):
    def test_global_maximum_and_first_flat_tie(self):
        attack, defense, details = potential_max_distributions(PHI)
        np.testing.assert_array_equal(attack, [0, 0, 0, 1, 0])
        np.testing.assert_array_equal(defense, [0, 0, 1])
        self.assertEqual((details["attack"], details["defense"]), ("right_short", "right"))
        self.assertEqual(details["maximum_potential"], 20)
        tied = np.zeros((5, 3))
        tied[0, 2] = tied[1, 0] = 7
        a, d, _ = potential_max_distributions(tied)
        self.assertEqual((a.argmax(), d.argmax()), (0, 2))
        a, d, _ = potential_max_distributions(np.zeros((5, 3)))
        self.assertEqual((a.argmax(), d.argmax()), (0, 0))

    def test_bad_matrix_and_threshold_rejected(self):
        for matrix in [np.zeros((3, 5)), np.full((5, 3), np.nan), np.full((5, 3), np.inf)]:
            with self.assertRaises(ValueError):
                potential_max_distributions(matrix)
        for threshold in [-1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                validate_regret_threshold(threshold)

    def test_current_input_schema_and_action_order(self):
        model = TableModel(PHI)
        state = np.arange(74, dtype=np.float32)
        original = state.copy()
        np.testing.assert_array_equal(potential_matrix(model, state), PHI)
        inputs = model.inputs[0]
        self.assertEqual(inputs.shape, (15, 82))
        np.testing.assert_array_equal(inputs[:, :74], np.tile(state, (15, 1)))
        for index in range(15):
            np.testing.assert_array_equal(inputs[index, 74:79], np.eye(5)[index // 3])
            np.testing.assert_array_equal(inputs[index, 79:], np.eye(3)[index % 3])
        np.testing.assert_array_equal(state, original)
        self.assertFalse(model.training)
        with self.assertRaises(ValueError):
            potential_matrix(model, np.zeros(76))

    def test_original_q_regret_signs(self):
        regrets = potential_max_q_regrets(Q, 3, 2)
        self.assertAlmostEqual(regrets["attacker_regret"], .7)
        self.assertAlmostEqual(regrets["defender_regret"], .5)
        self.assertEqual(potential_max_q_regrets(np.zeros((5, 3)), 0, 0),
                         dict(attacker_regret=0., defender_regret=0.))

    def run_matchup(self, max_player, opponent_policy):
        env = CrossingEnv()
        policies = {
            max_player: dict(attack="potential_max", defense="potential_max"),
            3 - max_player: dict(attack=opponent_policy, defense=opponent_policy),
        }
        q_model, phi_model = TableModel(Q), TableModel(PHI)
        original_solver = evaluator.solve_zero_sum_matrix
        solver_effect = original_solver if opponent_policy in ("nash_q", "br_q") else AssertionError("Nash must not be called")
        with patch.object(evaluator, "build_ppo_action", return_value=np.zeros(18)), \
             patch.object(evaluator, "_capture_env_step", side_effect=lambda env, action: (env.step(action), [])), \
             patch.object(evaluator, "state_for_attacker", side_effect=lambda obs, info, player: np.full(74, player)), \
             patch.object(evaluator, "position_snapshot", return_value=dict(world_x=0, world_y=0, qpos_x=0, qpos_y=0)), \
             patch.object(evaluator, "defense_action_diagnostics", return_value={}), \
             patch.object(evaluator, "solve_zero_sum_matrix", side_effect=solver_effect), \
             patch.object(evaluator, "potential_ibr", side_effect=AssertionError("IBR must not be called")), \
             patch.object(evaluator, "potential_adaptive_action", side_effect=AssertionError("Frequency estimates must not be used")):
            episode, decisions = evaluator.run_episode(
                episode_index=0, env=env, ppo=None, model=q_model, potential_model=phi_model,
                policies=policies, fixed_shots={1: "left", 2: "right"},
                fixed_defenses={1: "left", 2: "center"}, rng=np.random.default_rng(0),
                device="cpu", max_steps=3, greedy_iterations=4, render=False, debug_decisions=False,
            )
        self.assertEqual([r["attacker_id"] for r in decisions], [2, 1])
        self.assertEqual([x[0, 0] for x in phi_model.inputs], [2, 1])
        for row in decisions:
            self.assertEqual(row["nash_solver_status"], "linprog_highs" if opponent_policy in ("nash_q", "br_q") else "not_computed")
            if row["attacker_id"] == max_player:
                self.assertEqual(row["attack_skill"], "right_short")
                self.assertEqual(row["attacker_policy_probabilities"], [0, 0, 0, 1, 0])
            else:
                self.assertEqual(row["defense_preposition"], "right")
                self.assertEqual(row["defender_policy_probabilities"], [0, 0, 1])
        return episode, decisions

    def test_both_roles_max_and_opponents_requiring_nash(self):
        for opponent in ["potential_max", "nash_q", "br_q"]:
            with self.subTest(opponent=opponent):
                _, decisions = self.run_matchup(1, opponent)
                if opponent == "potential_max":
                    for row in decisions:
                        self.assertEqual((row["attack_skill"], row["defense_preposition"]), ("right_short", "right"))
                    self.assertEqual(len(potential_max_regret_rows(decisions, .05)), 2)

    def test_roles_against_random_fixed_and_potential_minimax_and_csv(self):
        for player in [1, 2]:
            for opponent in ["random", "fixed", "potential_minimax"]:
                with self.subTest(player=player, opponent=opponent):
                    episode, decisions = self.run_matchup(player, opponent)
                    # Existing summary and CSV serialization must work with skipped Nash diagnostics.
                    summary, _, _ = evaluator.build_summary([episode], decisions, metadata={}, example_count=2)
                    self.assertEqual(summary["average_nash_attacker_probabilities"], {})
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "decisions.csv"
                        evaluator._write_csv(path, [evaluator._decision_csv_row(row) for row in decisions])
                        with path.open() as stream:
                            saved = list(csv.DictReader(stream))
                        for before, after in zip(decisions, saved):
                            for key in ["attacker_policy_probabilities", "defender_policy_probabilities"]:
                                self.assertEqual(json.loads(after[key + "_json"]), before[key])
                            self.assertEqual(after["attack_skill"], before["attack_skill"])
                            self.assertEqual(after["defense_preposition"], before["defense_preposition"])
                        rows = potential_max_regret_rows(decisions, .1)
                        stats = write_regret_diagnostics(Path(directory), rows, .1)
                        self.assertEqual(stats["state_count"], 2)
                        self.assertAlmostEqual(stats["mean_attacker_regret"], .7)
                        self.assertAlmostEqual(stats["mean_defender_regret"], .5)

    def test_regret_frequency_boundary_and_pair_scope(self):
        _, decisions = self.run_matchup(1, "fixed")
        rows = potential_max_regret_rows(decisions, .8)
        self.assertEqual(regret_summary(rows, .8)["both_regrets_below_threshold_frequency"], 1.)
        self.assertNotEqual(rows[0]["potential_max_attack"], rows[0]["executed_attack"])
        boundary = max(rows[0]["attacker_regret"], rows[0]["defender_regret"])
        rows = potential_max_regret_rows(decisions, boundary)
        self.assertEqual(regret_summary(rows, boundary)["both_regrets_below_threshold_frequency"], 0.)
        self.assertIsNone(regret_summary([], .05)["mean_attacker_regret"])

    def test_cli_and_batch_registration(self):
        args = evaluator.build_parser().parse_args(["--p1-policy", "potential_max", "--potential-max-regret-threshold", ".1"])
        self.assertEqual(args.p1_policy, "potential_max")
        self.assertEqual(args.potential_max_regret_threshold, .1)
        args = batch.build_parser().parse_args(["--policies", "potential_max", "--other-opponent", "potential_max"])
        self.assertEqual(args.other_opponent, "potential_max")
        self.assertIn("potential_max", batch.DEFAULT_POLICIES)
        self.assertEqual(batch.DISPLAY_NAMES["potential_max"], "Potential Max")

    def test_batch_writes_sidecar_and_preserves_summary_schema(self):
        class BatchEnv(CrossingEnv):
            def __init__(self, **kwargs):
                super().__init__()
            def close(self):
                pass

        sb3 = types.ModuleType("stable_baselines3")
        sb3.PPO = SimpleNamespace(load=lambda *args: None)
        wrapper = types.ModuleType("v3.env_wrapper")
        wrapper.V3PrePositionEnv = BatchEnv
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict("sys.modules", {"stable_baselines3": sb3, "v3.env_wrapper": wrapper}), \
             patch.object(batch, "load_q_model", return_value=TableModel(Q)), \
             patch.object(batch, "load_potential_model", return_value=TableModel(PHI)), \
             patch.object(evaluator, "build_ppo_action", return_value=np.zeros(18)), \
             patch.object(evaluator, "_capture_env_step", side_effect=lambda env, action: (env.step(action), [])), \
             patch.object(evaluator, "state_for_attacker", return_value=np.zeros(74)), \
             patch.object(evaluator, "position_snapshot", return_value=dict(world_x=0, world_y=0, qpos_x=0, qpos_y=0)), \
             patch.object(evaluator, "defense_action_diagnostics", return_value={}), \
             patch.object(evaluator, "solve_zero_sum_matrix", side_effect=AssertionError("Unexpected Nash solve")), \
             patch("builtins.print"):
            self.assertEqual(batch.main(["--policies", "potential_max", "--episodes", "1", "--both-sides",
                                         "--other-opponent", "potential_minimax", "--output-dir", directory]), 0)
            path = Path(directory)
            summary = json.loads((path / "batch_policy_eval_summary.json").read_text())
            self.assertEqual(set(summary["summary_table"][0]),
                             {"Method", "Static mean", "Static minimum", "Random", "IBR", "Other proposed"})
            stats = json.loads((path / "potential_max_regret_summary.json").read_text())
            self.assertEqual(stats["state_count"], 72)
            self.assertEqual(len(stats["matchups"]), 36)
            self.assertAlmostEqual(stats["mean_attacker_regret"], .7)

    @unittest.skipUnless(Path("models/v3/potential_5k_d7.pth").exists(), "Local trained checkpoint unavailable")
    def test_existing_checkpoint_uses_same_inference_path(self):
        model = load_potential_model("models/v3/potential_5k_d7.pth")
        self.assertEqual(model.fc[0].in_features, 82)
        state = np.zeros(74, dtype=np.float32)
        state[0], state[18], state[36:39] = -.5, 3.5, [1.5, .1, .8]
        matrix = potential_matrix(model, state)
        attack, defense, _ = potential_max_distributions(matrix)
        self.assertEqual((attack.argmax(), defense.argmax()), np.unravel_index(np.argmax(matrix), (5, 3)))


if __name__ == "__main__":
    unittest.main()
