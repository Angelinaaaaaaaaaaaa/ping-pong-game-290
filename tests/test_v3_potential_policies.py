import unittest

import numpy as np

from v3.potential_policy import (
    new_action_counts,
    potential_adaptive_action,
    potential_minimax_action,
    record_joint_actions,
)


class TestV3PotentialPolicies(unittest.TestCase):
    def test_adaptive_attacker_uses_smoothed_opponent_defense_mix(self):
        phi = np.full((5, 3), -20.0)
        phi[0] = [0.0, 0.0, 10.0]
        phi[1] = [6.0, 6.0, 6.0]
        attack_counts = np.zeros(5, dtype=int)
        defense_counts = np.array([0, 0, 99])

        action, details = potential_adaptive_action(phi, "attacker", attack_counts, defense_counts)

        self.assertEqual(action, 0)
        self.assertEqual(details["selected_action"], "left")
        self.assertAlmostEqual(sum(details["opponent_defense_probabilities"]), 1.0)
        self.assertAlmostEqual(details["opponent_attack_probabilities"][0], 0.2)
        self.assertGreater(
            details["expected_potential_by_action"]["left"],
            details["expected_potential_by_action"]["left_short"],
        )

    def test_adaptive_defender_uses_opponent_attack_mix(self):
        phi = np.full((5, 3), -10.0)
        phi[4] = [1.0, 9.0, 2.0]
        attack_counts = np.array([0, 0, 0, 0, 99])
        defense_counts = np.zeros(3, dtype=int)

        action, details = potential_adaptive_action(phi, "defender", attack_counts, defense_counts)

        self.assertEqual(action, 1)
        self.assertEqual(details["selected_action"], "center")
        self.assertEqual(details["opponent_attack_counts"], attack_counts.tolist())
        self.assertEqual(details["opponent_defense_probabilities"], [1 / 3, 1 / 3, 1 / 3])

    def test_counts_are_separate_and_recorded_only_as_completed_joint_action(self):
        rally_counts = {1: new_action_counts(), 2: new_action_counts()}
        match_counts = {1: new_action_counts(), 2: new_action_counts()}

        record_joint_actions(rally_counts, 1, 4, 2, 2)
        record_joint_actions(match_counts, 1, 4, 2, 2)
        fresh_rally = {1: new_action_counts(), 2: new_action_counts()}

        self.assertEqual(rally_counts[1]["attack"][4], 1)
        self.assertEqual(rally_counts[2]["defense"][2], 1)
        self.assertEqual(match_counts[1]["attack"][4], 1)
        self.assertEqual(fresh_rally[1]["attack"].sum(), 0)
        self.assertEqual(fresh_rally[2]["defense"].sum(), 0)

    def test_minimax_chooses_action_with_best_worst_case_value(self):
        phi = np.array([
            [1.0, 3.0, 0.0],
            [5.0, 6.0, 7.0],
            [2.0, 8.0, 4.0],
            [0.0, 9.0, 2.0],
            [3.0, 7.0, 1.0],
        ])

        attack, attack_details = potential_minimax_action(phi, "attacker")
        defense, defense_details = potential_minimax_action(phi, "defender")

        self.assertEqual(attack, 1)
        self.assertEqual(attack_details["worst_case_potential_by_action"]["left_short"], 5.0)
        self.assertEqual(defense, 1)
        self.assertEqual(defense_details["worst_case_potential_by_action"]["center"], 3.0)

    def test_argmax_ties_choose_first_action(self):
        phi = np.zeros((5, 3), dtype=float)

        attack, _ = potential_minimax_action(phi, "attacker")
        defense, _ = potential_minimax_action(phi, "defender")
        adaptive_attack, _ = potential_adaptive_action(phi, "attacker", np.zeros(5), np.zeros(3))

        self.assertEqual(attack, 0)
        self.assertEqual(defense, 0)
        self.assertEqual(adaptive_attack, 0)


if __name__ == "__main__":
    unittest.main()
