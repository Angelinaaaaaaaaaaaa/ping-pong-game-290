import unittest

from nash_skills.skills import SKILL_NAMES
from v3.constants import DEFENSE_POSITIONS
from v3.eval_policy_fixed_grid import _build_summary


class TestV3FixedGridAggregation(unittest.TestCase):
    def test_static_summary_pools_seeds_before_taking_minimum(self):
        rows = []
        for attack in SKILL_NAMES:
            for defense in DEFENSE_POSITIONS:
                seed_records = [(0, 2, 1)]
                if attack == SKILL_NAMES[0] and defense == DEFENSE_POSITIONS[0]:
                    # Per-seed rates are 0% and 100%; pooled matchup WR is 90%.
                    seed_records = [(0, 1, 0), (1, 9, 9)]
                for seed, episodes, wins in seed_records:
                    rows.append({
                        "tested_policy": "nash_q",
                        "opponent_type": "static",
                        "fixed_attack_skill": attack,
                        "fixed_defense_position": defense,
                        "seed": seed,
                        "episode_count": episodes,
                        "wins": wins,
                    })

        table, detail = _build_summary(rows, ["nash_q"], other_opponent="potential_adaptive_match")

        expected_mean = (0.9 + 14 * 0.5) / 15
        self.assertAlmostEqual(table[0]["Static mean"], expected_mean)
        self.assertAlmostEqual(table[0]["Static minimum"], 0.5)
        self.assertAlmostEqual(detail["nash_q"]["static_matchup_win_rates"][0]["win_rate"], 0.9)


if __name__ == "__main__":
    unittest.main()
