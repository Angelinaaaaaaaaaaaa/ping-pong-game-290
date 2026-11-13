import unittest
import numpy as np
import pandas as pd
from v3.human_comparison import analyze_random_similarity as comparison

from v3.human_comparison.analyze_random_similarity import (
    all_metrics, context_indices, count_clusters, paired_bootstrap, rebin_positions,
)


class RandomSimilarityTests(unittest.TestCase):
    def test_potential_max_retains_actual_model_names(self):
        try:
            comparison.configure_potential_policy('potential_max')
            self.assertEqual(comparison.LEARNED,('nash_q','potential_max'))
            self.assertIn('random_p2_vs_potential_max',comparison.MODELS)
            self.assertNotIn('potential_ibr',comparison.MODELS)
            self.assertEqual(comparison.NAMES['potential_max'],'Potential Max P1')
        finally:
            comparison.configure_potential_policy('potential_ibr')

    def test_wider_position_band_retains_actions_and_frame(self):
        f = pd.DataFrame(dict(attacker_id=[1,2,1,2],
            p1_own_lateral=np.array([.04,.10,.20,.30])/.7625,
            p2_own_lateral=-np.array([.04,.10,.20,.30])/.7625,
            attack_skill=['left','center_safe','right','left_short']))
        narrow = rebin_positions(f,.05); medium = rebin_positions(f,.15); wide = rebin_positions(f,.25)
        self.assertEqual(narrow.p1_position.tolist(),['center','left','left','left'])
        self.assertEqual(medium.p1_position.tolist(),['center','center','left','left'])
        self.assertEqual(wide.p1_position.tolist(),['center','center','center','left'])
        self.assertEqual(wide.p2_position.tolist(),['center','center','center','right'])
        self.assertEqual(wide.opponent_position.iloc[3],'left') # P2 hitter faces P1.
        pd.testing.assert_series_equal(narrow.attack_skill,wide.attack_skill)
        self.assertNotIn('p1_position',f)

    def test_p2_role_swap_preserves_physical_pairs(self):
        f = pd.DataFrame(dict(attacker_id=[1, 2], p1_position=['left']*2,
            p2_position=['right']*2, attack_skill=['left', 'right'], cluster=['a', 'a']))
        np.testing.assert_array_equal(context_indices(f), [[2, 11, 14], [2, 9, 18]])
        _, counts = count_clusters(f)
        self.assertEqual(counts[:, :9].sum(), 2)
        self.assertEqual(counts[:, 9:12].sum(), 2)
        self.assertEqual(counts[:, 12:].sum(), 2)

    def test_uniform_has_distances_but_no_correlation(self):
        p = np.array([.4, .3, .2, .1, 0.])
        s = all_metrics(p, np.full(5, .2))
        self.assertTrue(np.isfinite(s['js_divergence_bits']))
        self.assertAlmostEqual(float(s['total_variation']), .3)
        self.assertTrue(np.isnan(s['spearman_rho']))
        self.assertTrue(np.isnan(s['pearson_r']))
        s = all_metrics([1., 0, 0, 0, 0], [0., 1, 0, 0, 0])
        self.assertAlmostEqual(float(s['js_divergence_bits']), 1.)
        self.assertAlmostEqual(float(s['pearson_r']), -.25)

    def test_shared_episode_bootstrap_and_zero_contexts(self):
        counts = np.zeros((2, 21, 5), dtype=int)
        counts[0, 0, 0] = 2; counts[1, 0, 4] = 1
        result = paired_bootstrap({'learned': counts, 'random': counts*3}, 100, np.random.default_rng(42))
        np.testing.assert_allclose(result['learned'], result['random'], equal_nan=True)
        self.assertTrue(np.isnan(result['random'][:, 1:]).all())
        second = paired_bootstrap({'learned': counts, 'random': counts*3}, 100, np.random.default_rng(42))
        np.testing.assert_array_equal(result['learned'], second['learned'])


if __name__ == '__main__': unittest.main()
