import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from v3.human_comparison.analyze_opponent_position import (
    aggregate, bootstrap, cluster_counts, normalize, similarity,
)
from v3.human_comparison.analyze_joint_position import joint_cluster_counts, PAIRS
from v3.human_comparison.analyze_spatial_v2 import human_events
from v3.human_comparison.events import HALF_WIDTH


class OpponentPositionTests(unittest.TestCase):
    def test_incoming_alignment_for_each_human_player(self):
        rows = []
        for index, player in enumerate([1, 2, 1, 2]):
            row = dict(rally_id='match1_1', shot_index=index, attacker_id=player,
                contact_frame=index * 10, interval_end_frame=(index + 1) * 10,
                crossing_frame=index * 10 + 5, valid_net_crossing=True,
                valid_opponent_landing=True, classified=True,
                landing_depth_normalized=.5, landing_lateral_normalized=.6)
            for p in (1, 2):
                row[f'p{p}_crossing_y_m'] = .1 * (index + p)
                row[f'p{p}_pre_contact_y_m'] = .2 * p
            rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name in ('shot_events.csv', 'all_shot_events.csv'):
                pd.DataFrame(rows).to_csv(path / name, index=False)
            h = human_events(path, players=(1, 2))
        self.assertTrue(np.isnan(h.loc[0, 'incoming_frame']))
        self.assertEqual(h.loc[1, 'incoming_frame'], 5)
        self.assertEqual(h.loc[2, 'incoming_frame'], 15)
        self.assertAlmostEqual(h.loc[1, 'p2_incoming_own_lateral'], -.2 / HALF_WIDTH)
        self.assertAlmostEqual(h.loc[2, 'p1_incoming_own_lateral'], .2 / HALF_WIDTH)
        self.assertEqual(h.loc[1, 'lateral'], .6) # Already hitter-relative; no second flip.

    def test_pool_players_uses_actual_defender_not_always_p2(self):
        rows = []
        for dataset in ('human_p1', 'nash_q', 'potential_ibr'):
            for player, opponent, skill in [('P1', 'right', 'left'), ('P2', 'left', 'right')]:
                rows.append(dict(dataset=dataset, analysis_run=dataset, strategy=dataset,
                    analysis_player=player, p1_position='left', p2_position='right',
                    opponent_position=opponent, attack_skill=skill, policy_vector=None))
        joint, opponent = aggregate(pd.DataFrame(rows))
        human_joint = joint[joint.dataset == 'human_p1']
        self.assertEqual(human_joint.selected_count.sum(), 2)
        human_opponent = opponent[opponent.dataset == 'human_p1']
        self.assertEqual(human_opponent[human_opponent.opponent_position == 'left'].selected_count.sum(), 1)
        self.assertEqual(human_opponent[human_opponent.opponent_position == 'right'].selected_count.sum(), 1)

    def test_joint_context_order_and_generalized_bootstrap(self):
        f = pd.DataFrame(dict(cluster=['m1', 'm1', 'm2'],
            p1_position=['left', 'center', 'right'],
            p2_position=['right', 'left', 'center'], attack_skill=['left', 'right', 'center_safe']))
        clusters, counts = joint_cluster_counts(f)
        self.assertEqual(counts.shape, (2, 9, 5))
        self.assertEqual(counts[0, PAIRS.index(('left', 'right')), 0], 1)
        self.assertEqual(counts[0, PAIRS.index(('center', 'left')), 4], 1)
        a = bootstrap(counts, 100, np.random.default_rng(42))
        self.assertEqual(a.shape, (100, 9, 5))
        self.assertTrue(np.isnan(a[:, PAIRS.index(('left', 'left'))]).all())
        self.assertEqual(counts.sum(), 3)

    def test_raw_counts_weight_unequal_contexts(self):
        rows = []
        for dataset in ('human_p1', 'nash_q', 'potential_ibr'):
            for own, skill, n in [('left', 'left', 9), ('center', 'right', 1)]:
                rows.extend(dict(dataset=dataset, analysis_run=dataset, strategy=dataset,
                    analysis_player='P1', p1_position=own, p2_position='left',
                    opponent_position='left', attack_skill=skill, policy_vector=None,
                    cluster=dataset + own) for _ in range(n))
        f = pd.DataFrame(rows)
        joint, opponent = aggregate(f)
        selected = opponent[(opponent.dataset == 'human_p1') & (opponent.opponent_position == 'left')]
        p = selected.set_index('attack_skill').empirical_probability
        self.assertAlmostEqual(p['left'], .9)
        self.assertAlmostEqual(p['right'], .1)
        empty = opponent[opponent.opponent_position == 'right']
        self.assertTrue(empty.empirical_probability.isna().all())

    def test_similarity_endpoints_ties_and_missing(self):
        p = np.array([1., 0, 0, 0, 0]); q = np.array([0., 1, 0, 0, 0])
        s = similarity(p, q)
        self.assertAlmostEqual(float(s['js_divergence_bits']), 1)
        self.assertAlmostEqual(float(s['total_variation']), 1)
        self.assertAlmostEqual(float(s['spearman_rho']), -.25)
        same = similarity(p, p)
        self.assertAlmostEqual(float(same['js_divergence_bits']), 0)
        self.assertAlmostEqual(float(same['spearman_rho']), 1)
        self.assertTrue(np.isnan(similarity(np.full(5, .2), p)['spearman_rho']))
        self.assertTrue(np.isnan(similarity(np.full(5, np.nan), p)['js_divergence_bits']))

    def test_cluster_bootstrap_reproducibility_and_counts(self):
        f = pd.DataFrame(dict(cluster=['match1', 'match1', 'match2'],
            opponent_position=['left', 'right', 'left'], attack_skill=['left', 'right', 'right']))
        clusters, counts = cluster_counts(f)
        self.assertEqual(list(clusters), ['match1', 'match2'])
        np.testing.assert_array_equal(counts.sum(axis=(1, 2)), [2, 1])
        p = normalize(counts.sum(axis=0))
        self.assertAlmostEqual(p[0, 0], .5)
        self.assertTrue(np.isnan(p[1]).all())
        a = bootstrap(counts, 100, np.random.default_rng(42))
        b = bootstrap(counts, 100, np.random.default_rng(42))
        np.testing.assert_array_equal(a, b)
        # Both rows of match1 receive the same resampling multiplicity:
        # whenever the left category is absent, the right condition is absent.
        np.testing.assert_array_equal(a[:, 0, 0] == 0, np.isnan(a[:, 2, 0]))
        self.assertTrue(np.allclose(np.nansum(a[:, 0], axis=1), 1))


if __name__ == '__main__': unittest.main()
