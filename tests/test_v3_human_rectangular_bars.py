import unittest
import numpy as np
import pandas as pd

from v3.human_comparison.spatial import FourQuadrantsCenterOverride
from v3.human_comparison.plot_human_rectangular_skill_bars import summarize_humans


class HumanRectangularBarTests(unittest.TestCase):
    def test_reclassifies_coordinates_instead_of_nearest_labels(self):
        f=pd.DataFrame(dict(attacker_id=[1,1,2],p1_position=['left']*3,p2_position=['left']*3,
            depth=[.95,.4,.1],lateral=[0.,0.,-.7],human_skill_proxy=['center_safe']*3,
            human_landing_category=['center_safe']*3))
        classified,stats=summarize_humans(f,FourQuadrantsCenterOverride())
        self.assertEqual(classified.human_landing_category.tolist(),['left','center_safe','right_short'])
        both=stats[(stats.player_cohort=='both_players')&(stats.p1_position=='left')&(stats.p2_position=='left')].set_index('landing_category')
        self.assertEqual(both.n.iloc[0],3)
        self.assertAlmostEqual(both.loc['left','probability'],1/3)
        self.assertAlmostEqual(both.loc['right_short','probability'],1/3)
        self.assertEqual(both['count'].sum(),3)
        self.assertEqual(len(stats),135)
        self.assertTrue(stats[stats.n==0].probability.isna().all())
        self.assertTrue(f.human_landing_category.eq('center_safe').all()) # Source untouched.

    def test_center_override_priority_and_invalid_landings(self):
        f=pd.DataFrame(dict(attacker_id=[2,2],p1_position=['center']*2,p2_position=['center']*2,
            depth=[.2,.7],lateral=[.35,-.35]))
        classified,stats=summarize_humans(f,FourQuadrantsCenterOverride())
        self.assertTrue(classified.human_landing_category.eq('center_safe').all())
        nonempty=stats[stats.n>0].groupby(['player_cohort','p1_position','p2_position']).probability.sum()
        np.testing.assert_allclose(nonempty,1)
        f.loc[0,'depth']=1.1
        with self.assertRaises(ValueError):summarize_humans(f,FourQuadrantsCenterOverride())


if __name__=='__main__':unittest.main()
