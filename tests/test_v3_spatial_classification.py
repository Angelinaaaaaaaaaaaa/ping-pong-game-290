"""Landing-region and position-bin tests without simulator dependencies."""
import unittest
import numpy as np
import pandas as pd
from v3.human_comparison.spatial import (
    Regions, FourQuadrantsCenterOverride, region_from_dict, CATEGORIES,
    position_labels, conditional_statistics,
)


class SpatialClassificationTests(unittest.TestCase):
    def test_complete_quadrants_center_priority_and_boundaries(self):
        r=FourQuadrantsCenterOverride();r.validate()
        self.assertEqual(r.classify([.1,.51,.1,.51],[.8,.8,-.8,-.8]).tolist(),
                         ['left_short','left','right_short','right'])
        self.assertEqual(r.classify([.20,.70,.51,.1,1],[.35,-.35,0,0,0]).tolist(),
                         ['center_safe','center_safe','center_safe','left_short','left'])
        self.assertEqual(r.classify([.1,.1],[0,-1e-10]).tolist(),['left_short','right_short'])
        d,y=np.meshgrid(np.linspace(0,1,501),np.linspace(-1,1,501))
        labels=r.classify(d,y)
        self.assertTrue(np.isin(labels,CATEGORIES[:5]).all())
        self.assertEqual(r.classify([-1e-8,1.01,np.nan,.5],[0,0,0,1.01]).tolist(),['invalid/outside']*4)


    def test_complete_quadrants_config_and_five_probabilities(self):
        r=region_from_dict(dict(name='four_quadrants_center_override',depth_split=.6,lateral_split=.1,
                               center_depth_min=.3,center_depth_max=.4,center_lateral_min=-.1,center_lateral_max=.2))
        r.validate()
        self.assertEqual(r.classify([.55,.6,.35],[.2,.2,.1]).tolist(),['left_short','left','center_safe'])
        f=pd.DataFrame(dict(dataset=['human']*5,p1_own_lateral=[0]*5,p2_own_lateral=[0]*5,
            valid_landing=[True]*5,landing_observed=[True]*5,depth=[.1,.9,.4,.1,.9],lateral=[.8,.8,0,-.8,-.8]))
        stats=conditional_statistics(f,FourQuadrantsCenterOverride())
        self.assertEqual(len(stats),45)
        cell=stats[(stats.p1_position=='center')&(stats.p2_position=='center')]
        np.testing.assert_allclose(cell.probability_valid,[.2]*5)
        self.assertTrue(stats[stats.n_valid_landings==0].probability_valid.isna().all())


    def test_unknown_outcomes_and_mapped_denominators(self):
        f=pd.DataFrame(dict(dataset=['human']*2,p1_own_lateral=[0,0],p2_own_lateral=[0,0],valid_landing=[True,True],landing_observed=[True,True],depth=[.4,.95],lateral=[0,0]))
        stats=conditional_statistics(f,Regions());cell=stats[(stats.p1_position=='center')&(stats.p2_position=='center')]
        self.assertAlmostEqual(cell.probability_valid.sum(),1)
        self.assertAlmostEqual(cell.probability_mapped.sum(),1)
        f=f.iloc[:1].copy();f['dataset']='robot';f['valid_landing']=pd.NA;f['landing_observed']=False
        stats=conditional_statistics(f,Regions());self.assertTrue(stats.n_valid_landings.isna().all());self.assertTrue(stats.region_count.isna().all())


    def test_center_bounded_and_ambiguity_preserved(self):
        r=Regions();r.validate()
        self.assertEqual(r.classify([.4,.95,.05],[0,0,0]).tolist(),['center_safe','other/unmapped','other/unmapped'])
        self.assertEqual(r.classify([.3,.8,.5],[.7,-.7,.7]).tolist(),['left_short','right','other/unmapped'])
        self.assertEqual(r.classify([1.1],[0]).tolist(),['invalid/outside'])


    def test_identical_normalized_position_bands(self):
        for band in [.05,.15,.25]:
            y=band/.7625
            self.assertEqual(position_labels([y,-y,y+.00001],band).tolist(),['center','center','left'])



if __name__ == "__main__":
    unittest.main()
