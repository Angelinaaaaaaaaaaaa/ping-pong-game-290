import unittest
import numpy as np
import pandas as pd
import io
import tempfile
import zipfile
from pathlib import Path
from v3.human_comparison.data import RallyStore
from v3.human_comparison.events import (UNIT_M, HALF_WIDTH, HALF_LENGTH, TABLE_Z, Config,
    starts, normalized_landing, position_bin, target_prototypes, landing_category, extract)
from v3.human_comparison.analyze_human import conditional_tables
from nash_skills.skills import SKILL_NAMES


class HumanComparisonTests(unittest.TestCase):
    def test_zip_directory_sampling_equivalence(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);release=root/'release_data';buffer=io.BytesIO()
            np.save(buffer,np.zeros((3,91,3)));blob=buffer.getvalue()
            with zipfile.ZipFile(root/'release_data.zip','w') as archive:
                for match in range(3):
                    for rally in range(2):
                        relative=Path(f'match{match}/match{match}_{rally}/match{match}_{rally}_recon.npy')
                        file=release/relative;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(blob)
                        archive.writestr('release_data/'+str(relative),blob)
            directory=RallyStore(release);archive=RallyStore(root/'release_data.zip')
            chosen=directory.sample(3,42)
            self.assertEqual(chosen,directory.sample(3,42))
            self.assertEqual(chosen,[n.removeprefix('release_data/') for n in archive.sample(3,42)])
            self.assertEqual(len({Path(n).parent.parent.name for n in chosen}),3)
            self.assertEqual(archive.array(archive.sample(1,42)[0]).shape,(3,91,3))
            directory.close();archive.close()

    def test_orientation_and_units(self):
        self.assertAlmostEqual(HALF_LENGTH*2,2.7432)
        np.testing.assert_allclose(normalized_landing([HALF_LENGTH/2,HALF_WIDTH/2],1),[.5,.5])
        np.testing.assert_allclose(normalized_landing([-HALF_LENGTH/2,-HALF_WIDTH/2],2),[.5,.5])
        self.assertEqual(position_bin(.2,1,.1),'left')
        self.assertEqual(position_bin(-.2,2,.1),'left')
        self.assertEqual(position_bin(.2,2,.1),'right')
        self.assertEqual(position_bin(HALF_WIDTH*.1,1,.1),'center')

    def test_contact_deduplication(self):
        np.testing.assert_array_equal(starts([0,1,1,0,0,1,1,0]),[1,5])

    def test_provisional_category_prototypes_and_exterior(self):
        for point,name in zip(target_prototypes('aggressive'),SKILL_NAMES):
            self.assertEqual(landing_category(point),name)
        self.assertIsNone(landing_category([-.1,.1]))
        self.assertIsNone(landing_category([.5,1.01]))
        targets=target_prototypes('aggressive')
        self.assertLess(targets[1,0],targets[0,0])
        self.assertLess(targets[3,0],targets[4,0])

    def test_probability_normalization_and_empty_cells(self):
        events=pd.DataFrame([dict(attacker_id=1,classified=True,human_skill_proxy=s,p1_position='left',p2_position='center') for s in ['left','left','right']])
        joint,_,freq,vectors=conditional_tables(events,[1])
        cell=joint[(joint.p1_position=='left')&(joint.p2_position=='center')]
        self.assertAlmostEqual(cell.empirical_probability.sum(),1.)
        self.assertEqual(cell.selected_count.sum(),3)
        self.assertTrue((cell.n_policy_vectors==0).all())
        self.assertTrue(joint[joint.n_decisions==0].empirical_probability.isna().all())
        self.assertAlmostEqual(freq.condition_frequency.sum(),1.)
        self.assertIsNone(next(x for x in vectors if x['n']==0)['empirical_probability_vector'])

    def test_hit_crossing_bounce_order_and_missing_path_rejection(self):
        scene=np.zeros((20,91,3));scene[0,0]=[30,20,20]
        scene[:,2:46,0]=-.9/UNIT_M;scene[:,46:90,0]=.9/UNIT_M
        scene[:,90,0]=np.array([-1,-1.1,-1.2,-.95,-.7,-.45,-.2,.05,.3,.5,.6,.7,.8,.9,1.,1.05,1.1,1.2,1.,.8])/UNIT_M
        scene[:,90,2]=1.1/UNIT_M;scene[8:11,90,2]=np.array([.85,TABLE_Z,.9])/UNIT_M
        scene[2,1,0]=1;scene[17,1,1]=1;scene[9,1,2]=1
        events=extract(scene,'synthetic',Config());first=events[0]
        self.assertTrue(first['classified'])
        self.assertLess(first['contact_frame'],first['crossing_frame'])
        self.assertLess(first['crossing_frame'],first['bounce_frame'])
        mirrored=scene.copy()
        mirrored[:,2:46]=scene[:,46:90]
        mirrored[:,46:90]=scene[:,2:46]
        mirrored[:,2:,:2]*=-1
        mirrored[:,1,0]=scene[:,1,1];mirrored[:,1,1]=scene[:,1,0]
        mirrored_first=extract(mirrored,'synthetic_p2',Config())[0]
        self.assertTrue(mirrored_first['classified'])
        self.assertEqual(mirrored_first['attacker_id'],2)
        self.assertEqual(first['human_skill_proxy'],mirrored_first['human_skill_proxy'])
        scene[5,90]=np.nan
        first=extract(scene,'synthetic_missing',Config())[0]
        self.assertFalse(first['classified'])
        self.assertEqual(first['exclusion_reason'],'missing_ball_in_hit_crossing_bounce_path')


if __name__=='__main__':unittest.main()
