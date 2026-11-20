import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from fullbody.spatial_skills import capture_reference
from fullbody.tools_spatial import schemas
from robot_tools import RobotToolkit


class SpatialSkillTest(unittest.TestCase):
    def test_spatial_schema_registered(self):
        toolkit=RobotToolkit(None)
        self.assertTrue({'robot_spatial_skill_list','robot_spatial_skill_get','robot_spatial_skill_run'}<=toolkit.names())
        self.assertEqual({x['function']['name'] for x in toolkit.schemas()},toolkit.names())

    def test_capture_has_absolute_keypoints(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            ref=Path('motions/fullbody/reference.npz')
            if not ref.is_file(): self.skipTest('reference not present')
            path=capture_reference(ref,out)
            with np.load(path,allow_pickle=False) as z:
                self.assertEqual(z['positions'].shape[2],3)
                self.assertEqual(z['positions'].shape[1],len(z['keypoint_names']))
                self.assertTrue(np.isfinite(z['positions']).all())
                self.assertEqual(z['coordinate_system'].item(),'world_absolute_g1_meters')


if __name__=='__main__': unittest.main()
