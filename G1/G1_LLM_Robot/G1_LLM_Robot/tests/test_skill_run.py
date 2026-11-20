import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fullbody.skills import register
from fullbody.jobs import TrainingJobs


class SkillRunTest(unittest.TestCase):
    def test_training_status_contains_skill_after_pass(self):
        with tempfile.TemporaryDirectory() as d, patch('fullbody.skills.DB',Path(d)/'registry.json'):
            root=Path(d); (root/'training').mkdir(); (root/'training/policy.pt').write_bytes(b'x'); (root/'reference.npz').write_bytes(b'x')
            item=register('skill',root,{'fallen':False,'reference_completed':True,'max_root_xy_error_m':.05,'max_root_speed_mps':.4,'joint_mse':.02,'completed':True})
            self.assertTrue(item['safe_to_run'])
            self.assertEqual(item['checkpoint'],str((root/'training/policy.pt').resolve()))

    def test_schema_names(self):
        from robot_tools import RobotToolkit
        toolkit=RobotToolkit(None)
        self.assertGreaterEqual(len(toolkit.names()),22)
        self.assertTrue({'robot_skill_list','robot_skill_get','robot_skill_run'}<=toolkit.names())


if __name__=='__main__': unittest.main()
