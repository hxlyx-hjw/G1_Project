"""训练技能库回归：通过评估才能登记，并可直接调用。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fullbody.skills import register,list_skills,get
from fullbody.tools import schemas
from robot_tools import RobotToolkit


class SkillTest(unittest.TestCase):
    def test_registry_requires_pass(self):
        with tempfile.TemporaryDirectory() as d, patch('fullbody.skills.DB',Path(d)/'registry.json'):
            root=Path(d); (root/'training').mkdir(); (root/'training/policy.pt').write_bytes(b'x'); (root/'reference.npz').write_bytes(b'x')
            with self.assertRaises(ValueError): register('bad',root,{'fallen':True,'reference_completed':False})
            item=register('good',root,{'fallen':False,'reference_completed':True,'max_root_xy_error_m':.05,'max_root_speed_mps':.4,'joint_mse':.02})
            self.assertEqual(get(item['skill_id'])['name'],'good')
            self.assertEqual(len(list_skills()),1)

    def test_tools_registered(self):
        toolkit=RobotToolkit(None)
        names=toolkit.names()
        self.assertTrue({'robot_skill_list','robot_skill_get','robot_skill_run'}<=names)
        self.assertEqual({x['function']['name'] for x in toolkit.schemas()},names)


if __name__=='__main__': unittest.main()
