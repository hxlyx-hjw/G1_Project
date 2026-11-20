import unittest
from unittest.mock import Mock,patch
from types import SimpleNamespace
from fullbody.self_check import diagnose
from harness_search import search_web


class SelfCheckTest(unittest.TestCase):
    def test_uses_original_harness_client(self):
        client=Mock(); client.search.return_value={'results':[]}
        self.assertEqual(search_web('wave',client=client),{'results':[]})
        client.search.assert_called_once_with(query='wave',search_depth='advanced',max_results=5,include_answer=True)

    def test_choose_only_improved_candidate(self):
        env=SimpleNamespace(video_leg_weight=.5,video_arm_weight=1.)
        def score(policy,env,device):
            ok=env.video_leg_weight<.5
            return {'completed':ok,'fallen':not ok,'steps':100 if ok else 20,'max_root_xy_error_m':.05 if ok else .6,
                'max_root_speed_mps':.4,'joint_mse':.02,'leg_mse':.02,'arm_mse':.01}
        result=diagnose(None,env,'cpu',score)
        self.assertTrue(result['changed']); self.assertAlmostEqual(env.video_leg_weight,.45)
        self.assertEqual(env.video_arm_weight,1.)

    def test_bad_model_suggestion_cannot_bypass_physics(self):
        env=SimpleNamespace(video_leg_weight=.5,video_arm_weight=1.)
        def score(*args): return {'completed':False,'fallen':True,'steps':20,'max_root_xy_error_m':.6,'max_root_speed_mps':2.,'joint_mse':.1}
        with patch('fullbody.self_check.model_advice',return_value=(.6,1.,'increase')):
            result=diagnose(None,env,'cpu',score,allow_model=True)
        self.assertFalse(result['changed']); self.assertEqual(env.video_leg_weight,.5)


if __name__=='__main__': unittest.main()
