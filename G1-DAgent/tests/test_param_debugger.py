import unittest

from param_debugger import ParameterDebugger


class ParameterDebuggerTest(unittest.TestCase):
    def record(self):
        return {
            'job_id': 'job', 'outcome': 'failed', 'failure_reason': 'root_height',
            'reference': r'E:\TEST_RECORD\G1_Project\G1\motions\demo\reference.npz',
            'config': {
                'motion_gain': 1.35, 'leg_weight': .15, 'arm_weight': 1.0,
                'rl_pd_influence': .6, 'joint_assist_gain': .18,
                'hand_rl_pd_range': [.4, .6], 'foot_rl_pd_range': [.5, .9],
            },
            'evaluation': {
                'fallen': True, 'reference_completed': False,
                'completion_fraction': .2, 'joint_mse': .2,
                'max_root_xy_error_m': .4, 'termination_reason': 'root_height',
            },
        }

    def test_changes_are_bounded_and_skip_original(self):
        proposal = ParameterDebugger().propose(self.record())
        self.assertTrue(proposal['changes'])
        for change in proposal['changes']:
            self.assertLessEqual(change['new'], 2.0)
            self.assertGreaterEqual(change['new'], 0.0)
        self.assertIn('train_balanced', proposal['launch_command'])
        self.assertTrue(proposal['requires_user_confirmation'])

    def test_success_does_not_force_changes(self):
        record = self.record()
        record['outcome'] = 'success'
        record['evaluation'] = {
            'fallen': False, 'reference_completed': True,
            'completion_fraction': 1.0, 'joint_mse': .01,
            'max_root_xy_error_m': .05,
        }
        proposal = ParameterDebugger().propose(record)
        self.assertEqual(proposal['changes'], [])
        self.assertTrue(proposal['directions'])


if __name__ == '__main__':
    unittest.main()
