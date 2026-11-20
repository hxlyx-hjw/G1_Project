import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import failure_memory
from robot_tools import RobotToolkit


class FailureMemoryTest(unittest.TestCase):
    def test_record_and_retrieve_structured_failure(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(failure_memory, 'MEMORY_PATH',
                             Path(tmp) / 'failure_memory.jsonl'):
            first = failure_memory.record(
                'robot_goto', {'x': 1.0, 'y': 0.0}, 'partial',
                'goal_not_reached', {'distance_m': 0.4},
                '增大 timeout 或拆成两段', {'x': 0}, {'x': .6}, '未到达')
            failure_memory.record(
                'robot_walk', {'vx': .5}, 'failed', 'fallen',
                {'fallen': True}, '降低速度', {}, {}, '摔倒')
            rows = failure_memory.retrieve(tool='robot_goto', limit=5)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['id'], first['id'])
            self.assertEqual(rows[0]['reason'], 'goal_not_reached')
            self.assertIn('timeout', rows[0]['correction'])
            summary = failure_memory.summary(tool='robot_goto')
            self.assertEqual(summary[0]['tool'], 'robot_goto')

    def test_action_postcondition_verification(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(failure_memory, 'MEMORY_PATH',
                             Path(tmp) / 'failure_memory.jsonl'):
            toolkit = RobotToolkit(None)
            before = {'x': 0., 'y': 0., 'yaw': 0., 'z': .8,
                      'speed': 0., 'fallen': False,
                      'reliability': {'quality': 1.0}}
            after = {'x': .4, 'y': 0., 'yaw': 0., 'z': .8,
                     'speed': .1, 'fallen': False,
                     'reliability': {'quality': .9}}
            result = toolkit._verify_action(
                'robot_goto', {'x': 1., 'y': 0.}, before, after, '未到达')
            self.assertEqual(result['outcome'], 'partial')
            self.assertEqual(result['reason'], 'goal_not_reached')
            self.assertGreater(result['evidence']['distance_m'], .1)
            good = dict(after, x=1.0)
            result = toolkit._verify_action(
                'robot_goto', {'x': 1., 'y': 0.}, before, good, '已到达')
            self.assertTrue(result['success'])
            fallen = dict(after, fallen=True)
            result = toolkit._verify_action(
                'robot_walk', {'vx': .5, 'seconds': 1.}, before, fallen, '摔倒')
            self.assertEqual(result['reason'], 'fallen')
            self.assertIn('复位', result['correction'])

    def test_status_contains_reliability(self):
        from robot_bridge import RobotSession
        with RobotSession(verbose=False) as bot:
            state = bot.status()
            self.assertIn('reliability', state)
            self.assertIn('quality', state['reliability'])
            self.assertIn('稳定性', bot.describe())


if __name__ == '__main__':
    unittest.main()
