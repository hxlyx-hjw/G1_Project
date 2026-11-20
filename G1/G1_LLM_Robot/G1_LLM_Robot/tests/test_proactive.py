import tempfile
import time
import unittest
from pathlib import Path

from proactive import ProactiveReporter, format_duration


class ProactiveReporterTest(unittest.TestCase):
    def test_duration_and_report_content(self):
        self.assertEqual(format_duration(0), '0秒')
        self.assertEqual(format_duration(65), '1分05秒')
        self.assertEqual(format_duration(3661), '1小时01分01秒')
        status = {'job_id': 'job_1', 'state': 'training', 'elapsed_seconds': 72,
                  'elapsed_text': '1分12秒', 'training_seconds': 60,
                  'training_text': '1分00秒', 'iteration': 3, 'total_iterations': 10,
                  'metrics': {'mean_reward': .5},
                  'extraction': {'current': 5, 'total': 10, 'percent': 50}}
        reporter = ProactiveReporter(lambda job: status, lambda text: None)
        text = reporter.build_report(status)
        self.assertIn('已用时间', text)
        self.assertIn('纯训练用时', text)
        self.assertIn('3/10', text)
        self.assertIn('骨架提取', text)
        self.assertIn('平均奖励', text)

    def test_send_now_writes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            sent = []
            status = {'job_id': 'job_2', 'state': 'completed',
                      'duration_seconds': 125, 'duration_text': '2分05秒',
                      'output_dir': directory}
            reporter = ProactiveReporter(
                lambda job: status, sent.append, interval_seconds=600)
            result = reporter.send_now('job_2')
            self.assertTrue(sent and 'job_2' in sent[0])
            self.assertTrue(Path(result['path']).is_file())
            self.assertIn('总耗时', Path(result['path']).read_text(encoding='utf-8'))

    def test_interval_report_stops_after_terminal_state(self):
        with tempfile.TemporaryDirectory() as directory:
            sent = []
            status = {'job_id': 'job_3', 'state': 'completed',
                      'duration_seconds': 5, 'duration_text': '5秒',
                      'output_dir': directory}
            reporter = ProactiveReporter(
                lambda job: status, sent.append, interval_seconds=1.0)
            reporter.watch('job_3')
            deadline = time.monotonic() + 3.0
            while not sent and time.monotonic() < deadline:
                time.sleep(.05)
            reporter.stop()
            self.assertTrue(sent)
            self.assertIn('completed', sent[0])


if __name__ == '__main__':
    unittest.main()
