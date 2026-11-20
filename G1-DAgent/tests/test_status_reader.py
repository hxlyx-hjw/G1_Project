import json
import tempfile
import unittest
from pathlib import Path

from status_reader import JobStatusReader


class StatusReaderTest(unittest.TestCase):
    def test_reads_completed_and_running_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            completed = root / 'chat_training' / 'job_done'
            (completed / 'training').mkdir(parents=True)
            (completed / 'evaluation').mkdir()
            (completed / 'status.json').write_text(json.dumps({
                'job_id': 'job_done', 'state': 'completed', 'iteration': 10,
                'total_iterations': 10, 'elapsed_text': '1分',
                'evaluation': {'fallen': False, 'reference_completed': True,
                               'completion_fraction': 1.0, 'joint_mse': .02},
            }), encoding='utf-8')
            (completed / 'config.json').write_text(json.dumps({
                'iterations': 10, 'leg_weight': .15, 'arm_weight': 1.0,
                'motion_gain': 1.35, 'rl_pd_influence': .7,
            }), encoding='utf-8')
            (completed / 'training' / 'metrics.jsonl').write_text(
                '{"iteration": 1, "mean_reward": 0.2}\n'
                '{"iteration": 10, "mean_reward": 0.8}\n', encoding='utf-8')
            running = root / 'docker_jobs' / 'job_run'
            running.mkdir(parents=True)
            (running / 'status.json').write_text(json.dumps({
                'job_id': 'job_run', 'state': 'training', 'iteration': 2,
                'total_iterations': 100,
            }), encoding='utf-8')
            reader = JobStatusReader(root)
            rows = reader.list_jobs()
            self.assertEqual(len(rows), 2)
            done = reader.get('job_done')
            self.assertEqual(done['outcome'], 'success')
            self.assertEqual(done['reward_best'], .8)
            self.assertEqual(reader.get('job_r')['state'], 'training')

    def test_tail_metric_handles_large_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'chat_training' / 'job'
            (folder / 'training').mkdir(parents=True)
            (folder / 'status.json').write_text(
                json.dumps({'job_id': 'job', 'state': 'completed',
                            'evaluation': {'fallen': True}}), encoding='utf-8')
            with (folder / 'training' / 'metrics.jsonl').open('w', encoding='utf-8') as handle:
                for i in range(1000):
                    handle.write(json.dumps({'iteration': i, 'mean_reward': i / 1000}) + '\n')
            record = JobStatusReader(root).get('job')
            self.assertEqual(record['iteration'], 999)
            self.assertEqual(record['outcome'], 'failed')


if __name__ == '__main__':
    unittest.main()
