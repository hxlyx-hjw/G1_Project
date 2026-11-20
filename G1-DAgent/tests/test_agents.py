import tempfile
import unittest
from pathlib import Path

from agents import build_agents
from orchestrator import Orchestrator
from status_reader import JobStatusReader


class AgentTest(unittest.TestCase):
    def test_offline_agents_work_without_api_key(self):
        status, debug = build_agents(use_llm=False)
        record = {
            'job_id': 'job', 'state': 'completed', 'outcome': 'failed',
            'evaluation': {'fallen': True, 'reference_completed': False,
                           'joint_mse': .2, 'max_root_xy_error_m': .4},
            'config': {}, 'last_metric': {},
        }
        s = status.analyze(record)
        d = debug.debug(record)
        self.assertIsNone(s['llm_analysis'])
        self.assertTrue(s['risks'])
        self.assertIn('proposal', d)
        self.assertTrue(d['proposal']['changes'])

    def test_orchestrator_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'runs'
            job = root / 'chat_training' / 'job'
            job.mkdir(parents=True)
            (job / 'status.json').write_text(
                '{"job_id":"job","state":"completed","evaluation":'
                '{"fallen":false,"reference_completed":true,"joint_mse":0.02}}',
                encoding='utf-8')
            (job / 'config.json').write_text('{}', encoding='utf-8')
            report = Orchestrator(root, use_llm=False).run('job')
            self.assertEqual(report['job_id'], 'job')
            self.assertTrue(Path(report['report_paths']['markdown']).is_file())


if __name__ == '__main__':
    unittest.main()
