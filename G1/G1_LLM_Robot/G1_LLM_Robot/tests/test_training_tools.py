"""聊天训练工具的参数、进程管理、取消和schema回归。"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock,patch
from fullbody.jobs import TrainingJobs,atomic_json
from robot_tools import RobotToolkit
from proactive import ProactiveReporter


class TrainingToolsTest(unittest.TestCase):
    def test_schema_and_validation(self):
        tools=RobotToolkit(None)
        self.assertEqual({t['function']['name'] for t in tools.schemas()},tools.names())
        with self.assertRaises(FileNotFoundError): tools.training.start('missing_video.mp4')
        self.assertIn('没有此训练任务',tools.dispatch('robot_training_status',{}))

    def test_process_status_cancel_and_viewer(self):
        with tempfile.TemporaryDirectory() as tmp, patch('fullbody.jobs.ROOT',Path(tmp)), patch('fullbody.jobs.subprocess.Popen') as spawn:
            video=Path(tmp)/'video.mp4'; video.write_bytes(b'test')
            process=Mock(); process.poll.return_value=None; spawn.return_value=process
            jobs=TrainingJobs()
            started=[]
            jobs.on_job_started=started.append
            result=jobs.start(str(video),iterations=1,device='auto',render=True)
            self.assertEqual(spawn.call_count,2)
            self.assertTrue(result['worker_running'])
            self.assertIn('started_at',result)
            self.assertIn('elapsed_text',result)
            self.assertGreaterEqual(result['elapsed_seconds'],0)
            self.assertEqual(started,[result['job_id']])
            job=result['job_id']; folder=Path(result['output_dir'])
            self.assertTrue((folder/'config.json').exists())
            with self.assertRaises(RuntimeError): jobs.start(str(video),render=False)
            jobs.stop(job); self.assertTrue((folder/'STOP').exists())
            atomic_json(folder/'status.json',{'state':'completed','duration_seconds':12.5,'evaluation':{'fallen':False}})
            process.poll.return_value=0
            completed=jobs.status(job)
            self.assertEqual(completed['state'],'completed')
            self.assertEqual(completed['duration_seconds'],12.5)
            self.assertIn('duration_text',completed)
            jobs.open_viewer(job)
            self.assertEqual(spawn.call_count,3)

    def test_unexpected_worker_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            process=Mock(); process.poll.return_value=7
            jobs=TrainingJobs(); jobs.latest='x'; jobs.jobs['x']={'folder':Path(tmp),'worker':process,'viewer':None}
            self.assertEqual(jobs.status()['state'],'failed')
            self.assertEqual(jobs.status()['exit_code'],7)

    def test_proactive_report_tool(self):
        tools=RobotToolkit(None)
        self.assertIn('robot_proactive_report',tools.names())
        tools.proactive_reporter=ProactiveReporter(
            lambda job:{'job_id':job,'state':'training',
                        'elapsed_text':'1分','iteration':1,'total_iterations':2},
            lambda text:None)
        result=json.loads(tools.dispatch('robot_proactive_report',
                                         {'enabled':True,'interval_minutes':5}))
        self.assertTrue(result['enabled'])
        self.assertEqual(result['interval_minutes'],5.0)


if __name__=='__main__': unittest.main()
