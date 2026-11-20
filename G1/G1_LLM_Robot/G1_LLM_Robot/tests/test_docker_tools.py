import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import fullbody.docker_jobs as docker_jobs


class DockerJobsTest(unittest.TestCase):
    def test_reference_training_summary_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / 'runs'
            inputs = root / 'inputs'
            inputs.mkdir()
            (inputs / 'reference.npz').write_bytes(b'reference')
            job_holder = {}

            def fake_run(command, **kwargs):
                if command[1:3] == ['run', '-d']:
                    return Mock(returncode=0, stdout='container-id\n', stderr='')
                if command[1] == 'inspect':
                    return Mock(returncode=0, stdout='running 0\n', stderr='')
                if 'fullbody.gif_export' in command:
                    index = command.index('fullbody.gif_export')
                    container = command[index + 2]
                    host = runs / Path(container).relative_to(
                        '/workspace/G1_LLM_Robot/runs')
                    host.parent.mkdir(parents=True, exist_ok=True)
                    host.write_bytes(b'gif')
                    status = command[command.index('--status') + 1]
                    status_host = runs / Path(status).relative_to(
                        '/workspace/G1_LLM_Robot/runs')
                    status_host.write_text(json.dumps({'state': 'completed'}))
                    job_holder['gif'] = host
                    return Mock(returncode=0, stdout='', stderr='')
                if command[-1].endswith('.mp4'):
                    host = runs / Path(command[-1]).relative_to(
                        '/workspace/G1_LLM_Robot/runs')
                    host.write_bytes(b'mp4')
                    return Mock(returncode=0, stdout='', stderr='')
                return Mock(returncode=0, stdout='', stderr='')

            with patch.object(docker_jobs, 'RUNS', runs), \
                 patch.object(docker_jobs, 'SMPL_DIR', root / 'missing'), \
                 patch.object(docker_jobs, 'WHAM_DIR', root / 'missing'), \
                 patch.object(docker_jobs, 'ensure_data_dirs', lambda: None), \
                 patch.object(docker_jobs.shutil, 'which', return_value='/usr/bin/docker'), \
                 patch.object(docker_jobs.subprocess, 'run', side_effect=fake_run):
                jobs = docker_jobs.DockerJobs()
                state = jobs.start(str(inputs), iterations=1, device='cuda',
                                   num_envs=1, coordination=False)
                job = state['job_id']
                folder = runs / 'docker_jobs' / job
                config = json.loads((folder / 'config.json').read_text())
                self.assertEqual(config['mode'], 'reference')
                self.assertTrue((folder / 'reference.npz').is_file())

                (folder / 'status.json').write_text(json.dumps({
                    'job_id': job, 'state': 'completed', 'iteration': 1,
                    'total_iterations': 1, 'duration_text': '10秒',
                    'training_text': '7秒', 'metrics': {'mean_reward': .9},
                    'evaluation': {'fallen': False, 'reference_completed': True},
                }))
                (folder / 'training').mkdir()
                (folder / 'training' / 'policy.pt').write_bytes(b'policy')
                summary = jobs.summary(job)
                self.assertTrue(Path(summary['summary_path']).is_file())
                self.assertIn('Docker', summary['summary'])

                result = jobs.export_video(job, format='mp4', fps=10,
                                           max_seconds=5)
                self.assertTrue(Path(result['output']).is_file())
                self.assertTrue(Path(result['gif']).is_file())


if __name__ == '__main__':
    unittest.main()
