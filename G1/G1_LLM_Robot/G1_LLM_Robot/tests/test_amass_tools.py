import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from data_paths import AMASS_MOTIONS
from fullbody.amass import (build_catalog, dataset_inventory, list_motions,
                            load_catalog, resolve_motion)
from fullbody.jobs import TrainingJobs
from robot_tools import RobotToolkit


class AmassToolsTest(unittest.TestCase):
    def test_catalog_covers_every_converted_motion(self):
        catalog = build_catalog(AMASS_MOTIONS, force=True)
        files = [p for p in AMASS_MOTIONS.rglob('*.npz')
                 if p.name != 'catalog.json']
        self.assertEqual(catalog['count'], len(files))
        self.assertEqual(catalog['count'], 447)
        self.assertIn('ACCAD', catalog['datasets'])
        self.assertIn('BMLhandball', catalog['datasets'])
        self.assertEqual(
            sum(item['motions'] for item in catalog['datasets'].values()),
            catalog['count'])

    def test_each_dataset_is_listable_and_resolvable(self):
        for dataset in ('ACCAD', 'BMLhandball'):
            page = list_motions(dataset=dataset, limit=3)
            self.assertGreater(page['total'], 0)
            self.assertEqual(len(page['motions']), 3)
            for item in page['motions']:
                self.assertEqual(item['dataset'], dataset)
                resolved = resolve_motion(motion_id=item['motion_id'])
                self.assertEqual(resolved['path'], item['path'])

    def test_deepseek_tools_are_registered(self):
        toolkit = RobotToolkit(None)
        names = toolkit.names()
        self.assertIn('robot_amass_datasets', names)
        self.assertIn('robot_amass_list', names)
        self.assertIn('robot_amass_train', names)
        payload = json.loads(toolkit.dispatch('robot_amass_datasets', {}))
        self.assertEqual(payload['count'], 447)
        missing = toolkit.dispatch('robot_amass_train', {
            'motion_id': load_catalog(AMASS_MOTIONS)['motions'][0]['motion_id'],
            'render': False,
        })
        self.assertIn('必须向用户询问', missing)

    def test_reference_job_and_amass_dispatch(self):
        catalog = load_catalog(AMASS_MOTIONS)
        motion = catalog['motions'][0]
        with tempfile.TemporaryDirectory() as tmp, \
                patch('fullbody.jobs.ROOT', Path(tmp)), \
                patch('fullbody.jobs.DATA_RUNS', Path(tmp) / 'runs'), \
                patch('fullbody.jobs.subprocess.Popen') as spawn:
            process = Mock()
            process.poll.return_value = None
            spawn.return_value = process
            toolkit = RobotToolkit(None)
            result = json.loads(toolkit.dispatch('robot_amass_train', {
                'motion_id': motion['motion_id'],
                'iterations': 1,
                'num_envs': 1,
                'device': 'cpu',
                'render': False,
                'coordination': False,
                'rl_pd_influence': 0.7,
                'hand_rl_pd_range': [0.2, 0.6],
                'foot_rl_pd_range': [0.5, 0.9],
            }))
            folder = Path(result['output_dir'])
            config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
            self.assertEqual(config['mode'], 'reference')
            self.assertEqual(config['dataset'], motion['dataset'])
            self.assertEqual(config['source_motion']['motion_id'], motion['motion_id'])
            self.assertEqual(config['rl_pd_influence'], 0.7)
            self.assertEqual(config['hand_rl_pd_range'], [0.2, 0.6])
            self.assertEqual(config['foot_rl_pd_range'], [0.5, 0.9])
            self.assertTrue((folder / 'reference.npz').is_file())
            self.assertEqual(spawn.call_count, 1)

    def test_reference_job_validation(self):
        jobs = TrainingJobs()
        with tempfile.TemporaryDirectory() as tmp:
            reference = Path(tmp) / 'reference.npz'
            np.savez_compressed(reference, times=np.array([0., .1]))
            with patch('fullbody.jobs.ROOT', Path(tmp)), \
                    patch('fullbody.jobs.DATA_RUNS', Path(tmp) / 'runs'), \
                    patch('fullbody.jobs.subprocess.Popen') as spawn:
                process = Mock()
                process.poll.return_value = None
                spawn.return_value = process
                started = jobs.start_reference(reference, iterations=1,
                                               render=False, num_envs=1,
                                               rl_pd_influence=0.6,
                                               hand_rl_pd_range=[0.1, 0.4],
                                               foot_rl_pd_range=[0.5, 0.9])
                self.assertTrue(started['worker_running'])
                self.assertTrue((Path(started['output_dir']) /
                                 'reference.npz').is_file())
                config = json.loads((Path(started['output_dir']) /
                                     'config.json').read_text(encoding='utf-8'))
                self.assertEqual(config['rl_pd_influence'], 0.6)


if __name__ == '__main__':
    unittest.main()
