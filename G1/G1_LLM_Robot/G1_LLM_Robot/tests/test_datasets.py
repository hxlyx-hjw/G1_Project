import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fullbody import datasets


class DatasetRegistryTest(unittest.TestCase):
    def test_inventory_clips_and_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'AISTPP' / 'videos').mkdir(parents=True)
            (root / 'AISTPP' / 'calibrations').mkdir()
            (root / 'AISTPP' / 'manifests').mkdir()
            (root / 'AISTPP' / 'videos' / 'a.mp4').write_bytes(b'a' * 2000)
            (root / 'AISTPP' / 'videos' / 'b.mp4').write_bytes(b'b' * 2000)
            (root / 'AISTPP' / 'calibrations' / 'a.json').write_text(
                '{}', encoding='utf-8')
            (root / 'AISTPP' / 'manifests' / 'pairs.json').write_text(
                json.dumps([{
                    'key': 'seq', 'pair_key': 'a',
                    'a': 'videos/a.mp4', 'b': 'videos/b.mp4',
                    'calibration': 'calibrations/a.json',
                }]), encoding='utf-8')
            (root / 'MotionX' / 'videos').mkdir(parents=True)
            (root / 'MotionX' / 'videos' / 'm.mp4').write_bytes(b'm' * 2000)
            (root / 'LAFAN1' / 'bvh').mkdir(parents=True)
            (root / 'LAFAN1' / 'bvh' / 'x.bvh').write_text('x', encoding='utf-8')
            (root / '100STYLE' / 'data').mkdir(parents=True)
            (root / '100STYLE' / 'data' / 'y.bvh').write_text('y', encoding='utf-8')
            (root / 'AMASS' / 'motions').mkdir(parents=True)
            (root / 'AMASS' / 'motions' / 'z.npz').write_bytes(b'z')

            with patch.object(datasets, 'DATASET_ROOT', root), \
                    patch.object(datasets, 'AMASS_MOTIONS',
                                 root / 'AMASS' / 'motions'):
                report = datasets.inventory()
                by_name = {item['name']: item for item in report['datasets']}
                self.assertEqual(by_name['AISTPP']['dual_view_pairs'], 1)
                self.assertEqual(by_name['MotionX']['videos'], 1)
                self.assertEqual(by_name['LAFAN1']['bvh_files'], 1)
                self.assertEqual(by_name['100STYLE']['bvh_files'], 1)
                self.assertEqual(by_name['AMASS']['references'], 1)
                self.assertEqual(len(datasets.clips('AISTPP', limit=10)), 1)
                catalog = datasets.build_manifests()
                self.assertTrue((root / '00_INDEX' / 'catalog.json').is_file())
                self.assertTrue((root / 'MotionX' / 'manifests'
                                 / 'video_list.txt').is_file())
                dry = datasets.start_training(
                    'AISTPP', 0, iterations=1, envs=1, dry_run=True)
                self.assertTrue(dry['dry_run'])
                self.assertEqual(dry['clip']['pair_key'], 'a')


if __name__ == '__main__':
    unittest.main()
