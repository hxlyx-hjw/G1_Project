"""视频全局轨迹、时序稳定和接触参考的回归测试。"""
from pathlib import Path as _ProjectPath
import sys as _project_sys
import os as _project_os

_project_root = _ProjectPath(__file__).resolve().parents[1]
_project_sys.path.insert(0, str(_project_root))
_project_os.chdir(_project_root)

import tempfile
import unittest
from pathlib import Path
import numpy as np
from fullbody.global_track import estimate_root_trajectory, yaw_quaternion
from fullbody.reference import Reference, convert
from fullbody.refine import refine
from fullbody.model import build_model
from video_motion.pose import stabilize


def skeleton(forward_shift=0.0, foot_lift=0.0):
    points = np.zeros((33, 3))
    points[11], points[12] = [-.2, -.5, 0], [.2, -.5, 0]
    points[23], points[24] = [-.15, 0, 0], [.15, 0, 0]
    for shoulder, elbow, wrist in ((11, 13, 15), (12, 14, 16)):
        points[elbow] = points[shoulder] + [0, .25, 0]
        points[wrist] = points[elbow] + [0, .20, .10]
    for hip, knee, foot in ((23, 25, 27), (24, 26, 28)):
        points[knee] = points[hip] + [0, .45, 0]
        points[foot] = points[knee] + [0, .45, 0]
    # 人体世界里脚向后滑，等价于根节点向前移动。
    points[27, 2] -= forward_shift
    points[28, 2] -= forward_shift
    points[27, 1] -= foot_lift
    return points


class GlobalTrackTest(unittest.TestCase):
    def test_stabilize_fills_single_frame_dropout(self):
        times = np.arange(10) / 15.
        points = np.zeros((10, 33, 3))
        points[:, 11, 0] = np.sin(np.arange(10) / 3.)
        visibility = np.ones((10, 33))
        points[4, 11] = [100., 100., 100.]
        visibility[4, 11] = 0.
        filtered, filtered_visibility = stabilize(times, points, visibility, window=5)
        self.assertEqual(filtered.shape, (10, 33, 3))
        self.assertLess(abs(filtered[4, 11, 0]), 10.)
        np.testing.assert_array_equal(filtered_visibility, visibility)

    def test_contact_track_recovers_forward_motion(self):
        frames = 10
        points = np.array([skeleton(.16 * i / (frames - 1)) for i in range(frames)])
        result = estimate_root_trajectory(
            points, np.ones((frames, 33)), np.arange(frames) / 15.)
        self.assertGreater(result['root'][-1, 0], .05)
        self.assertAlmostEqual(result['root'][-1, 1], 0., places=6)
        self.assertAlmostEqual(result['yaw'][0], 0., places=6)
        self.assertTrue(result['contacts'].any())
        quat = yaw_quaternion(result['yaw'])
        np.testing.assert_allclose(np.linalg.norm(quat, axis=1), 1.)

    def test_contact_reference_round_trip(self):
        frames = 7
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            points = np.array([skeleton(.12 * i / (frames - 1)) for i in range(frames)])
            times = np.arange(frames) / 15.
            np.savez(root / 'pose.npz', times=times, points=points,
                     visibility=np.ones((frames, 33)))
            report = convert(root / 'pose.npz', root / 'raw.npz',
                             root_mode='contact_grounded')
            self.assertEqual(report['root_mode'], 'contact_grounded')
            self.assertGreater(report['root_displacement_m'], .03)
            refine(root / 'pose.npz', root / 'raw.npz', root / 'reference.npz')
            reference = Reference(root / 'reference.npz', build_model(root / 'model'))
            self.assertTrue(reference.root_mode.startswith('contact_grounded'))
            self.assertEqual(reference.contacts.shape, (frames, 2))
            self.assertIsNotNone(reference.contact_at(.1))


if __name__ == '__main__':
    unittest.main()
