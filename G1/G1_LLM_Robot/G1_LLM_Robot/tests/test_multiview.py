import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from video_motion.multiview import (camera_pair, dlt_point, normalize_scale,
                                    synchronize, triangulate, extract,
                                    discover_pairs, repair_tracks,
                                    estimate_sync_offset)
from fullbody.jobs import TrainingJobs


class MultiViewTest(unittest.TestCase):
    def test_dlt_recovers_synthetic_3d_points(self):
        info_a = {'width': 1280, 'height': 720}
        info_b = {'width': 1280, 'height': 720}
        cam_a, cam_b, _ = camera_pair(info_a, info_b, fov=60, baseline=2.0,
                                      azimuth_b=90)
        rng = np.random.default_rng(4)
        world = rng.normal(0, .12, (33, 3))
        world[:, 2] += 3.0
        pixel_a = np.stack([
            (cam_a['P'] @ np.r_[p, 1.0])[:2]
            / (cam_a['P'] @ np.r_[p, 1.0])[2] for p in world])
        pixel_b = np.stack([
            (cam_b['P'] @ np.r_[p, 1.0])[:2]
            / (cam_b['P'] @ np.r_[p, 1.0])[2] for p in world])
        norm_a = pixel_a / np.array([info_a['width'], info_a['height']])
        norm_b = pixel_b / np.array([info_b['width'], info_b['height']])
        vis = np.ones((1, 33))
        points, visibility, errors = triangulate(
            norm_a[None], norm_b[None], vis, vis, cam_a, cam_b,
            info_a, info_b, min_visibility=.1, max_reprojection=5)
        self.assertEqual(points.shape, (1, 33, 3))
        self.assertEqual(visibility.sum(), 33)
        self.assertLess(float(np.nanmedian(errors)), 1e-6)
        self.assertLess(float(np.max(np.linalg.norm(points[0] - world, axis=1))), 1e-5)

    def test_synchronize_and_scale(self):
        times = np.arange(0, 2, .1)
        points = np.zeros((len(times), 33, 3))
        points[:, :, 2] = 1
        vis = np.ones((len(times), 33))
        out_times, a, av, b, bv = synchronize(
            times, points, vis, times + .2, points + 1, vis,
            fps=10, offset_a=0, offset_b=0)
        self.assertGreater(len(out_times), 5)
        self.assertTrue(np.allclose(out_times[0], 0))
        self.assertEqual(a.shape, b.shape)
        # Synthetic leg lengths are scaled to the requested target.
        p = points.copy()
        p[:, 23] = [0, 0, 1]
        p[:, 25] = [0, 0, .5]
        p[:, 27] = [0, 0, 0]
        p[:, 24] = [0, 0, 1]
        p[:, 26] = [0, 0, .5]
        p[:, 28] = [0, 0, 0]
        scaled, info = normalize_scale(p, vis, target_leg_length=.8)
        self.assertGreater(info['scale_factor'], 0)
        self.assertGreater(np.linalg.norm(scaled[0, 23] - scaled[0, 27]), .5)

    def test_dlt_rejects_behind_camera(self):
        info = {'width': 640, 'height': 480}
        cam_a, cam_b, _ = camera_pair(info, info)
        with self.assertRaises(ValueError):
            dlt_point(cam_a['P'], cam_b['P'], np.array([320, 240.]),
                      np.array([320, 240.]))

    def test_repair_tracks_fills_only_bounded_gaps(self):
        count = 10
        points = np.zeros((count, 33, 3))
        visibility = np.zeros((count, 33))
        errors = np.full((count, 33), np.nan)
        points[0, 0, 0] = 0.0
        points[4, 0, 0] = 4.0
        points[9, 0, 0] = 9.0
        visibility[[0, 4, 9], 0] = 1.0
        errors[[0, 4, 9], 0] = 1.0
        points[1, 1, 0] = 10.0
        points[9, 1, 0] = 20.0
        visibility[[1, 9], 1] = 1.0
        errors[[1, 9], 1] = 1.0
        repaired, repaired_vis, corrected, report = repair_tracks(
            points, visibility, errors, min_visibility=.4,
            max_gap_frames=4)
        self.assertAlmostEqual(float(repaired[2, 0, 0]), 2.0, places=5)
        self.assertAlmostEqual(float(repaired[5, 0, 0]), 5.0, places=5)
        self.assertGreaterEqual(float(repaired_vis[2, 0]), .5)
        self.assertTrue(corrected[2, 0])
        self.assertFalse(corrected[5, 1])
        self.assertTrue(np.isnan(errors[2, 0]))
        self.assertGreater(report['corrected_joint_cells'], 0)

    def test_short_view_limits_scan_and_common_timeline(self):
        info_a = {'width': 1280, 'height': 720, 'fps': 30.0,
                  'frames': 300, 'duration': 10.0}
        info_b = {'width': 1280, 'height': 720, 'fps': 30.0,
                  'frames': 90, 'duration': 3.0}
        times_a = np.arange(0, 10, 1 / 30)
        times_b = np.arange(0, 3, 1 / 30)
        cam_a, cam_b, _ = camera_pair(
            info_a, info_b, fov=60, baseline=2.0, azimuth_b=90)
        body = np.zeros((33, 3))
        body[:, 2] = 3.0
        body[23] = [0., .12, 2.0]
        body[25] = [0., .12, 1.5]
        body[27] = [0., .12, 1.0]
        body[24] = [0., -.12, 2.0]
        body[26] = [0., -.12, 1.5]
        body[28] = [0., -.12, 1.0]
        pixel_a = np.stack([
            (cam_a['P'] @ np.r_[p, 1.0])[:2]
            / (cam_a['P'] @ np.r_[p, 1.0])[2] for p in body])
        pixel_b = np.stack([
            (cam_b['P'] @ np.r_[p, 1.0])[:2]
            / (cam_b['P'] @ np.r_[p, 1.0])[2] for p in body])
        norm_a = np.repeat(
            (pixel_a / [info_a['width'], info_a['height']])[None],
            len(times_a), axis=0)
        norm_b = np.repeat(
            (pixel_b / [info_b['width'], info_b['height']])[None],
            len(times_b), axis=0)
        visibility = np.ones((len(times_a), 33))
        calls = []

        def fake_detect(video, **kwargs):
            calls.append(kwargs['max_seconds'])
            if len(calls) == 1:
                return times_a, norm_a, visibility, 30.0
            return times_b, norm_b, visibility[:len(times_b)], 30.0

        with tempfile.TemporaryDirectory() as tmp, \
                patch('video_motion.multiview.video_info',
                      side_effect=[info_a, info_b]), \
                patch('video_motion.multiview.detect_2d',
                      side_effect=fake_detect):
            report = extract('a.mp4', 'b.mp4', Path(tmp) / 'pose.npz',
                             fps=30, max_seconds=60, backend='dlt')
        self.assertTrue(all(value <= 3.01 for value in calls))
        self.assertEqual(report['timeline']['shorter_source'], 'b')
        self.assertLessEqual(report['timeline']['common_duration_seconds'], 3.0)
        self.assertLessEqual(report['scan_seconds'], 3.01)

    def test_scale_falls_back_when_legs_are_unavailable(self):
        points = np.zeros((2, 33, 3))
        visibility = np.zeros((2, 33))
        points[:, 11] = [0., .2, 2.5]
        points[:, 12] = [0., -.2, 2.5]
        points[:, 23] = [0., .15, 2.0]
        points[:, 24] = [0., -.15, 2.0]
        visibility[:, [11, 12, 23, 24]] = 1.0
        _, info = normalize_scale(points, visibility, target_leg_length=.85)
        self.assertEqual(info['method'], 'torso_ratio_1.55')
        self.assertGreater(info['scale_factor'], 0)

        empty = normalize_scale(
            points, np.zeros_like(visibility), target_leg_length=.85)
        self.assertEqual(empty[1]['method'], 'unscaled')
        self.assertEqual(empty[1]['scale_factor'], 1.0)
        self.assertIn('warning', empty[1])

    def test_auto_sync_recovers_known_offset(self):
        info = {'width': 1280, 'height': 720}
        cam_a, cam_b, _ = camera_pair(
            info, info, fov=60, baseline=2.0, azimuth_b=90)
        times = np.arange(0, 4, 1 / 20)
        body = np.zeros((33, 3))
        body[:, 2] = 3.0
        body[11] = [0., .2, 2.6]
        body[12] = [0., -.2, 2.6]
        body[13] = [0., .4, 2.4]
        body[14] = [0., -.4, 2.4]
        body[23] = [0., .15, 2.0]
        body[24] = [0., -.15, 2.0]
        body[25] = [0., .15, 1.5]
        body[26] = [0., -.15, 1.5]
        body[27] = [0., .15, 1.0]
        body[28] = [0., -.15, 1.0]
        points = np.repeat(body[None], len(times), axis=0)
        pulse = np.zeros(len(times))
        pulse[20:41] = .4
        points[:, 15, 0] = points[:, 16, 0] = pulse
        pixel_a = np.stack([
            np.stack([(cam_a['P'] @ np.r_[p, 1.0])[:2]
                      / (cam_a['P'] @ np.r_[p, 1.0])[2] for p in frame])
            for frame in points])
        pixel_b = np.stack([
            np.stack([(cam_b['P'] @ np.r_[p, 1.0])[:2]
                      / (cam_b['P'] @ np.r_[p, 1.0])[2] for p in frame])
            for frame in points])
        norm_a = pixel_a / [info['width'], info['height']]
        norm_b = pixel_b / [info['width'], info['height']]
        vis = np.ones((len(times), 33))
        vis_b = vis.copy()
        vis_b[times <= 2.0] = 0.0
        result = estimate_sync_offset(
            times, norm_a, vis, times + 2.0, norm_b, vis_b,
            cam_a, cam_b, info, info, max_offset_seconds=3.0,
            step_seconds=.5, sample_hz=5.0, max_reprojection=5.0)
        self.assertTrue(result['applied'])
        self.assertAlmostEqual(result['offset_b'], -2.0, delta=.2)

    def test_auto_sync_keeps_good_baseline(self):
        info = {'width': 1280, 'height': 720}
        cam_a, cam_b, _ = camera_pair(
            info, info, fov=60, baseline=2.0, azimuth_b=90)
        times = np.arange(0, 4, 1 / 20)
        body = np.zeros((33, 3))
        body[:, 2] = 3.0
        body[11] = [0., .2, 2.6]
        body[12] = [0., -.2, 2.6]
        body[23] = [0., .15, 2.0]
        body[24] = [0., -.15, 2.0]
        points = np.repeat(body[None], len(times), axis=0)
        pixel_a = np.stack([
            np.stack([(cam_a['P'] @ np.r_[p, 1.0])[:2]
                      / (cam_a['P'] @ np.r_[p, 1.0])[2] for p in frame])
            for frame in points])
        pixel_b = np.stack([
            np.stack([(cam_b['P'] @ np.r_[p, 1.0])[:2]
                      / (cam_b['P'] @ np.r_[p, 1.0])[2] for p in frame])
            for frame in points])
        vis = np.ones((len(times), 33))
        result = estimate_sync_offset(
            times, pixel_a / [info['width'], info['height']], vis,
            times, pixel_b / [info['width'], info['height']], vis,
            cam_a, cam_b, info, info, max_offset_seconds=3.0,
            step_seconds=.5, sample_hz=5.0, max_reprojection=5.0)
        self.assertFalse(result['applied'])
        self.assertEqual(result['offset_b'], 0.0)

    def test_end_to_end_dlt_extract_with_synthetic_views(self):
        info = {'width': 1280, 'height': 720, 'fps': 30.0,
                'frames': 10, 'duration': 1.0}
        cam_a, cam_b, _ = camera_pair(info, info, fov=60, baseline=2.0,
                                      azimuth_b=90)
        body = np.zeros((33, 3))
        body[:, 2] = 3.0
        body[:, 0] = np.linspace(-.25, .25, 33)
        body[:, 1] = np.linspace(.25, -.25, 33)
        # Make a plausible leg chain so metric normalization has a target.
        body[23] = [0., .12, 2.0]
        body[25] = [0., .12, 1.5]
        body[27] = [0., .12, 1.0]
        body[24] = [0., -.12, 2.0]
        body[26] = [0., -.12, 1.5]
        body[28] = [0., -.12, 1.0]
        pixel_a = np.stack([
            (cam_a['P'] @ np.r_[p, 1.0])[:2] /
            (cam_a['P'] @ np.r_[p, 1.0])[2] for p in body])
        pixel_b = np.stack([
            (cam_b['P'] @ np.r_[p, 1.0])[:2] /
            (cam_b['P'] @ np.r_[p, 1.0])[2] for p in body])
        norm_a = np.repeat((pixel_a / [info['width'], info['height']])[None],
                           5, axis=0)
        norm_b = np.repeat((pixel_b / [info['width'], info['height']])[None],
                           5, axis=0)
        times = np.arange(5) / 30.0
        vis = np.ones((5, 33))
        calls = {'count': 0}

        def fake_detect(video, **kwargs):
            index = calls['count']
            calls['count'] += 1
            return times, (norm_a if index == 0 else norm_b), vis, 30.0

        with tempfile.TemporaryDirectory() as tmp, \
                patch('video_motion.multiview.video_info', return_value=info), \
                patch('video_motion.multiview.detect_2d', side_effect=fake_detect):
            output = Path(tmp) / 'pose.npz'
            report = extract('a.mp4', 'b.mp4', output, fps=30,
                             max_seconds=1, backend='dlt',
                             target_leg_length=.85)
            self.assertEqual(report['frames'], 5)
            self.assertEqual(report['backend'], 'multiview_dlt')
            with np.load(output) as data:
                self.assertEqual(data['points'].shape, (5, 33, 3))
                self.assertEqual(data['visibility'].shape, (5, 33))
                self.assertEqual(len(data['times']), 5)
                leg = np.linalg.norm(data['points'][0, 23] - data['points'][0, 25]) + \
                      np.linalg.norm(data['points'][0, 25] - data['points'][0, 27])
                self.assertAlmostEqual(float(leg), .85, places=4)

    def test_folder_list_pairs_and_single_a_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'a').mkdir()
            (root / 'b').mkdir()
            for name in ('a/001.mp4', 'b/001.mp4', 'a/002.mp4', 'b/002.mp4'):
                (root / name).write_bytes(b'')
            (root / 'list.txt').write_text(
                'a/001.mp4,b/001.mp4\na/002.mp4\n', encoding='utf-8')
            manifest = discover_pairs(root)
            self.assertEqual(len(manifest['pairs']), 2)
            first = manifest['pairs'][0]
            self.assertEqual(Path(first['a']).parts[-2:], ('a', '001.mp4'))
            self.assertEqual(Path(first['b']).parts[-2:], ('b', '001.mp4'))

    def test_deepseek_multiview_job_config(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch('fullbody.jobs.DATA_RUNS', Path(tmp) / 'runs'), \
                patch('fullbody.jobs.subprocess.Popen') as spawn:
            root = Path(tmp) / 'videos'
            root.mkdir()
            (root / 'a.mp4').write_bytes(b'')
            (root / 'b.mp4').write_bytes(b'')
            (root / 'list.txt').write_text('a.mp4,b.mp4\n', encoding='utf-8')
            process = Mock()
            process.poll.return_value = None
            spawn.return_value = process
            result = TrainingJobs().start_multiview(
                root, iterations=1, num_envs=1, device='cpu', render=False,
                rl_pd_influence=.7, hand_rl_pd_range=[.2, .6],
                foot_rl_pd_range=[.5, .9])
            folder = Path(result['output_dir'])
            config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
            self.assertEqual(config['mode'], 'multiview')
            self.assertTrue(config['fallback'])
            self.assertTrue(config['self_correct'])
            self.assertAlmostEqual(config['repair_gap_seconds'], .25)
            self.assertTrue(config['auto_sync'])
            self.assertAlmostEqual(config['min_valid_joint_fraction'], .05)
            self.assertTrue(config['video_a'].endswith('a.mp4'))
            self.assertTrue(config['video_b'].endswith('b.mp4'))


if __name__ == '__main__':
    unittest.main()
