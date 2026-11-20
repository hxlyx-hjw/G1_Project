"""统一轴测相机与训练窗口尺寸的回归测试。"""
from pathlib import Path as _ProjectPath
import sys as _project_sys
import os as _project_os

_project_root = _ProjectPath(__file__).resolve().parents[1]
_project_sys.path.insert(0, str(_project_root))
_project_os.chdir(_project_root)

import unittest
import mujoco
import numpy as np
from viewer_camera import (
    DEFAULT_LOOKAT,
    ISOMETRIC_AZIMUTH,
    ISOMETRIC_DISTANCE,
    ISOMETRIC_ELEVATION,
    apply_isometric_camera,
    isometric_camera,
    training_window_size,
)


class ViewerCameraTest(unittest.TestCase):
    def _model(self):
        xml = """<mujoco><worldbody>
          <body name="pelvis"><freejoint/><geom type="sphere" size="0.1"/></body>
        </worldbody></mujoco>"""
        return mujoco.MjModel.from_xml_string(xml)

    def test_isometric_camera_faces_robot_from_front_three_quarter(self):
        model = self._model()
        pelvis_id = model.body("pelvis").id
        cam = isometric_camera(model, tracking_body=pelvis_id)
        self.assertAlmostEqual(cam.azimuth, ISOMETRIC_AZIMUTH)
        self.assertAlmostEqual(cam.elevation, ISOMETRIC_ELEVATION)
        self.assertAlmostEqual(cam.distance, ISOMETRIC_DISTANCE)
        np.testing.assert_allclose(cam.lookat, DEFAULT_LOOKAT)
        self.assertEqual(cam.type, mujoco.mjtCamera.mjCAMERA_TRACKING)
        self.assertEqual(cam.trackbodyid, pelvis_id)

    def test_free_camera_keeps_isometric_orientation(self):
        cam = isometric_camera(tracking_body=None)
        self.assertEqual(cam.type, mujoco.mjtCamera.mjCAMERA_FREE)
        self.assertEqual(cam.trackbodyid, -1)
        self.assertAlmostEqual(cam.azimuth, ISOMETRIC_AZIMUTH)

    def test_camera_position_is_on_robot_front_side(self):
        model = self._model()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        cam = isometric_camera(model)
        scene = mujoco.MjvScene(model, maxgeom=10)
        mujoco.mjv_updateScene(
            model, data, mujoco.MjvOption(), mujoco.MjvPerturb(), cam,
            mujoco.mjtCatBit.mjCAT_ALL, scene)
        self.assertGreater(float(scene.camera[0].pos[0]), 0.0)

    def test_apply_updates_existing_camera(self):
        cam = mujoco.MjvCamera()
        apply_isometric_camera(
            cam, distance=4.2, lookat=(1.0, 2.0, 0.7), tracking_body=3)
        self.assertAlmostEqual(cam.distance, 4.2)
        np.testing.assert_allclose(cam.lookat, (1.0, 2.0, 0.7))
        self.assertEqual(cam.type, mujoco.mjtCamera.mjCAMERA_TRACKING)
        self.assertEqual(cam.trackbodyid, 3)

    def test_training_window_uses_most_of_the_screen(self):
        self.assertEqual(training_window_size(1920, 1080), (1804, 972))
        self.assertEqual(training_window_size(1366, 768), (1284, 678))
        self.assertEqual(training_window_size(2560, 1440), (1920, 1080))
        width, height = training_window_size(1024, 768)
        self.assertLessEqual(width, 1024)
        self.assertLessEqual(height, 768)


if __name__ == "__main__":
    unittest.main()
