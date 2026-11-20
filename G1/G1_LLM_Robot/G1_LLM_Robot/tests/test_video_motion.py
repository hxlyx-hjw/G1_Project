"""回归验证：骨架方向、遮挡保持、非法文件、真实MuJoCo回放。"""

# 支持从项目根目录、IDE及脚本所在目录启动。
from pathlib import Path as _ProjectPath
import sys as _project_sys
import os as _project_os
_project_root = _ProjectPath(__file__).resolve().parents[1]
_project_sys.path.insert(0, str(_project_root))
_project_os.chdir(_project_root)

from pathlib import Path
import tempfile
import unittest
import numpy as np
from video_motion.pose import load
from video_motion.retarget import Retargeter, arm_directions, play


def skeleton():
    p = np.zeros((33,3))
    p[11],p[12] = [-.2,-.5,0],[.2,-.5,0]
    p[23],p[24] = [-.15,0,0],[.15,0,0]
    for s,e,w in ((11,13,15),(12,14,16)):
        p[e] = p[s]+[0,.25,0]
        # 面向相机的人体：解剖学左侧为相机-X，上为-Y，前为+Z。
        p[w] = p[e]+[0,.20,.10]
    return p


class VideoMotionTest(unittest.TestCase):
    def test_body_coordinates(self):
        p = skeleton()
        upper,lower = arm_directions(p,'left')
        self.assertLess(upper[2],-.99)
        self.assertGreater(lower[0],0)
        np.testing.assert_allclose(arm_directions(p+10,'left'),(upper,lower),atol=1e-12)
        # 任意相机旋转不能改变人体局部方向。
        rotation = np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
        np.testing.assert_allclose(arm_directions(p @ rotation.T,'left'),(upper,lower),atol=1e-12)
        left = (p[11]-p[12])/np.linalg.norm(p[11]-p[12])
        up = np.array([0.,-1.,0.])
        self.assertAlmostEqual(np.linalg.det(np.column_stack((np.cross(left,up),left,up))),1.)

    def test_forward_ik_matches_robot_forward(self):
        import mujoco
        from g1_walkfree.controller import WalkController
        c = WalkController(verbose=False)
        converter = Retargeter(c)
        p = skeleton()
        # 双臂向人体前方伸出，稍屈肘以避免直臂奇异。
        for s,e,w in ((11,13,15),(12,14,16)):
            p[e] = p[s] + [0,.04,.25]
            p[w] = p[e] + [0,.10,.20]
        q = converter.solve(p,np.ones(33))
        converter.data.qpos[converter.addresses] = q
        mujoco.mj_forward(c.model,converter.data)
        rotation = converter.data.xmat[converter.torso].reshape(3,3)
        for i,side in ((0,'left'),(4,'right')):
            shoulder = converter.data.xpos[c.model.body(side+'_shoulder_pitch_link').id]
            wrist = converter.data.xpos[c.model.body(side+'_wrist_yaw_link').id]
            relative = rotation.T @ (wrist-shoulder)
            self.assertGreater(relative[0],.20,side+' wrist must point along robot +X')
            self.assertLess(q[i],0,side+' shoulder_pitch forward must be negative')
        # 正向行走命令与上述 +X 指向一致；实际动力学由回放用例验证。
        c.set_command(.2,0,0)
        self.assertGreater(c.cmd[0],0)

    def test_validation(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'bad.npz'
            np.savez(path,times=[0,0],points=np.zeros((2,33,3)),visibility=np.ones((2,33)))
            with self.assertRaises(ValueError):
                load(path)

    def test_simulation(self):
        from robot_bridge import RobotSession
        from robot_tools import RobotToolkit
        with tempfile.TemporaryDirectory() as d, RobotSession(verbose=False) as bot:
            converter = Retargeter(bot.ctrl)
            previous = converter.previous.copy()
            np.testing.assert_equal(converter.solve(skeleton(),np.zeros(33)),previous)
            self.assertTrue(np.isfinite(converter.solve(skeleton(),np.ones(33))).all())
            path = Path(d)/'motion.npz'
            np.savez(path,times=[0,.2,.4],points=np.array([skeleton()]*3),visibility=np.ones((3,33)))
            result = play(bot,path)
            print('Simulation report:',result)
            self.assertTrue(result['completed'])
            self.assertGreaterEqual(result['simulation_seconds'],1.)
            self.assertLess(result['max_drift_m'],.25)
            self.assertEqual(set(t['function']['name'] for t in RobotToolkit(bot).schemas()),RobotToolkit(bot).names())
            frozen = bot.sim_time()
            import time
            time.sleep(.05)
            self.assertEqual(bot.sim_time(),frozen)
            self.assertEqual(tuple(bot.ctrl.cmd),(0.,0.,0.))


class SmplXConfigTest(unittest.TestCase):
    def test_gender_fallback(self):
        from video_motion import smplx_fit
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'SMPLX_MALE.npz').write_bytes(b'placeholder')
            self.assertEqual(smplx_fit.resolve_gender(root,'neutral'),'male')
            self.assertEqual(smplx_fit.resolve_gender(root,'male'),'male')
            info=smplx_fit.describe(root)
            self.assertEqual(info['default_gender'],'male')
            self.assertTrue(str(info['model_file']).endswith('SMPLX_MALE.npz'))
            with self.assertRaises(ValueError):
                smplx_fit.resolve_gender(root,'unknown')

    def test_model_discovery_and_mapping(self):
        from video_motion import smplx_fit
        if not smplx_fit.available():
            self.skipTest('SMPL-X 包或 D:\\SMPL 权重不可用')
        info=smplx_fit.describe()
        self.assertTrue(info['available'])
        self.assertTrue(info['default_gender'])
        self.assertTrue(Path(info['model_file']).is_file())
        self.assertIsNotNone(smplx_fit.model_dir(r'D:\SMPL'))

        count=3
        joints=np.zeros((count,127,3),np.float32)
        fit={'joints':joints,'scale':1.0,'basis':np.eye(3),'pelvis_ref':np.zeros(3),
             'errors':np.full(count,.05,np.float32),'valid':np.ones(count,bool)}
        points,visibility=smplx_fit.map_joints_to_mediapipe(fit,np.ones((count,33)))
        self.assertEqual(points.shape,(count,33,3))
        self.assertEqual(visibility.shape,(count,33))
        self.assertTrue(np.all(visibility[:,list(smplx_fit.BODY_MEDIAPIPE)]>=.5))


class ExtractionProgressTest(unittest.TestCase):
    def test_extract_forwards_detection_progress(self):
        from unittest.mock import patch
        from video_motion.pose import extract
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            video=root/'video.mp4'; video.write_bytes(b'placeholder')
            model=root/'model.task'; model.write_bytes(b'placeholder')
            times=np.array([0.,.1])
            points=np.ones((2,33,3))
            visibility=np.ones((2,33))
            seen=[]
            def fake_detect(*args,progress=None,**kwargs):
                progress(1,2)
                return times,points,visibility,30.
            with patch('video_motion.pose.ensure_model',return_value=model), \
                 patch('video_motion.pose._detect_subprocess',side_effect=fake_detect):
                extract(video,root/'out.npz',fps=10,max_seconds=1,
                        backend='mediapipe',detect_progress=lambda current,total:seen.append((current,total)))
            self.assertEqual(seen,[(1,2)])


if __name__ == '__main__':
    unittest.main()
