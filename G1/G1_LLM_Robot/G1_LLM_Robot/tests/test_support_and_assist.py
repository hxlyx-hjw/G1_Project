import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np
import torch

from fullbody.env import ImitationEnv
from fullbody.balanced import heading_command
from fullbody.model import build_model, nominal
from fullbody.motion import MotionConfig, amplitude, arm_amplitude
from fullbody.ppo import ActorCritic
from fullbody.support import convex_hull, signed_distance, support_margin


class SupportAndAssistTest(unittest.TestCase):
    def test_heading_command_tracks_global_yaw_and_keeps_in_place(self):
        angle=np.pi/4
        reference=[np.cos(angle/2),0,0,np.sin(angle/2)]
        self.assertGreater(heading_command(reference,[1,0,0,0],True),.4)
        self.assertAlmostEqual(heading_command([1,0,0,0],[1,0,0,0],False),0.)
        turn=.5
        current=[np.cos(turn/2),0,0,np.sin(turn/2)]
        self.assertAlmostEqual(heading_command([1,0,0,0],current,False),-.3,places=6)

    def test_convex_hull_signed_margin(self):
        hull = convex_hull([[0, 0], [1, 0], [1, 1], [0, 1], [.5, .5]])
        self.assertEqual(len(hull), 4)
        self.assertAlmostEqual(signed_distance([.5, .5], hull), .5)
        self.assertAlmostEqual(signed_distance([2., .5], hull), -1.)

    def test_real_foot_support_hull_contains_com(self):
        with tempfile.TemporaryDirectory() as directory:
            model = build_model(Path(directory))
            data = mujoco.MjData(model)
            data.qpos[:] = nominal(model)
            mujoco.mj_forward(model, data)
            feet = [model.body(name).id for name in ('left_ankle_roll_link', 'right_ankle_roll_link')]
            com = data.subtree_com[model.body('pelvis').id][:2]
            margin, hull = support_margin(model, data, feet, com, .12, data.qvel[:2], .08)
            self.assertGreaterEqual(len(hull), 3)
            self.assertGreater(margin, 0.)

    def test_joint_assist_is_zero_and_legacy_checkpoint_loads(self):
        torch.manual_seed(3)
        policy = ActorCritic()
        x = torch.randn(2, 188)
        self.assertEqual(len(policy.joint_assist.heads), 29)
        self.assertTrue(torch.allclose(policy.joint_assist(x), torch.zeros(2, 29), atol=1e-8))
        self.assertTrue(torch.allclose(policy.mean(x), policy.actor(x), atol=1e-8))
        old = ActorCritic(joint_assist=False)
        legacy = {k: v for k, v in old.state_dict().items() if not k.startswith('joint_assist.')}
        restored = ActorCritic.from_checkpoint({'policy': legacy})
        self.assertFalse(restored.joint_assist_enabled)
        self.assertTrue(torch.allclose(restored.mean(x), old.actor(x), atol=1e-7))

    def test_arm_amplitude_prioritizes_stability(self):
        config = MotionConfig()
        stable = arm_amplitude(1., config)
        unstable = arm_amplitude(.2, config)
        self.assertAlmostEqual(stable, amplitude(1., config))
        self.assertLess(unstable, amplitude(.2, config))
        self.assertAlmostEqual(unstable, amplitude(.2, config) * (.35 + .65 * .2))
        self.assertEqual(arm_amplitude(.2, MotionConfig.legacy()), 1.)

    def test_airborne_info_and_arm_priority_reward_scale(self):
        reference = Path('motions/self_test1/reference.npz')
        if not reference.is_file(): self.skipTest('reference missing')
        with tempfile.TemporaryDirectory() as directory:
            env = ImitationEnv(reference, directory)
            env.reset(options={'phase': 0})
            env.data.qpos[2] += .6
            env.data.qvel[:] = 0
            mujoco.mj_forward(env.model, env.data)
            _, _, _, _, info = env.step(np.zeros(29))
            self.assertTrue(info['airborne'])
            self.assertEqual(info['foot_contacts'], 0)
            self.assertIsNotNone(info['support_margin_m'])
            self.assertIsInstance(info['com_in_support_hull'], bool)
            self.assertGreaterEqual(info['support_hull_vertices'], 3)
            self.assertAlmostEqual(info['airborne_task_scale'], .55)
            env.arm_tracking_priority = True
            env.arm_stability_floor = .35
            env._last_balance_margin = .2
            _, _, _, _, info = env.step(np.zeros(29))
            self.assertAlmostEqual(info['arm_tracking_scale'], .48)

    def test_previous_motion_checkpoint_keeps_arm_amplitude(self):
        from coordination.video import apply_motion
        class DummyEnv:
            def configure_motion(self, **kwargs): self.kwargs = kwargs
        old = DummyEnv()
        apply_motion(old, {'motion_gain': 1.35, 'motion': {'max_gain': 1.35}})
        self.assertEqual(old.kwargs['arm_stability_floor'], 1.0)
        self.assertFalse(old.kwargs['arm_priority'])
        current = DummyEnv()
        apply_motion(current, {'motion_gain': 1.35, 'motion': {'max_gain': 1.35, 'arm_stability_floor': .35}})
        self.assertEqual(current.kwargs['arm_stability_floor'], .35)


if __name__ == '__main__':
    unittest.main()
