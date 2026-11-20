import unittest,tempfile
from pathlib import Path
import numpy as np
import torch
from coordination.network import CoordinatorNet,Coordinator,Features
from coordination.jump import run
from fullbody.model import build_model,nominal
import mujoco


class CoordinationTest(unittest.TestCase):
    def test_graph_shapes_and_bounds(self):
        net=CoordinatorNet()
        output=net(torch.randn(3,14,6),torch.randn(3,6))
        self.assertEqual(tuple(output.shape),(3,29))
        self.assertLessEqual(float(output.detach().abs().max()),.08)

    def test_real_com_teacher(self):
        with tempfile.TemporaryDirectory() as d:
            m=build_model(d); data=mujoco.MjData(m); data.qpos[:]=nominal(m)
            nodes,central,correction=Features(m).teacher(data)
            self.assertEqual(nodes.shape,(14,6)); self.assertEqual(central.shape,(6,))
            self.assertTrue(np.isfinite(correction).all())
            self.assertLessEqual(float(np.max(np.abs(correction))),.080001)

    def test_small_jump_physics(self):
        path=Path('runs/coordination/com_graph.pt')
        if not path.is_file(): self.skipTest('trained network not present')
        with tempfile.TemporaryDirectory() as d:
            report=run(d,coordinator=path)
            self.assertTrue(report['success'],report)
            self.assertFalse(report['fallen'])
            self.assertGreaterEqual(report['airborne_seconds'],.04)
            self.assertGreaterEqual(report['simulation_seconds']-report['landing_time'],1.5)


if __name__=='__main__': unittest.main()
