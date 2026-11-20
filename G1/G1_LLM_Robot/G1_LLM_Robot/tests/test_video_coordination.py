import tempfile
import unittest
from pathlib import Path
import numpy as np
from coordination.video import annotate,restore_controller
from fullbody.balanced import BalancedEnv


class VideoCoordinationTest(unittest.TestCase):
    def test_event_validation(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'pose.npz'; t=np.linspace(0,4,41)
            p=np.zeros((41,33,3)); v=np.ones((41,33))
            np.savez(path,times=t,points=p,visibility=v)
            result=annotate(path,Path(d)/'events.json','explicit',[1.2])
            self.assertEqual(result['jump_times'],[1.2])
            with self.assertRaises(ValueError): annotate(path,Path(d)/'bad.json','explicit',[3.5])
            result=annotate(path,Path(d)/'auto.json','auto')
            self.assertEqual(result['jump_times'],[])

    def test_control_restore_and_physical_step(self):
        reference=Path('motions/self_test1/reference.npz')
        checkpoint=Path('runs/coordination/com_graph.pt').resolve()
        if not reference.exists() or not checkpoint.exists(): self.skipTest('local assets missing')
        with tempfile.TemporaryDirectory() as d:
            env=BalancedEnv(reference,d)
            config={'coordinator_checkpoint':str(checkpoint),'jump_times':[1.2],'jump_mode':'explicit'}
            restore_controller(env,{'leg_weight':.15,'arm_weight':1.,'video_controller':config})
            obs,_=env.reset()
            self.assertEqual(env.phase,0)
            seen=set()
            for _ in range(80):
                obs,r,f,t,info=env.step(np.zeros(29)); seen.add(info['jump']['phase'])
                self.assertTrue(np.isfinite(obs).all())
                self.assertTrue(info['coordinator_used'])
                if f or t: break
            self.assertIn('crouch',seen)
            env.close()


if __name__=='__main__': unittest.main()
