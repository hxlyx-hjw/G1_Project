"""全身模型、Gym环境、参考校验及PPO梯度更新回归。"""

# 支持从项目根目录、IDE及脚本所在目录启动。
from pathlib import Path as _ProjectPath
import sys as _project_sys
import os as _project_os
_project_root = _ProjectPath(__file__).resolve().parents[1]
_project_sys.path.insert(0, str(_project_root))
_project_os.chdir(_project_root)

import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
import mujoco
from fullbody.model import build_model,joint_info,nominal
from fullbody.env import ImitationEnv
from fullbody.ppo import ActorCritic,train,evaluate


class FullbodyTest(unittest.TestCase):
    def test_environment_and_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            model=build_model(root/'model')
            names,qa,va=joint_info(model)
            self.assertEqual(model.nu,29)
            self.assertTrue(np.all(model.actuator_biastype==0))
            times=np.linspace(0,.5,26)
            qpos=np.tile(nominal(model),(26,1))
            path=root/'reference.npz'
            np.savez(path,times=times,qpos=qpos,qvel=np.zeros((26,model.nv)),joint_names=names)
            env=ImitationEnv(path,root/'env')
            obs,_=env.reset(seed=1,options={'phase':0})
            self.assertEqual(obs.shape,(188,))
            self.assertTrue(env.observation_space.contains(obs))
            obs,reward,terminated,truncated,info=env.step(np.zeros(29))
            self.assertTrue(np.isfinite(reward))
            self.assertIn('contact_match',info)
            self.assertTrue(np.all(np.abs(env.data.ctrl)<=np.max(np.abs(env.model.actuator_ctrlrange),axis=1)+1e-8))
            with self.assertRaises(ValueError): env.step(np.full(29,np.nan))
            torch.manual_seed(4)
            initial=ActorCritic().actor[0].weight.detach().clone()
            train(path,root/'run',iterations=1,steps=16,seed=4)
            saved=torch.load(root/'run/policy.pt',weights_only=True)
            self.assertFalse(torch.equal(initial,saved['policy']['actor.0.weight']))
            self.assertTrue(all(torch.isfinite(v).all() for v in saved['policy'].values()))
            train(path,root/'resumed',iterations=1,steps=8,seed=4,resume=root/'run/policy.pt')
            resumed=torch.load(root/'resumed/policy.pt',weights_only=True)
            self.assertEqual(resumed['iteration'],2)
            result=evaluate(path,root/'run/policy.pt',root/'eval')
            self.assertTrue(result['policy_used'])
            self.assertGreater(result['steps'],0)
            self.assertIn('termination_reason',result)
            self.assertIn('completion_fraction',result)
            self.assertTrue((root/'eval/trajectory.json').is_file())

    def test_reference_validation(self):
        from fullbody.reference import Reference
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); model=build_model(root/'model')
            path=root/'bad.npz'
            np.savez(path,times=[0,0],qpos=np.tile(nominal(model),(2,1)),qvel=np.zeros((2,model.nv)),joint_names=joint_info(model)[0])
            with self.assertRaises(ValueError): Reference(path,model)


if __name__=='__main__': unittest.main()
