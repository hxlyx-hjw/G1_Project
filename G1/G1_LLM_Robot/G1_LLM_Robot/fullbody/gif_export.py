"""物理策略GIF导出；失败也保留片段，返回真实质量报告。"""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from runtime_io import load_checkpoint
import mujoco
from PIL import Image
from viewer_camera import isometric_camera
from .jobs import atomic_json
from .ppo import ActorCritic
from .env import ImitationEnv
from .balanced import BalancedEnv


def export(checkpoint,output,reference=None,fps=15,max_seconds=30):
    checkpoint=Path(checkpoint); output=Path(output)
    if output.suffix.lower()!='.gif': raise ValueError('输出文件必须为.gif')
    if not 1<=fps<=30 or not 0<max_seconds<=60: raise ValueError('fps:1~30，max_seconds:(0,60]')
    payload=load_checkpoint(checkpoint,map_location='cpu',weights_only=True)
    reference=Path(reference or payload['reference'])
    output.parent.mkdir(parents=True,exist_ok=True)
    cls=BalancedEnv if payload.get('algorithm')=='balanced_ppo' else ImitationEnv
    env=cls(reference,output.parent/(output.stem+'_model'),max_episode_seconds=payload.get('max_episode_seconds'))
    if payload['joint_names']!=env.names or payload.get('obs_dim')!=188: raise ValueError('权重与模型不兼容')
    if isinstance(env,BalancedEnv):
        env.video_leg_weight=payload.get('leg_weight',.15)
        env.video_arm_weight=payload.get('arm_weight',1.)
        env.airborne_reward=bool(payload.get('airborne_reward',True))
        from coordination.video import restore_controller
        restore_controller(env,payload)
    policy=ActorCritic.from_checkpoint(payload,eval_mode=True)
    torch.set_num_threads(1)
    obs,_=env.reset(options={'phase':0})
    duration=min(env.reference.duration,max_seconds)
    frames=[]; next_frame=0.; errors=[]; max_drift=0.; fallen=False
    camera=isometric_camera(env.model)
    with mujoco.Renderer(env.model,height=360,width=480) as renderer:
        for step in range(int(np.ceil(duration/.02))):
            with torch.no_grad(): action=policy.mean(torch.as_tensor(obs).unsqueeze(0))[0].numpy()
            obs,reward,fallen,ended,info=env.step(action)
            errors.append(info['joint_mse']); max_drift=max(max_drift,info['root_xy_error_m'])
            if env.elapsed+1e-8>=next_frame or fallen or ended:
                camera.lookat[:]=env.data.qpos[:3]; camera.lookat[2]=.65
                renderer.update_scene(env.data,camera=camera); frames.append(Image.fromarray(renderer.render().copy()))
                next_frame+=1/fps
            if fallen or ended: break
    frames[0].save(output,save_all=True,append_images=frames[1:],duration=round(1000/fps),loop=0)
    report={'output':str(output.resolve()),'checkpoint':str(checkpoint.resolve()),'frames':len(frames),'fps':fps,
        'fallen':bool(fallen),'simulation_seconds':env.elapsed,'reference_completed':bool(not fallen and env.phase>=env.reference.duration),
        'joint_rmse_rad':float(np.sqrt(np.mean(errors))),'max_root_xy_error_m':max_drift,
        'leg_weight':getattr(env,'video_leg_weight',None),'joint_assist':bool(policy.joint_assist_enabled),'render_mode':'physical_policy_rollout'}
    report['video_controller']=getattr(env,'video_controller_config',{})
    if getattr(env,'jump_controller',None):
        report['jump']=env.jump_controller.report()
        report['reference_completed']=report['reference_completed'] and report['jump']['all_scheduled_jumps_landed']
    atomic_json(output.with_suffix('.report.json'),report)
    env.close(); return report


def main():
    p=argparse.ArgumentParser(); p.add_argument('checkpoint'); p.add_argument('output'); p.add_argument('--reference'); p.add_argument('--fps',type=int,default=15); p.add_argument('--max-seconds',type=float,default=30); p.add_argument('--status')
    a=p.parse_args()
    try:
        if a.status: atomic_json(a.status,{'state':'rendering'})
        report=export(a.checkpoint,a.output,a.reference,a.fps,a.max_seconds)
        if a.status: atomic_json(a.status,{'state':'completed',**report})
        print(json.dumps(report))
    except Exception as exc:
        if a.status: atomic_json(a.status,{'state':'failed','error':str(exc)})
        raise


if __name__=='__main__': main()
