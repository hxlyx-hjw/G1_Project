"""导出用户视频训练策略的物理回放与误差报告。"""
from pathlib import Path
import json
import numpy as np
import torch
import mujoco
import imageio.v2 as imageio
from fullbody.balanced import BalancedEnv
from fullbody.ppo import ActorCritic
from viewer_camera import isometric_camera


def main():
    torch.set_num_threads(1)
    output=Path('outputs/self_video'); output.mkdir(parents=True,exist_ok=True)
    checkpoint=torch.load('runs/self_test1/training/policy.pt',weights_only=True)
    policy=ActorCritic.from_checkpoint(checkpoint,eval_mode=True)
    env=BalancedEnv('motions/self_test1/reference.npz',output/'render_model')
    obs,_=env.reset(options={'phase':0})
    camera=isometric_camera(env.model,lookat=(0,0,.65))
    renderer=mujoco.Renderer(env.model,height=480,width=640)
    frames=[]; errors=[]; actual=[]; target=[]; max_drift=0
    for i in range(334):
        with torch.no_grad(): a=policy.mean(torch.as_tensor(obs).unsqueeze(0))[0].numpy()
        obs,reward,fallen,truncated,info=env.step(a)
        ref,_=env.reference.sample(env.phase)
        actual.append(env.data.qpos[env.qa].copy()); target.append(ref[env.qa]); errors.append(info['joint_mse'])
        max_drift=max(max_drift,info['root_xy_error_m'])
        if i%3==0:
            renderer.update_scene(env.data,camera=camera); frames.append(renderer.render().copy())
        if fallen or truncated: break
    renderer.close()
    imageio.mimsave(output/'trained_replay.gif',frames,duration=60,loop=0)
    imageio.imwrite(output/'trained_preview.png',frames[len(frames)//2])
    actual=np.array(actual); target=np.array(target)
    report={'checkpoint_iteration':checkpoint['iteration'],'trained_iterations':40,'training_steps':20480,
        'device':'cuda','source_video':'../vedio_self/test1.mp4','source_interval_seconds':[4.181578947368421,10.85219298245614],
        'simulation_seconds':(i+1)*.02,'reference_completed':not fallen and env.phase>=env.reference.duration,
        'fallen':fallen,'joint_rmse_rad':float(np.sqrt(np.mean((actual-target)**2))),
        'leg_rmse_rad':float(np.sqrt(np.mean((actual[:,:12]-target[:,:12])**2))),
        'upper_body_rmse_rad':float(np.sqrt(np.mean((actual[:,12:]-target[:,12:])**2))),
        'max_root_xy_error_m':max_drift,'leg_reference_weight':env.video_leg_weight,
        'limitation':'腿部使用平衡先验，视频腿部偏移仅按0.15比例叠加；根部就地，没有复现原视频全幅横向步伐。'}
    (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False)); env.close()


if __name__=='__main__': main()
