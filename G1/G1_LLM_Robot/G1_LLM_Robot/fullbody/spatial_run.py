"""空间关键点技能的独立回放和节点策略联动。"""
from pathlib import Path
import json
import time
import numpy as np
import mujoco
from .spatial_skills import get,KEYPOINT_BODIES
from .model import build_model,joint_info
from .reference import Reference
from .ppo import evaluate
from viewer_camera import passive_viewer


def run(skill_id,mode='spatial',viewer=False):
    item=get(skill_id); job=Path(item['job_dir']); report=item['evaluation']
    if mode=='node':
        if not item.get('node_skill_id'): raise ValueError('该空间技能没有关联节点策略技能')
        from .skills import get as get_node
        node=get_node(item['node_skill_id'])
        return evaluate(node['reference'],node['checkpoint'],job/'spatial_node_run',viewer=viewer)
    if mode not in ('spatial','linked'): raise ValueError('mode必须是spatial/node/linked')
    model=build_model(job/'spatial_run_model'); data=mujoco.MjData(model)
    with np.load(item['points'],allow_pickle=False) as z:
        times=z['times'].astype(float); positions=z['positions'].astype(float); names=z['keypoint_names'].tolist()
    ids=[model.body(n).id for n in names]
    context=passive_viewer(model,data) if viewer else None
    if context is not None:
        from viewer_camera import apply_isometric_camera
        pelvis_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'pelvis')
        apply_isometric_camera(
            context.cam,tracking_body=pelvis_id if pelvis_id>=0 else None)
        context.sync()
    errors=[]; frames=0
    try:
        for t,targets in zip(times,positions):
            # 空间技能直接播放登记的绝对关键点；以参考qpos作为稳定的全身姿态基线。
            ref=Reference(item['reference'],model); q,v=ref.sample(float(t)); data.qpos[:]=q; data.qvel[:]=v; mujoco.mj_forward(model,data)
            errors.append(float(np.sqrt(np.mean((data.xpos[ids]-targets)**2)))); frames+=1
            if context: context.sync(); time.sleep(max(0,.02))
    finally:
        if context: context.close()
    result={'skill_id':skill_id,'mode':mode,'frames':frames,'reference_seconds':float(times[-1]),
        'spatial_rmse_m':float(np.mean(errors)),'max_spatial_error_m':float(np.max(errors)),
        'node_skill_id':item.get('node_skill_id'),'note':'空间模式使用绝对关键点参考与登记qpos基线；节点策略模式需调用mode=node'}
    return result
