"""独立跳跃状态机：仅施加关节电机力矩，不使用根部外力或姿态瞬移。"""
from pathlib import Path
import time
import json
import numpy as np
import mujoco
from contextlib import nullcontext
from fullbody.model import build_model,joint_info,nominal
from .config import JumpConfig


def run(output,viewer=False,coordinator=None,strength=1.,record=False,keep_open=True,start_pose=None):
    if not .5<=strength<=1.6: raise ValueError('strength:.5~1.6')
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    cfg=JumpConfig(); model=build_model(out/'model'); data=mujoco.MjData(model)
    names,qa,va=joint_info(model); q=nominal(model)
    if start_pose is not None:
        pose=np.asarray(start_pose,float)
        if pose.shape!=(4,) or not np.isfinite(pose).all(): raise ValueError('start_pose需要[x,y,z,yaw]')
        q[:2]=pose[:2]; q[3:7]=[np.cos(pose[3]/2),0,0,np.sin(pose[3]/2)]
    data.qpos[:]=q; mujoco.mj_forward(model,data)
    feet=[model.body(s+'_ankle_roll_link').id for s in ('left','right')]
    spheres=[g for g in range(model.ngeom) if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_bodyid[g] in feet]
    data.qpos[2]-=min(data.geom_xpos[g,2]-model.geom_size[g,0] for g in spheres)
    mujoco.mj_forward(model,data)
    net=None
    if coordinator:
        from .network import Coordinator
        net=Coordinator(model,coordinator)
    default=q[qa].copy(); crouch=default.copy(); extend=default.copy()
    for side in ('left','right'):
        for suffix,value in [('hip_pitch',cfg.crouch_hip),('knee',cfg.crouch_knee),('ankle_pitch',cfg.crouch_ankle)]: crouch[names.index(side+'_'+suffix+'_joint')]=value
        for suffix,value in [('hip_pitch',cfg.thrust_hip),('knee',cfg.thrust_knee),('ankle_pitch',cfg.thrust_ankle)]: extend[names.index(side+'_'+suffix+'_joint')]=value
    kp=np.array([220 if 'knee' in n else 160 if 'hip' in n else 80 if 'ankle' in n else 100 for n in names],float)
    kd=np.array([6 if 'knee' in n else 4 if 'hip' in n else 3 for n in names],float)
    bounds=model.jnt_range[model.actuator_trnid[:,0].astype(int)]
    from mujoco import viewer as mujoco_viewer
    from viewer_camera import passive_viewer
    handle=passive_viewer(model,data) if viewer else None
    window=handle if handle is not None else nullcontext(None)
    history=[]; phases=[]; airborne=0.; max_airborne=0.; liftoff=None; landed=None; fallen=False; start_com=None; peak_com=0.; min_height=10.
    renderer=mujoco.Renderer(model,height=360,width=480) if record else None
    from viewer_camera import apply_isometric_camera
    frames=[]; cam=mujoco.MjvCamera()
    apply_isometric_camera(cam,lookat=(data.qpos[0],data.qpos[1],.65))
    from g1_walkfree.resources import find_policy
    from g1_walkfree import config as C
    from g1_walkfree.controller import WalkController
    import torch
    balance=torch.jit.load(find_policy(),map_location='cpu').eval()
    previous_action=np.zeros(12,np.float32); balance_target=C.DEFAULT_ANGLES.copy()
    correction=np.zeros(29); flight_started=None; landing_started=None
    wall=time.monotonic()
    with window as view:
        if view is not None:
            pelvis_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'pelvis')
            apply_isometric_camera(
                view.cam,lookat=(data.qpos[0],data.qpos[1],.65),
                tracking_body=pelvis_id if pelvis_id>=0 else None)
            view.sync()
        for step in range(2500):
            t=float(data.time)
            contacts=sum(1 for c in data.contact if model.geom_bodyid[c.geom1] in feet or model.geom_bodyid[c.geom2] in feet)
            if t<.5: phase='stand'; target=default
            elif t<.5+cfg.crouch_seconds:
                phase='crouch'; alpha=np.clip((t-.5)/cfg.crouch_seconds,0,1); alpha=alpha*alpha*(3-2*alpha); target=(1-alpha)*default+alpha*crouch
            elif liftoff is None and t<.5+cfg.crouch_seconds+cfg.thrust_seconds:
                phase='thrust'; target=extend
            elif contacts==0:
                phase='flight'; target=default*.4+crouch*.6
                if flight_started is None: flight_started=t
            else:
                phase='landing' if liftoff is not None else 'recover'; target=default.copy()
                if landing_started is None: landing_started=t
            if phase=='thrust' and contacts==0 and liftoff is None: liftoff=t
            if t>.5+cfg.crouch_seconds and contacts==0:
                airborne+=cfg.dt; max_airborne=max(max_airborne,airborne)
                if liftoff is None: liftoff=t
            else:
                if airborne>.02 and liftoff is not None and landed is None: landed=t
                airborne=0.
            if phase not in phases: phases.append(phase)
            if net and step%10==0: correction=net.correction(data) if contacts else np.zeros(29)
            desired=np.clip(target+correction,*bounds.T)
            if landed is not None:
                if step%10==0:
                    ph=(data.time%C.PHASE_PERIOD)/C.PHASE_PERIOD
                    obs=np.r_[data.qvel[3:6]*.25,WalkController._gravity_orientation(data.qpos[3:7]),np.zeros(3),
                        data.qpos[qa[:12]]-C.DEFAULT_ANGLES,data.qvel[va[:12]]*.05,previous_action,np.sin(2*np.pi*ph),np.cos(2*np.pi*ph)].astype(np.float32)
                    with torch.inference_mode(): previous_action=balance(torch.from_numpy(obs)[None])[0].numpy()
                    balance_target=C.DEFAULT_ANGLES+.25*previous_action
                desired[:12]=balance_target
            stiffness=kp*(strength if phase=='thrust' else 1.)
            damping=kd.copy()
            if landed is not None: stiffness[:12]=C.KPS; damping[:12]=C.KDS
            torque=stiffness*(desired-data.qpos[qa])-damping*data.qvel[va]
            data.ctrl[:]=np.clip(torque,*model.actuator_ctrlrange.T)
            mujoco.mj_step(model,data)
            com=data.subtree_com[model.body('pelvis').id].copy()
            if t>=.45 and start_com is None: start_com=com[2]
            if start_com is not None: peak_com=max(peak_com,com[2]-start_com)
            upright=data.xmat[model.body('pelvis').id].reshape(3,3)[2,2]
            min_height=min(min_height,data.qpos[2])
            fallen=bool(data.qpos[2]<.35 or upright<.35 or not np.isfinite(data.qpos).all())
            if step%10==0: history.append({'time':t,'phase':phase,'contacts':contacts,'height':float(data.qpos[2]),'com_z':float(com[2]),'upright':float(upright)})
            if renderer and step%30==0: renderer.update_scene(data,camera=cam); frames.append(renderer.render().copy())
            if view:
                if not view.is_running(): break
                if step%10==0: view.sync()
                delay=t-(time.monotonic()-wall)
                if delay>0: time.sleep(min(delay,.01))
            if fallen or (landed is not None and t>landed+cfg.settle_seconds): break
        if renderer:
            renderer.close()
            from PIL import Image
            ims=[Image.fromarray(f) for f in frames]; ims[0].save(out/'jump.gif',save_all=True,append_images=ims[1:],duration=60,loop=0)
        success=bool(max_airborne>=.04 and landed is not None and not fallen and float(data.time)>=landed+cfg.settle_seconds)
        report={'success':success,'fallen':fallen,'airborne_seconds':max_airborne,'liftoff_time':liftoff,'landing_time':landed,
            'peak_com_rise_m':float(peak_com),'final_root_height_m':float(data.qpos[2]),'phases':phases,'simulation_seconds':float(data.time),
            'coordinator_used':bool(net),'control':'joint_torques_only','note':'成功后可交接完整落地状态到行走会话；跳跃窗口保持快照直到用户关闭'}
        from .handoff import save
        save(out/'handoff.npz',model,data,previous_action,balance_target,success)
        report.update(handoff_ready=success,handoff_file=str((out/'handoff.npz').resolve()),window_held=bool(viewer and keep_open))
        from fullbody.jobs import atomic_json
        atomic_json(out/'report.json',report)
        (out/'trajectory.json').write_text(json.dumps(history),encoding='utf-8')
        if view and keep_open:
            while view.is_running() and not (out/'CLOSE_VIEWER').exists():
                view.sync()
                time.sleep(.05)
    # 跳跃仿真数据导出到 outputs/jump_<时间戳>/（失败不影响主流程）
    log_dir = None
    try:
        import shutil
        import uuid
        from datetime import datetime
        root = Path(__file__).resolve().parents[1]
        dest = root/'outputs'/('jump_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:4])
        dest.mkdir(parents=True, exist_ok=True)
        for name in ('report.json', 'trajectory.json', 'jump.gif'):
            src = out/name
            if src.is_file():
                shutil.copy2(src, dest/name)
        log_dir = str(dest)
    except Exception:
        log_dir = None
    if log_dir:
        report['log_dir'] = log_dir
    return report
