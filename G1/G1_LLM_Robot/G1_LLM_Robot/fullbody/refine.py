"""对视频参考加入脚部落点/足底水平/支撑重心约束。"""
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from video_motion.retarget import unit
from .model import build_model,joint_info


def refine(pose,reference,output):
    output=Path(output); model=build_model(output.parent/'model')
    data=mujoco.MjData(model)
    names,qa,va=joint_info(model)
    with np.load(reference) as z: contents={k:z[k] for k in z.files}
    with np.load(pose) as z: points=z['points']; visibility=z['visibility']
    root_mode_value=(str(np.asarray(contents['root_mode']).item()) if 'root_mode' in contents else 'in_place')
    global_mode=root_mode_value.startswith('contact_grounded')
    original=contents['qpos'].copy(); results=[]
    bounds=model.jnt_range[model.actuator_trnid[:12,0].astype(int)]
    feet=[model.body(s+'_ankle_roll_link').id for s in ('left','right')]
    last=original[0,qa[:12]].copy()
    for frame,(p,raw) in enumerate(zip(points,original)):
        needed=[11,12,23,24,25,26,27,28]
        if np.min(visibility[frame,needed])<.35:
            results.append(raw.copy()); continue
        try:
            left=unit(p[23]-p[24]); up=unit((p[11]+p[12]-p[23]-p[24])/2)
            forward=unit(np.cross(left,up)); up=unit(np.cross(forward,left)); basis=np.column_stack([forward,left,up])
        except ValueError:
            results.append(raw.copy()); continue
        pelvis=(p[23]+p[24])/2
        leg_length=np.mean([np.linalg.norm(p[k]-p[h])+np.linalg.norm(p[a]-p[k]) for h,k,a in [(23,25,27),(24,26,28)]])
        scale=.62/leg_length
        targets=np.array([basis.T@(p[a]-pelvis)*scale for a in [27,28]])
        # 足部间距不能小于模型髋部宽度；不强制所有动作双脚接地。
        middle=np.mean(targets[:,1]); width=max(.16,targets[0,1]-targets[1,1])
        targets[0,1]=middle+width/2; targets[1,1]=middle-width/2
        targets[:,0]=np.clip(targets[:,0],-.20,.25)
        targets[:,1]=np.clip(targets[:,1],-.36,.36)
        targets[:,2]-=np.min(targets[:,2]); targets[:,2]+=.035
        # 放宽抬脚高度以支持更大步幅，同时仍限制单目深度噪声造成的伪抬脚。
        targets[:,2]=np.clip(targets[:,2],.035,.18)
        def residual(x):
            q=raw.copy(); q[qa[:12]]=x[:12]; q[2]=x[12]
            data.qpos[:]=q; mujoco.mj_forward(model,data)
            foot=np.array([data.xpos[b] for b in feet])
            errors=list((8*(foot-targets)).ravel())
            for b in feet:
                rot=data.xmat[b].reshape(3,3)
                errors.extend(1.5*(rot[:,2]-[0,0,1]))
                errors.append(.5*rot[1,0])
            support=foot[np.argmin(foot[:,2]),:2] if abs(targets[0,2]-targets[1,2])>.035 else foot[:,:2].mean(axis=0)
            errors.extend(2*(data.subtree_com[1,:2]-support))
            errors.extend(.12*(x[:12]-raw[qa[:12]]))
            errors.extend(.2*(x[:12]-last))
            return errors
        fit=least_squares(
            residual,np.r_[last,raw[2]],
            bounds=(np.r_[bounds[:,0]+1e-5,.45],np.r_[bounds[:,1]-1e-5,.95]),
            loss='soft_l1',f_scale=.08,max_nfev=50)
        q=raw.copy(); q[qa[:12]]=fit.x[:12]
        if global_mode: q[:3]=raw[:3]
        else: q[2]=fit.x[12]
        results.append(q); last=fit.x[:12]
    qpos=np.array(results)
    # 在原始帧时间轴平滑，防止目标角加速度尖峰。
    if len(qpos)>=7:
        qpos[:,qa]=savgol_filter(qpos[:,qa],7,2,axis=0)
        if global_mode:
            qpos[:,:2]=savgol_filter(qpos[:,:2],5,2,axis=0)
        qpos[:,2]=savgol_filter(qpos[:,2],7,2)
    sphere=[g for g in range(model.ngeom) if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_bodyid[g] in feet]
    contacts=[]
    if global_mode:
        stored=np.asarray(contents.get('contacts'))
        expected=stored.astype(bool) if stored.shape==(len(qpos),2) else np.ones((len(qpos),2),bool)
        ground=np.full(len(qpos),np.nan)
        for frame,q in enumerate(qpos):
            if not expected[frame].any(): continue
            data.qpos[:]=q; mujoco.mj_forward(model,data)
            low=min(data.geom_xpos[g,2]-model.geom_size[g,0] for g in sphere)
            ground[frame]=q[2]-low
        known=np.flatnonzero(np.isfinite(ground))
        if len(known):
            for frame in range(len(qpos)):
                if np.isfinite(ground[frame]): continue
                before=known[known<frame]; after=known[known>frame]
                if len(before) and len(after):
                    a,b=int(before[-1]),int(after[0]); ratio=(frame-a)/max(1,b-a)
                    ground[frame]=ground[a]*(1-ratio)+ground[b]*ratio
                elif len(before): ground[frame]=ground[int(before[-1])]
                else: ground[frame]=ground[int(after[0])]
            qpos[:,2]=ground
        for q in qpos:
            data.qpos[:]=q; mujoco.mj_forward(model,data)
            contacts.append([bool(min(data.geom_xpos[g,2]-model.geom_size[g,0] for g in sphere if model.geom_bodyid[g]==f)<.025) for f in feet])
    else:
        for q in qpos:
            data.qpos[:]=q; mujoco.mj_forward(model,data)
            low=min(data.geom_xpos[g,2]-model.geom_size[g,0] for g in sphere)
            q[2]-=low
        for q in qpos:
            data.qpos[:]=q; mujoco.mj_forward(model,data)
            contacts.append([bool(min(data.geom_xpos[g,2]-model.geom_size[g,0] for g in sphere if model.geom_bodyid[g]==f)<.025) for f in feet])
    qvel=np.zeros((len(qpos),model.nv)); times=contents['times']
    for i in range(len(times)-1): mujoco.mj_differentiatePos(model,qvel[i],times[i+1]-times[i],qpos[i],qpos[i+1])
    qvel[-1]=qvel[-2]
    contents.update(qpos=qpos,qvel=qvel,root_mode=('contact_grounded_support_refined' if global_mode else 'in_place_support_refined'),contacts=np.asarray(contacts,bool))
    np.savez_compressed(output,**contents)
    print('Refined reference:',output)


if __name__=='__main__':
    refine('motions/self_test1/pose.npz','motions/self_test1/reference_raw.npz','motions/self_test1/reference.npz')
