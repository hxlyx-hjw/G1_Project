"""基于骨段方向和足底约束的全身IK，支持接触约束的全局水平轨迹。"""
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import least_squares
from video_motion.pose import load
from video_motion.retarget import unit
from .global_track import estimate_root_trajectory, yaw_quaternion
from .model import build_model, joint_info, nominal

# 人体索引、机器人骨段。手腕的旋转没有可靠观测，保持默认值。
SEGMENTS = []
for side, ids in [('left',(11,13,15,23,25,27,29,31)),('right',(12,14,16,24,26,28,30,32))]:
    s,e,w,h,k,a,heel,toe=ids
    SEGMENTS.extend([(s,e,side+'_shoulder_pitch_link',side+'_elbow_link'),
                     (e,w,side+'_elbow_link',side+'_wrist_yaw_link'),
                     (h,k,side+'_hip_pitch_link',side+'_knee_link'),
                     (k,a,side+'_knee_link',side+'_ankle_roll_link')])


def convert(pose_file, output, root_mode='in_place', min_visibility=0.5):
    if root_mode not in ('in_place', 'contact_grounded', 'auto'):
        raise ValueError('root_mode 必须为 in_place/contact_grounded/auto')
    if not 0 <= float(min_visibility) <= 1:
        raise ValueError('min_visibility 应在 0~1')
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    model = build_model(output.parent/'model')
    data = mujoco.MjData(model)
    names, addresses, _ = joint_info(model)
    ids = model.actuator_trnid[:,0].astype(int)
    bounds = model.jnt_range[ids]
    default = nominal(model)
    previous = default[addresses].copy()
    times,points,visibility = load(pose_file)
    trajectory = None
    fallback = None
    if root_mode != 'in_place':
        try:
            trajectory = estimate_root_trajectory(points, visibility, times)
        except (ValueError, np.linalg.LinAlgError) as exc:
            if root_mode == 'contact_grounded':
                fallback = str(exc)
    optimized = [i for i,n in enumerate(names) if 'wrist' not in n]
    frames, errors, rejected, contacts = [],[],[],[]
    last = default.copy()
    last_error = 0.
    body_pairs = [(model.body(a).id,model.body(b).id) for _,_,a,b in SEGMENTS]
    feet = [model.body(s+'_ankle_roll_link').id for s in ('left','right')]
    for index,(p,v) in enumerate(zip(points,visibility)):
        needed = [11,12,23,24,25,26,27,28]
        if np.min(v[needed]) < float(min_visibility):
            frames.append(last.copy()); errors.append(last_error); rejected.append(True); contacts.append([False,False])
            continue
        try:
            left = unit(p[23]-p[24])
            up = unit((p[11]+p[12]-p[23]-p[24])/2)
            forward = unit(np.cross(left,up))
            up = unit(np.cross(forward,left))
            basis = np.column_stack((forward,left,up))
            shoulder_left = unit(p[11]-p[12])
            shoulder_up = unit(np.cross(unit(np.cross(shoulder_left,up)),shoulder_left))
            torso = basis.T @ np.column_stack((unit(np.cross(shoulder_left,shoulder_up)),shoulder_left,shoulder_up))
            targets = [(basis.T@unit(p[b]-p[a]), min(v[a],v[b])) for a,b,_,_ in SEGMENTS]
        except ValueError:
            frames.append(last.copy()); errors.append(last_error); rejected.append(True); contacts.append([False,False]); continue
        def residual(x):
            q = previous.copy()
            q[optimized] = x
            data.qpos[:] = default
            data.qpos[addresses] = q
            mujoco.mj_forward(model,data)
            residuals=[]
            for (a,b),(target,confidence) in zip(body_pairs,targets):
                if confidence >= .5:
                    residuals.extend(2*(unit(data.xpos[b]-data.xpos[a])-target))
            residuals.extend((data.xmat[model.body('torso_link').id].reshape(3,3)-torso).ravel()*.5)
            for side, heel, toe in [('left',29,31),('right',30,32)]:
                if min(v[heel],v[toe]) >= .5:
                    direction = p[toe] - p[heel]
                    if np.linalg.norm(direction) > 1e-6:
                        foot_dir = data.xmat[model.body(side+'_ankle_roll_link').id].reshape(3,3)[:,0]
                        residuals.extend(.5*(foot_dir-basis.T@unit(direction)))
            residuals.extend(.08*(x-previous[optimized]))
            return np.array(residuals)
        fit=least_squares(
            residual,previous[optimized],
            bounds=(bounds[optimized,0]+1e-5,bounds[optimized,1]-1e-5),
            loss='soft_l1',f_scale=.08,max_nfev=45)
        previous[optimized] = fit.x
        if index:
            delta = 4.5*(times[index]-times[index-1])
            previous = np.clip(previous,last[addresses]-delta,last[addresses]+delta)
        q=default.copy(); q[addresses]=previous
        data.qpos[:]=q; mujoco.mj_forward(model,data)
        # 将最低脚底球放到地面，保留双足相对高度，允许抬腿。
        foot_geoms=[i for i in range(model.ngeom) if model.geom_type[i]==mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_bodyid[i] in feet]
        lowest=min(data.geom_xpos[i,2]-model.geom_size[i,0] for i in foot_geoms)
        q[2]-=lowest
        data.qpos[:]=q; mujoco.mj_forward(model,data)
        contact=[bool(min(data.geom_xpos[i,2]-model.geom_size[i,0] for i in foot_geoms if model.geom_bodyid[i]==f)<.025) for f in feet]
        last=q; last_error=float(np.sqrt(np.mean(fit.fun**2)))
        frames.append(q.copy()); errors.append(last_error); rejected.append(False); contacts.append(contact)
    if all(rejected):
        raise ValueError('没有肩、髋、膝、踝同时可见的有效全身帧')
    qpos=np.array(frames)
    contacts_out = np.asarray(contacts, bool)
    if trajectory is not None:
        root_xy = trajectory['root'][:, :2]
        qpos[:, 0:2] = root_xy
        qpos[:, 3:7] = yaw_quaternion(trajectory['yaw'])
        contacts_out = trajectory['contacts'].copy()
        # 接触帧用机器人实际足底几何确定根高；空中帧在最近接触帧之间插值，
        # 避免再次把抬起的脚强行压回地面。
        ground_z = np.full(len(qpos), np.nan)
        foot_geoms = [i for i in range(model.ngeom)
                      if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE
                      and model.geom_bodyid[i] in feet]
        for frame, q in enumerate(qpos):
            if not contacts_out[frame].any():
                continue
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            low = min(data.geom_xpos[g, 2] - model.geom_size[g, 0] for g in foot_geoms)
            ground_z[frame] = q[2] - low
        known = np.flatnonzero(np.isfinite(ground_z))
        if len(known):
            for frame in range(len(qpos)):
                if np.isfinite(ground_z[frame]):
                    continue
                before = known[known < frame]
                after = known[known > frame]
                if len(before) and len(after):
                    left, right = int(before[-1]), int(after[0])
                    ratio = (frame - left) / max(1, right - left)
                    ground_z[frame] = ground_z[left] * (1 - ratio) + ground_z[right] * ratio
                elif len(before):
                    ground_z[frame] = ground_z[int(before[-1])]
                else:
                    ground_z[frame] = ground_z[int(after[0])]
            qpos[:, 2] = ground_z
    qvel=np.zeros((len(times),model.nv))
    for i in range(len(times)-1):
        mujoco.mj_differentiatePos(model,qvel[i],times[i+1]-times[i],qpos[i],qpos[i+1])
    qvel[-1]=qvel[-2]
    with output.open('wb') as f:
        np.savez_compressed(f,times=times,qpos=qpos,qvel=qvel,joint_names=names,
            contacts=contacts_out,ik_error=errors,rejected=rejected,version=2,
            root_mode=('contact_grounded' if trajectory is not None else 'in_place'),
            root_scale=(float(trajectory['scale']) if trajectory is not None else 1.0),
            root_track_method=(trajectory['method'] if trajectory is not None else 'in_place'),
            root_track_fallback=(fallback or ''),
            source=str(Path(pose_file).resolve()))
    displacement = (float(np.linalg.norm(qpos[-1,:2]-qpos[0,:2])) if trajectory is not None else 0.)
    return {'output':str(output),'frames':len(times),'rejected_frames':int(sum(rejected)),
            'mean_ik_residual':float(np.mean(errors)),
            'root_mode':('contact_grounded' if trajectory is not None else 'in_place'),
            'root_displacement_m':displacement,
            'contact_frames':int(np.sum(contacts_out.any(axis=1))),
            'root_track_fallback':fallback}


class Reference:
    def __init__(self,path,model):
        with np.load(path,allow_pickle=False) as z:
            self.times,self.qpos,self.qvel = [z[k].copy() for k in ('times','qpos','qvel')]
            names=z['joint_names'].tolist()
            contacts=(np.asarray(z['contacts'],bool) if 'contacts' in z.files else None)
            root_mode=(np.asarray(z['root_mode']).item() if 'root_mode' in z.files else 'in_place')
        expected=joint_info(model)[0]
        if names != expected or self.qpos.shape!=(len(self.times),model.nq) or self.qvel.shape!=(len(self.times),model.nv):
            raise ValueError('参考动作与模型关节或维度不匹配')
        if len(self.times)<2 or np.any(np.diff(self.times)<=0) or not all(np.isfinite(a).all() for a in (self.times,self.qpos,self.qvel)):
            raise ValueError('参考动作时间戳或数值非法')
        self.duration=float(self.times[-1]-self.times[0])
        self.times-=self.times[0]
        self.contacts=(contacts if contacts is not None and contacts.shape==(len(self.times),2) else None)
        self.root_mode=str(root_mode)

    def contact_at(self,t):
        if self.contacts is None: return None
        index=int(np.clip(np.searchsorted(self.times,float(t),side='right')-1,0,len(self.times)-1))
        return self.contacts[index].copy()

    def sample(self,t):
        t=np.clip(t,0,self.duration)
        q=np.array([np.interp(t,self.times,self.qpos[:,i]) for i in range(self.qpos.shape[1])])
        q[3:7]/=np.linalg.norm(q[3:7])
        v=np.array([np.interp(t,self.times,self.qvel[:,i]) for i in range(self.qvel.shape[1])])
        return q,v
