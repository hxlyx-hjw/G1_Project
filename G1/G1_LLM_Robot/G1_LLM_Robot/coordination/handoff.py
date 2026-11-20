"""跳跃落地快照与行走控制器交接，按关节名称映射。"""
from pathlib import Path
import numpy as np
import mujoco
from runtime_io import atomic_write


def save(path,model,data,action,target,success):
    names=[model.joint(int(j)).name for j in model.actuator_trnid[:,0]]
    ids=model.actuator_trnid[:,0].astype(int)
    atomic_write(path,lambda p:_npz(p,version=1,success=success,joint_names=np.array(names),
        root_qpos=data.qpos[:7].copy(),root_qvel=data.qvel[:6].copy(),
        joint_qpos=data.qpos[model.jnt_qposadr[ids]].copy(),joint_qvel=data.qvel[model.jnt_dofadr[ids]].copy(),
        policy_action=action.copy(),leg_target=target.copy(),simulation_time=float(data.time)))


def _npz(path,**values):
    with Path(path).open('wb') as stream: np.savez_compressed(stream,**values)


def import_state(session,path):
    from g1_walkfree import config as C
    with np.load(path,allow_pickle=False) as z: values={k:z[k].copy() for k in z.files}
    if not bool(values['success']): raise ValueError('跳跃未稳定，不允许交接')
    for key in ('root_qpos','root_qvel','joint_qpos','joint_qvel','policy_action','leg_target'):
        if not np.isfinite(values[key]).all(): raise ValueError('交接数据含非法数值')
    with session._lock:
        if session._active: raise RuntimeError('请等待当前行走动作完成')
        c=session.ctrl; m,d=c.model,c.data
        names=values['joint_names'].tolist()
        expected=[m.joint(int(j)).name for j in m.actuator_trnid[:,0]]
        if set(names)!=set(expected): raise ValueError('跳跃和行走模型关节不一致')
        order=[names.index(n) for n in expected]
        ids=m.actuator_trnid[:,0].astype(int)
        mujoco.mj_resetData(m,d)
        d.qpos[:7]=values['root_qpos']; d.qvel[:6]=values['root_qvel']
        d.qpos[m.jnt_qposadr[ids]]=values['joint_qpos'][order]
        d.qvel[m.jnt_dofadr[ids]]=values['joint_qvel'][order]
        d.time=float(values['simulation_time'])
        c.pose[:]=values['joint_qpos'][order][12:]
        c._action[:]=values['policy_action']; c._target[:]=values['leg_target']
        c._obs[:]=0; c._counter=round(d.time/C.SIM_DT)
        c._start_xy=d.qpos[:2].copy(); c.stop(); session._cmd=[0.,0.,0.]
        d.ctrl[12:]=c.pose; session._last_fall=False
        mujoco.mj_forward(m,d)
        return {'imported':True,'position':d.qpos[:3].tolist(),'simulation_time':float(d.time),
            'source':str(path),'message':'已导入落地完整状态，后续行走命令从此位置继续；空闲暂停'}
