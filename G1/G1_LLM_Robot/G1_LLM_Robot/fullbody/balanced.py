"""已训练腿部平衡先验 + 视频参考残差；不是直接全身PD。"""
import numpy as np
import torch
import mujoco
from .env import ImitationEnv
from .motion import MotionConfig, amplitude, arm_amplitude, balance_margin
from g1_walkfree import config as C
from g1_walkfree.resources import find_policy


def heading_command(reference_quaternion,current_quaternion,track_reference):
    """把朝向误差转成平衡策略的偏航速度指令。"""
    quat=np.asarray(current_quaternion,float)
    yaw=np.arctan2(2*(quat[0]*quat[3]+quat[1]*quat[2]),1-2*(quat[2]**2+quat[3]**2))
    if track_reference:
        ref=np.asarray(reference_quaternion,float)
        reference_yaw=np.arctan2(2*(ref[0]*ref[3]+ref[1]*ref[2]),1-2*(ref[2]**2+ref[3]**2))
        yaw_error=(reference_yaw-yaw+np.pi)%(2*np.pi)-np.pi
        value=1.1*yaw_error
    else:
        value=-.6*yaw
    return float(np.clip(value,-.5,.5))


class BalancedEnv(ImitationEnv):
    def __init__(self,reference,workdir,seed=0,max_episode_seconds=None):
        self.balance=torch.jit.load(find_policy(),map_location='cpu').eval()
        self.video_leg_weight=.15
        self.video_arm_weight=1.
        self.motion=MotionConfig()
        self._last_motion_gain=float(self.motion.max_gain)
        self._last_balance_margin=1.
        self.video_controller_config={}
        self.coordinator=None
        self.jump_controller=None
        self.balance_action=np.zeros(12,np.float32)
        self.anchor=np.zeros(2)
        self.anchor_yaw=0.
        super().__init__(reference,workdir,seed,max_episode_seconds=max_episode_seconds)
        self.pelvis_id=self.model.body('pelvis').id
        self.kp[:12]=C.KPS; self.kd[:12]=C.KDS
        self.kp[12:]=np.array([150 if 'waist' in n else 100 if 'shoulder' in n else 80 if 'elbow' in n else 30 for n in self.names[12:]])
        self.kd[12:]=2*np.sqrt(self.kp[12:])*.45

    def configure_motion(self,motion_gain=None,legacy=False,arm_priority=None,**overrides):
        """配置动作包络；legacy=True 精确恢复旧权重的控制语义。"""
        values=dict(overrides)
        if motion_gain is not None:
            values['max_gain']=float(motion_gain)
        self.motion=MotionConfig.from_mapping(values,legacy=legacy)
        self._last_motion_gain=float(self.motion.max_gain)
        self.arm_tracking_priority=bool((not legacy and self.motion.guard) if arm_priority is None else arm_priority)
        self.arm_stability_floor=float(self.motion.arm_stability_floor)
        self.arm_stability_power=float(self.motion.arm_stability_power)
        return self.motion

    def _balance_state(self):
        rotation=self.data.xmat[self.pelvis_id].reshape(3,3)
        upright=float(rotation[2,2])
        speed=float(np.linalg.norm(self.data.qvel[:3]))
        com=self.data.subtree_com[self.pelvis_id][:2]
        support=np.mean(self.data.xpos[self.key_bodies[2:],:2],axis=0)
        com_error=float(np.linalg.norm(com-support))
        return balance_margin(upright,speed,com_error,self.motion),upright,speed,com_error

    def configure_video_controller(self,config):
        self.video_controller_config=dict(config)
        times=config.get('jump_times',[])
        if any(not np.isfinite(t) or t<.7 or t>self.reference.duration-1.6 for t in times): raise ValueError('跳跃时刻须保留前0.7秒与后1.6秒')
        if any(b-a<2.5 for a,b in zip(times,times[1:])): raise ValueError('跳跃时间须递增且相隔至少2.5秒')
        if config.get('coordinator_checkpoint'):
            from coordination.network import Coordinator
            self.coordinator=Coordinator(self.model,config['coordinator_checkpoint'])
        else: self.coordinator=None
        if config.get('jump_times'):
            from coordination.video_jump import VideoJumpController
            self.jump_controller=VideoJumpController(self,config['jump_times'])
        else: self.jump_controller=None

    def step(self,action):
        obs,reward,terminated,truncated,info=super().step(action)
        info['coordinator_used']=self.coordinator is not None
        info['motion_gain']=self._last_motion_gain
        info['balance_margin']=self._last_balance_margin
        info['arm_gain']=arm_amplitude(self._last_balance_margin,self.motion,self._last_motion_gain)
        info['rl_pd_influence']=float(self.motion.rl_pd_influence)
        info['hand_rl_pd_influence']=self.motion.effective_hand_influence()
        info['foot_rl_pd_influence']=self.motion.effective_foot_influence()
        info['jump']=self.jump_controller.report() if self.jump_controller else {'phase':'tracking','events':[],'jump_verified':False}
        return obs,reward,terminated,truncated,info

    def reset(self,*,seed=None,options=None):
        if self.jump_controller and not options: options={'phase':0}
        obs,info=super().reset(seed=seed,options=options)
        if self.jump_controller: self.jump_controller.reset()
        self.data.qpos[self.qa[:12]]=C.DEFAULT_ANGLES
        self.data.qpos[2]=.79
        self.data.qvel[:]=0
        self.balance_action[:]=0
        self._last_balance_margin=1.
        self._last_motion_gain=float(self.motion.max_gain)
        mujoco.mj_forward(self.model,self.data)
        self.anchor=self.data.qpos[:2].copy()
        self.anchor_yaw=0.
        return self.observation(),info

    def control_target(self,q,action):
        from g1_walkfree.controller import WalkController
        margin,_,_,_=self._balance_state()
        gain=amplitude(margin,self.motion)
        self._last_balance_margin=margin
        self._last_motion_gain=gain
        quat=self.data.qpos[3:7]
        yaw=np.arctan2(2*(quat[0]*quat[3]+quat[1]*quat[2]),1-2*(quat[2]**2+quat[3]**2))
        yaw_command=heading_command(q[3:7],quat,self.reference.root_mode.startswith('contact_grounded'))
        delta=q[:2]-self.data.qpos[:2]
        rot=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])
        xy=rot@delta
        cmd=np.array([np.clip(1.3*xy[0],-.4,.4),np.clip(1.3*xy[1],-.35,.35),np.clip(yaw_command,-.5,.5)],np.float32)
        phase=(self.data.time%C.PHASE_PERIOD)/C.PHASE_PERIOD
        obs=np.r_[self.data.qvel[3:6]*C.ANG_VEL_SCALE,WalkController._gravity_orientation(quat),
            cmd*C.CMD_SCALE,(self.data.qpos[self.qa[:12]]-C.DEFAULT_ANGLES),
            self.data.qvel[self.va[:12]]*.05,self.balance_action,np.sin(2*np.pi*phase),np.cos(2*np.pi*phase)].astype(np.float32)
        with torch.inference_mode(): self.balance_action=self.balance(torch.from_numpy(obs).unsqueeze(0))[0].numpy()
        target=q[self.qa].copy()+gain*self.motion.waist_action_scale*action
        from .model import nominal
        default=nominal(self.model)[self.qa]
        waist_indices=[i for i,n in enumerate(self.names) if n.startswith('waist')]
        waist_ref=q[self.qa[waist_indices]]-default[waist_indices]
        target[waist_indices]=default[waist_indices]+gain*(waist_ref+self.motion.waist_action_scale*action[waist_indices])
        arm_indices=[i for i,n in enumerate(self.names) if 'shoulder' in n or 'elbow' in n or 'wrist' in n]
        arm_ref=np.clip(q[self.qa[arm_indices]]-default[arm_indices],
                        -self.motion.max_arm_reference_delta,self.motion.max_arm_reference_delta)
        arm_gain=arm_amplitude(margin,self.motion,gain)
        hand_influence=self.motion.effective_hand_influence()
        target[arm_indices]=default[arm_indices]+(1.0-hand_influence)*arm_gain*(
            self.video_arm_weight*arm_ref+self.motion.arm_action_scale*action[arm_indices])
        # 保留稳定步态大部分控制权，视频腿部动作作为有限残差。
        video_delta=np.clip(q[self.qa[:12]]-C.DEFAULT_ANGLES,
                            -self.motion.max_leg_reference_delta,self.motion.max_leg_reference_delta)
        foot_influence=self.motion.effective_foot_influence()
        target[:12]=(C.DEFAULT_ANGLES+foot_influence*.25*self.balance_action
                     +gain*(self.video_leg_weight*video_delta
                            +self.motion.leg_action_scale*action[:12]))
        if self.jump_controller:
            target=self.jump_controller.target(target)
        foot=set(self.key_bodies[2:])
        contact=any(int(self.model.geom_bodyid[c.geom1]) in foot or int(self.model.geom_bodyid[c.geom2]) in foot for c in self.data.contact)
        if self.coordinator and contact:
            target+=self.coordinator.correction(self.data)
        return np.clip(target,*self.limits.T)
