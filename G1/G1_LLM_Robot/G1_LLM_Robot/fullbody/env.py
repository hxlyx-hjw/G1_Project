"""29维动作的物理模仿训练环境，50Hz控制/500Hz物理。"""
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from .model import build_model,joint_info,gains
from .reference import Reference
from .support import support_margin


class ImitationEnv(gym.Env):
    def __init__(self,reference,workdir,seed=0,max_episode_seconds=None):
        self.model=build_model(workdir)
        self.data=mujoco.MjData(self.model)
        self.reference=Reference(reference,self.model)
        self.max_episode_seconds=(float(max_episode_seconds) if max_episode_seconds is not None
                                  else max(10.,self.reference.duration+2.))
        if self.max_episode_seconds<=0: raise ValueError('max_episode_seconds必须为正数')
        self.action_scale=.25
        self.reference_data=mujoco.MjData(self.model)
        self.key_bodies=[self.model.body(n).id for n in (
            'left_wrist_yaw_link','right_wrist_yaw_link','left_ankle_roll_link','right_ankle_roll_link')]
        self.foot_bodies=set(self.key_bodies[2:])
        self.names,self.qa,self.va=joint_info(self.model)
        self.kp,self.kd=gains(self.names)
        self.limits=self.model.jnt_range[self.model.actuator_trnid[:,0].astype(int)]
        self.arm_indices=[i for i,n in enumerate(self.names) if any(k in n for k in ('shoulder','elbow','wrist'))]
        self.non_arm_indices=[i for i in range(len(self.names)) if i not in self.arm_indices]
        # 平衡训练可打开：手臂跟踪在失衡时让位于机体稳定。
        self.arm_tracking_priority=False
        self.arm_stability_floor=0.0
        self.arm_stability_power=1.0
        self.arm_task_fraction=0.30
        # 腾空时降低模仿任务与动作惩罚，并以足部支撑凸包约束质心。
        self.airborne_reward=True
        self.airborne_task_scale=0.55
        self.airborne_action_penalty_scale=0.45
        self.support_reach_m=0.12
        self.support_horizon_s=0.08
        self.support_reward_weight=0.12
        self.support_penalty_weight=0.30
        self.support_softness=0.05
        self.action_space=spaces.Box(-1.,1.,(29,),dtype=np.float32)
        self.observation_space=spaces.Box(-np.inf,np.inf,(188,),dtype=np.float32)
        self.rng=np.random.default_rng(seed)
        self.previous=np.zeros(29)
        self.elapsed=0.
        self.phase=0.
        self.reset(seed=seed)

    def observation(self):
        q,v=self.reference.sample(self.phase)
        rotation=self.data.xmat[self.model.body('pelvis').id].reshape(3,3)
        phase=2*np.pi*self.phase/self.reference.duration
        obs=np.r_[rotation.T@np.array([0,0,-1.]),self.data.qvel[3:6]*.25,
            q[:3]-self.data.qpos[:3], self.data.qvel[:3],
            self.data.qpos[self.qa],self.data.qvel[self.va]*.05,q[self.qa],v[self.va]*.05,
            q[self.qa]-self.data.qpos[self.qa],self.previous,np.sin(phase),np.cos(phase)]
        return np.clip(obs,-20,20).astype(np.float32)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng=np.random.default_rng(seed)
        self.phase=float((options or {}).get('phase',self.rng.uniform(0,max(.001,self.reference.duration-.3))))
        self.phase=float(np.clip(self.phase,0,max(0,self.reference.duration-.02)))
        q,v=self.reference.sample(self.phase)
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[:]=q
        self.data.qvel[:]=v
        self.data.qvel[:6]=0
        self.elapsed=0.; self.previous[:]=0
        mujoco.mj_forward(self.model,self.data)
        return self.observation(),{}

    def control_target(self,q,action):
        return np.clip(q[self.qa]+self.action_scale*action,*self.limits.T)

    def foot_contact_count(self):
        if not hasattr(self,'foot_bodies'): return 0
        return sum(1 for contact in self.data.contact
            if int(self.model.geom_bodyid[contact.geom1]) in self.foot_bodies
            or int(self.model.geom_bodyid[contact.geom2]) in self.foot_bodies)

    def _arm_tracking_scale(self):
        if not self.arm_tracking_priority: return 1.0
        margin=float(np.clip(getattr(self,'_last_balance_margin',1.0),0.,1.))
        return float(self.arm_stability_floor+(1.-self.arm_stability_floor)*margin**self.arm_stability_power)

    def _airborne_support(self,airborne):
        if not airborne: return None,None
        com=self.data.subtree_com[self.model.body('pelvis').id][:2]
        margin,hull=support_margin(self.model,self.data,self.key_bodies[2:],com,
            self.support_reach_m,self.data.qvel[:2],self.support_horizon_s)
        return float(margin),hull

    def step(self,action):
        action=np.asarray(action,dtype=float)
        if action.shape!=(29,) or not np.isfinite(action).all():
            raise ValueError('动作必须是29维有限数值')
        action=np.clip(action,-1,1)
        q,v=self.reference.sample(self.phase)
        target=self.control_target(q,action)
        for _ in range(10):
            torque=self.kp*(target-self.data.qpos[self.qa])-self.kd*self.data.qvel[self.va]
            self.data.ctrl[:]=np.clip(torque,*self.model.actuator_ctrlrange.T)
            mujoco.mj_step(self.model,self.data)
        self.phase+=.02; self.elapsed+=.02
        q,v=self.reference.sample(self.phase)
        joint_error=(self.data.qpos[self.qa]-q[self.qa])**2
        position=float(np.mean(joint_error))
        velocity=float(np.mean((self.data.qvel[self.va]-v[self.va])**2))
        root=float(np.sum((self.data.qpos[:3]-q[:3])**2))
        upright=float(self.data.xmat[self.model.body('pelvis').id].reshape(3,3)[2,2])
        self.reference_data.qpos[:]=q
        mujoco.mj_forward(self.model,self.reference_data)
        end_errors=(self.data.xpos[self.key_bodies]-self.reference_data.xpos[self.key_bodies])**2
        end_error=float(np.mean(end_errors))
        expected_contact=set()
        actual_contact=set()
        stored_contact=self.reference.contact_at(self.phase)
        if stored_contact is not None:
            if stored_contact[0]: expected_contact.add(self.key_bodies[2])
            if stored_contact[1]: expected_contact.add(self.key_bodies[3])
        else:
            for geom in range(self.model.ngeom):
                body=int(self.model.geom_bodyid[geom])
                if body in self.foot_bodies and self.model.geom_type[geom]==mujoco.mjtGeom.mjGEOM_SPHERE:
                    if self.reference_data.geom_xpos[geom,2]-self.model.geom_size[geom,0]<.025:
                        expected_contact.add(body)
        for contact in self.data.contact:
            for geom in (contact.geom1,contact.geom2):
                body=int(self.model.geom_bodyid[geom])
                if body in self.foot_bodies: actual_contact.add(body)
        contact_match=1-len(expected_contact.symmetric_difference(actual_contact))/2
        airborne=len(actual_contact)==0
        support_value,support_hull=self._airborne_support(airborne)
        arm_scale=self._arm_tracking_scale()
        if self.arm_tracking_priority and self.arm_indices:
            core_error=float(np.mean(joint_error[self.non_arm_indices]))
            arm_error=float(np.mean(joint_error[self.arm_indices]))
            position_reward=(1.-self.arm_task_fraction)*np.exp(-5*core_error)+self.arm_task_fraction*arm_scale*np.exp(-5*arm_error)
            core_end=float(np.mean(end_errors[2:])); arm_end=float(np.mean(end_errors[:2]))
            end_reward=(1.-self.arm_task_fraction)*np.exp(-20*core_end)+self.arm_task_fraction*arm_scale*np.exp(-20*arm_end)
        else:
            position_reward=np.exp(-5*position); end_reward=np.exp(-20*end_error)
        task_scale=self.airborne_task_scale if airborne and self.airborne_reward else 1.
        action_scale=self.airborne_action_penalty_scale if airborne and self.airborne_reward else 1.
        reward=float(.45*position_reward+.1*np.exp(-.05*velocity)+.15*task_scale*np.exp(-10*root)
            +.15*task_scale*end_reward+.05*task_scale*contact_match+.1*max(0,upright)
            -action_scale*(.005*np.mean(action**2)+.01*np.mean((action-self.previous)**2)))
        if airborne and support_value is not None and self.airborne_reward:
            quality=float(np.exp(-max(0.,-support_value)/self.support_softness))
            reward+=self.support_reward_weight*quality
            if support_value<0:
                reward-=self.support_penalty_weight*min(1.,-support_value/self.support_softness)
        finite=bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        height=float(self.data.qpos[2])
        reason='non_finite' if not finite else 'root_height' if height<.35 else 'pelvis_tilt' if upright<.35 else None
        failed=reason is not None
        if failed: reward-=1.
        truncated=bool(self.phase>=self.reference.duration or self.elapsed>=self.max_episode_seconds)
        self.previous=action.copy()
        info={'joint_mse':position,'root_error_m':float(np.sqrt(root)),'fallen':failed,'phase':self.phase,'end_effector_mse':end_error,'contact_match':contact_match,
            'termination_reason':reason,'root_height_m':height,'pelvis_up_z':upright,
            'airborne':bool(airborne),'foot_contacts':len(actual_contact),
            'support_margin_m':support_value,'com_in_support_hull':(bool(support_value>=0) if support_value is not None else None),
            'support_hull_vertices':(int(len(support_hull)) if support_hull is not None else 0),
            'airborne_task_scale':float(task_scale),'arm_tracking_scale':float(arm_scale),
            'reference_root_mode':self.reference.root_mode,
            'reference_contact_left':(bool(stored_contact[0]) if stored_contact is not None else None),
            'reference_contact_right':(bool(stored_contact[1]) if stored_contact is not None else None),
            'root_xy_error_m':float(np.linalg.norm(self.data.qpos[:2]-q[:2])),
            'root_speed_mps':float(np.linalg.norm(self.data.qvel[:3])),
            'max_episode_seconds':self.max_episode_seconds,
            'torque_saturation_fraction':float(np.mean(np.abs(self.data.ctrl)>=.99*np.max(np.abs(self.model.actuator_ctrlrange),axis=1)))}
        return self.observation(),reward,failed,truncated,info
