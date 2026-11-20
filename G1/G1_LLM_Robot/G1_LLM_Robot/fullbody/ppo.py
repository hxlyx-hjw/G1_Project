"""原生PyTorch PPO：包含GAE、截断bootstrap、裁剪目标和checkpoint。"""
from pathlib import Path
import json
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn.modules.module import _IncompatibleKeys
from runtime_io import save_checkpoint,load_checkpoint
from proactive import format_duration
from .env import ImitationEnv


class JointAssist(nn.Module):
    """29个关节各自的小型残差网络；最后一层零初始化，默认不改变旧策略。"""
    def __init__(self,obs=188,actions=29,hidden=16,residual=.20):
        super().__init__()
        if obs<12+6*actions+2:
            raise ValueError('观测维度不足以容纳每关节局部特征')
        self.obs=obs; self.actions=actions; self.residual=float(residual)
        self.context=nn.Sequential(nn.Linear(obs,32),nn.Tanh())
        self.heads=nn.ModuleList([
            nn.Sequential(nn.Linear(32+6,hidden),nn.Tanh(),nn.Linear(hidden,1))
            for _ in range(actions)])
        for head in self.heads:
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)

    def forward(self,x):
        if x.shape[-1]!=self.obs:
            raise ValueError('关节辅助网络观测维度不匹配')
        n=self.actions; context=self.context(x)
        # 观测布局：12个根/相位量后依次为q、dq、参考q、参考dq、q误差、上一动作。
        local=torch.stack([x[:,12:12+n],x[:,12+n:12+2*n],x[:,12+2*n:12+3*n],
            x[:,12+3*n:12+4*n],x[:,12+4*n:12+5*n],x[:,12+5*n:12+6*n]],dim=-1)
        outputs=[head(torch.cat([context,local[:,i,:]],dim=-1))
                 for i,head in enumerate(self.heads)]
        return self.residual*torch.tanh(torch.cat(outputs,dim=-1))


class ActorCritic(nn.Module):
    def __init__(self,obs=188,actions=29,joint_assist=True,joint_assist_gain=.18,joint_assist_residual=.20):
        super().__init__()
        self.obs_dim=obs; self.action_dim=actions
        self.actor=nn.Sequential(nn.Linear(obs,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,actions))
        self.critic=nn.Sequential(nn.Linear(obs,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1))
        self.log_std=nn.Parameter(torch.full((actions,),-.7))
        self.joint_assist=JointAssist(obs,actions,residual=joint_assist_residual)
        self.joint_assist_enabled=bool(joint_assist)
        self.joint_assist_gain=float(joint_assist_gain)

    def mean(self,x):
        mean=self.actor(x)
        if self.joint_assist_enabled:
            mean=mean+self.joint_assist_gain*self.joint_assist(x)
        return mean

    def distribution(self,x):
        return Normal(self.mean(x),self.log_std.clamp(-4,1).exp())

    def set_joint_assist(self,enabled,gain=None):
        self.joint_assist_enabled=bool(enabled)
        if gain is not None:
            gain=float(gain)
            if not 0<=gain<=2: raise ValueError('joint_assist_gain应在0~2之间')
            self.joint_assist_gain=gain
        return self

    def load_state_dict(self,state_dict,strict=True,assign=False):
        # 旧checkpoint没有joint_assist.*；保留本模块的零初始化结果并继续加载其余权重。
        result=super().load_state_dict(state_dict,strict=False,assign=assign)
        unexpected=[k for k in result.unexpected_keys if not k.startswith('joint_assist.')]
        missing=[k for k in result.missing_keys if not k.startswith('joint_assist.')]
        if strict and (unexpected or missing):
            raise RuntimeError('策略权重不完整：缺少'+str(missing)+'，多余'+str(unexpected))
        return _IncompatibleKeys(missing,unexpected)

    @classmethod
    def from_checkpoint(cls,payload,obs=188,actions=29,eval_mode=True):
        config=payload.get('joint_assist')
        if config is None:
            enabled=False; gain=.18; residual=.20
        elif isinstance(config,dict):
            enabled=bool(config.get('enabled',True)); gain=float(config.get('gain',.18)); residual=float(config.get('residual',.20))
        else:
            enabled=bool(config); gain=float(payload.get('joint_assist_gain',.18)); residual=float(payload.get('joint_assist_residual',.20))
        policy=cls(obs,actions,joint_assist=enabled,joint_assist_gain=gain,joint_assist_residual=residual)
        policy.load_state_dict(payload['policy'])
        if eval_mode: policy.eval()
        return policy


def train(reference,output,iterations=100,steps=256,seed=0,device='cpu',resume=None,joint_assist=True,joint_assist_gain=.18,airborne_reward=True):
    wall_started=time.monotonic()
    if iterations<1 or steps<8:
        raise ValueError('iterations>=1，steps>=8')
    if not 0<=joint_assist_gain<=2: raise ValueError('joint_assist_gain应在0~2之间')
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(2)
    env=ImitationEnv(reference,output/'model',seed)
    env.airborne_reward=bool(airborne_reward)
    policy=ActorCritic(joint_assist=joint_assist,joint_assist_gain=joint_assist_gain).to(device)
    optimizer=torch.optim.Adam(policy.parameters(),lr=3e-4)
    start_iteration=0
    if resume:
        checkpoint=load_checkpoint(resume,map_location=device,weights_only=True)
        if checkpoint['joint_names']!=env.names or checkpoint.get('obs_dim')!=188:
            raise ValueError('checkpoint关节顺序不一致')
        policy=ActorCritic.from_checkpoint(checkpoint,obs=188,actions=29,eval_mode=False).to(device)
        optimizer=torch.optim.Adam(policy.parameters(),lr=3e-4)
        try: optimizer.load_state_dict(checkpoint['optimizer'])
        except (ValueError,KeyError): pass
        start_iteration=checkpoint['iteration']
        airborne_reward=bool(checkpoint.get('airborne_reward',airborne_reward))
        env.airborne_reward=airborne_reward
    obs,_=env.reset(seed=seed)
    logs=[]
    for iteration in range(start_iteration,start_iteration+iterations):
        observations=[]; actions=[]; old_logs=[]; values=[]; rewards=[]; next_values=[]; ends=[]
        episodes=0
        for _ in range(steps):
            x=torch.as_tensor(obs,device=device).unsqueeze(0)
            with torch.no_grad():
                dist=policy.distribution(x); a=dist.sample()
                log_prob=dist.log_prob(a).sum(-1)
                value=policy.critic(x).squeeze(-1)
            # 保存原始高斯动作并用其密度计算PPO比率，环境内执行限幅。
            next_obs,reward,terminated,truncated,info=env.step(a[0].cpu().numpy())
            with torch.no_grad():
                next_value=0. if terminated else policy.critic(torch.as_tensor(next_obs,device=device).unsqueeze(0)).item()
            observations.append(obs); actions.append(a[0].cpu().numpy()); old_logs.append(log_prob.item()); values.append(value.item())
            rewards.append(reward); next_values.append(next_value); ends.append(terminated or truncated)
            if terminated or truncated:
                next_obs,_=env.reset(); episodes+=1
            obs=next_obs
        advantages=np.zeros(steps,np.float32); gae=0.
        for t in reversed(range(steps)):
            delta=rewards[t]+.99*next_values[t]-values[t]
            gae=delta+.99*.95*(not ends[t])*gae
            advantages[t]=gae
        returns=advantages+np.array(values,np.float32)
        advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
        X=torch.as_tensor(np.array(observations),dtype=torch.float32,device=device)
        A=torch.as_tensor(np.array(actions),dtype=torch.float32,device=device)
        L=torch.tensor(old_logs,dtype=torch.float32,device=device)
        G=torch.tensor(advantages,device=device); R=torch.tensor(returns,device=device)
        for _ in range(4):
            for indices in torch.randperm(steps,device=device).split(64):
                dist=policy.distribution(X[indices]); ratio=(dist.log_prob(A[indices]).sum(-1)-L[indices]).exp()
                actor=-torch.minimum(ratio*G[indices],ratio.clamp(.8,1.2)*G[indices]).mean()
                critic=(policy.critic(X[indices]).squeeze(-1)-R[indices]).square().mean()
                loss=actor+.5*critic-.001*dist.entropy().sum(-1).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('训练损失非有限数值')
                optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(),.5); optimizer.step()
        log={'iteration':iteration+1,'mean_reward':float(np.mean(rewards)),'episodes':episodes,'loss':float(loss.detach()),'last_joint_mse':info['joint_mse']}
        logs.append(log); print(json.dumps(log),flush=True)
        checkpoint={'policy':policy.state_dict(),'optimizer':optimizer.state_dict(),'iteration':iteration+1,
            'joint_names':env.names,'reference':str(Path(reference).resolve()),'seed':seed,'obs_dim':188,'action_dim':29,
            'joint_assist':bool(policy.joint_assist_enabled),'joint_assist_gain':policy.joint_assist_gain,
            'joint_assist_residual':policy.joint_assist.residual,'airborne_reward':bool(airborne_reward)}
        save_checkpoint(checkpoint,output/'policy.pt')
        with (output/'metrics.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(log)+'\n')
    env.close()
    duration=time.monotonic()-wall_started
    print(json.dumps({'event':'training_finished','duration_seconds':round(duration,3),
        'duration_text':format_duration(duration)},ensure_ascii=False),flush=True)
    return logs


def evaluate(reference,checkpoint,output,viewer=False,kinematic=False):
    import time
    import mujoco
    from contextlib import nullcontext
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    if checkpoint and not kinematic:
        header=load_checkpoint(checkpoint,map_location='cpu',weights_only=True)
        if header.get('algorithm')=='balanced_ppo':
            from .balanced import BalancedEnv
            env=BalancedEnv(reference,output/'model',max_episode_seconds=header.get('max_episode_seconds'))
            env.video_leg_weight=header.get('leg_weight',.15)
            env.video_arm_weight=header.get('arm_weight',1.)
            env.airborne_reward=bool(header.get('airborne_reward',True))
            from coordination.video import restore_controller
            restore_controller(env,header)
        else: env=ImitationEnv(reference,output/'model')
    else: env=ImitationEnv(reference,output/'model')
    if not kinematic and checkpoint is not None and not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    policy=None
    if not kinematic and checkpoint:
        data=load_checkpoint(checkpoint,map_location='cpu',weights_only=True)
        if data['joint_names']!=env.names or data.get('obs_dim')!=188: raise ValueError('checkpoint与环境关节不匹配')
        policy=ActorCritic.from_checkpoint(data,eval_mode=True)
    if viewer:
        from viewer_camera import passive_viewer
        handle=passive_viewer(env.model,env.data)
        context=handle if handle is not None else nullcontext(None)
    else:
        context=nullcontext(None)
    obs,_=env.reset(options={'phase':0})
    rewards=[]; errors=[]; fallen=False; trace=[]; info={}; executed=0
    with context as window:
        if window is not None:
            from viewer_camera import apply_isometric_camera
            pelvis_id=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_BODY,'pelvis')
            apply_isometric_camera(
                window.cam, tracking_body=pelvis_id if pelvis_id>=0 else None)
            window.sync()
        for i in range(int(env.reference.duration/.02)+1):
            before=time.monotonic()
            if window is not None and not window.is_running(): break
            executed+=1
            if kinematic:
                q,v=env.reference.sample(i*.02)
                env.data.qpos[:]=q; env.data.qvel[:]=v; mujoco.mj_forward(env.model,env.data)
            else:
                with torch.no_grad():
                    action=policy.mean(torch.as_tensor(obs).unsqueeze(0))[0].numpy() if policy else np.zeros(29)
                obs,reward,terminated,truncated,info=env.step(action)
                trace.append({'step':executed,'simulation_seconds':executed*.02,'reward':reward,**info})
                rewards.append(reward); errors.append(info['joint_mse']); fallen=terminated
                if terminated or truncated:
                    if window: window.sync()
                    break
            if window:
                window.sync(); time.sleep(max(0,.02-(time.monotonic()-before)))
    # 硬指标：skills.register() 依赖这三个字段做门槛判定。
    # 从逐步 trace 中取最大值，缺失时回退到 final_state，再缺失则用 0 占位并标注。
    def _max_of(key):
        values=[t[key] for t in trace if isinstance(t.get(key),(int,float))]
        if values: return float(max(values))
        fallback=info.get(key)
        return float(fallback) if isinstance(fallback,(int,float)) else None
    max_root_xy_error_m=_max_of('root_xy_error_m')
    max_root_speed_mps=_max_of('root_speed_mps')
    joint_mse=float(np.mean(errors)) if errors else (info.get('joint_mse') if isinstance(info.get('joint_mse'),(int,float)) else None)
    airborne_steps=sum(1 for item in trace if item.get('airborne'))
    support_values=[float(item['support_margin_m']) for item in trace if isinstance(item.get('support_margin_m'),(int,float))]
    contact_values=[float(item['contact_match']) for item in trace if isinstance(item.get('contact_match'),(int,float))]
    end_errors=[float(item['end_effector_mse']) for item in trace if isinstance(item.get('end_effector_mse'),(int,float))]
    reference_displacement=float(np.linalg.norm(env.reference.qpos[-1,:2]-env.reference.qpos[0,:2]))
    report={'kinematic_only':kinematic,'policy_used':bool(checkpoint and not kinematic),'fallen':fallen,
        'steps':executed,'simulation_seconds':executed*.02,'reference_duration_seconds':env.reference.duration,
        'completion_fraction':min(1.,executed*.02/env.reference.duration),
        'termination_reason':info.get('termination_reason'),
        'final_state':info,
        'mean_reward':float(np.mean(rewards)) if rewards else None,
        'mean_joint_mse':float(np.mean(errors)) if errors else None,
        'max_root_xy_error_m':max_root_xy_error_m,
        'max_root_speed_mps':max_root_speed_mps,
        'joint_mse':joint_mse,
        'joint_assist':bool(policy.joint_assist_enabled) if policy else False,
        'airborne_steps':airborne_steps,
        'min_support_margin_m':(min(support_values) if support_values else None),
        'airborne_support_compliance':(sum(value>=0 for value in support_values)/len(support_values) if support_values else None),
        'airborne_fraction':(airborne_steps/executed if executed else 0.),
        'mean_contact_match':(float(np.mean(contact_values)) if contact_values else None),
        'keypoint_rmse_m':(float(np.sqrt(np.mean(end_errors))) if end_errors else None),
        'reference_root_mode':getattr(env.reference,'root_mode','in_place'),
        'reference_root_displacement_m':reference_displacement,
        'reference_completed':bool(not fallen and executed*.02>=env.reference.duration)}
    if getattr(env,'jump_controller',None):
        report['jump']=env.jump_controller.report()
        report['reference_completed']=report['reference_completed'] and report['jump']['all_scheduled_jumps_landed']
    report['video_controller']=getattr(env,'video_controller_config',{})
    report['motion_gain']=getattr(env,'_last_motion_gain',None)
    report['motion_config']=getattr(env,'motion',None).as_dict() if getattr(env,'motion',None) else None
    (output/'trajectory.json').write_text(json.dumps(trace,indent=2),encoding='utf-8')
    (output/'evaluation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    return report
