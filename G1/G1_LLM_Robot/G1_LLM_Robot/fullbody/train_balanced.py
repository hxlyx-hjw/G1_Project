"""批量网络CUDA PPO，多个CPU MuJoCo环境；保存最佳完整片段策略。"""
from pathlib import Path
import json
import argparse
import time
import numpy as np
import torch
from .balanced import BalancedEnv
from .motion import MotionConfig
from .ppo import ActorCritic
from runtime_io import save_checkpoint,load_checkpoint
from proactive import format_duration


def score(policy,env,device):
    obs,_=env.reset(options={'phase':0}); total=0; errors=[]; fell=False
    max_drift=0.; max_speed=0.; leg_errors=[]; arm_errors=[]
    for step in range(int(np.ceil(env.reference.duration/.02))):
        with torch.no_grad(): a=policy.mean(torch.as_tensor(obs,device=device).unsqueeze(0))[0].cpu().numpy()
        obs,r,f,t,info=env.step(a); total+=r; errors.append(info['joint_mse'])
        ref,_=env.reference.sample(env.phase)
        delta=env.data.qpos[env.qa]-ref[env.qa]
        leg_errors.append(float(np.mean(delta[:12]**2)))
        arms=[i for i,n in enumerate(env.names) if any(j in n for j in ('shoulder','elbow','wrist'))]
        arm_errors.append(float(np.mean(delta[arms]**2)))
        max_drift=max(max_drift,info['root_xy_error_m']); max_speed=max(max_speed,info['root_speed_mps'])
        if f or t: fell=f; break
    completed=not fell and env.phase>=env.reference.duration
    jump_report=info.get('jump',{})
    if getattr(env,'jump_controller',None) and not jump_report.get('all_scheduled_jumps_landed'): completed=False
    return {'completed':completed,'fallen':fell,'steps':step+1,'mean_reward':total/(step+1),'joint_mse':float(np.mean(errors)),'max_root_xy_error_m':max_drift,'max_root_speed_mps':max_speed,'leg_mse':float(np.mean(leg_errors)),'arm_mse':float(np.mean(arm_errors)),'joint_assist':bool(getattr(policy,'joint_assist_enabled',False)),'video_controller':env.video_controller_config,'jump':info.get('jump'),'final_state':info}


def train(reference,output,iterations=100,num_envs=4,horizon=128,device='cuda',seed=42,leg_weight=.15,resume=None,callback=None,should_stop=None,curriculum_target=None,auto_adjust=True,model_review=False,arm_weight=1.,video_controller=None,motion_gain=1.35,max_episode_seconds=None,joint_assist=True,joint_assist_gain=.18,airborne_reward=True,rl_pd_influence=1.0,hand_rl_pd_range=None,foot_rl_pd_range=None):
    wall_started=time.monotonic()
    if iterations<1 or num_envs<1 or horizon<2: raise ValueError('iterations、num_envs必须为正，horizon至少2')
    if not 0<=leg_weight<=2 or not .25<=arm_weight<=1.5: raise ValueError('手腿权重超出范围：腿0~2，臂0.25~1.5')
    if not 1<=motion_gain<=2: raise ValueError('motion_gain应在1~2之间')
    if not 0<=joint_assist_gain<=2: raise ValueError('joint_assist_gain应在0~2之间')
    if max_episode_seconds is not None and not 1<=max_episode_seconds<=600: raise ValueError('max_episode_seconds应在1~600秒')
    if not 0<=float(rl_pd_influence)<=1: raise ValueError('rl_pd_influence应在0~1之间')
    motion_overrides={'rl_pd_influence':float(rl_pd_influence)}
    if hand_rl_pd_range is not None: motion_overrides['hand_rl_pd_range']=hand_rl_pd_range
    if foot_rl_pd_range is not None: motion_overrides['foot_rl_pd_range']=foot_rl_pd_range
    MotionConfig.from_mapping(motion_overrides)
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1); torch.manual_seed(seed); np.random.seed(seed)
    envs=[BalancedEnv(reference,output/f'model_{i}',seed+i,max_episode_seconds=max_episode_seconds) for i in range(num_envs)]
    evaluation=BalancedEnv(reference,output/'eval_model',seed,max_episode_seconds=max_episode_seconds)
    for e in envs+[evaluation]:
        e.video_leg_weight=leg_weight; e.video_arm_weight=arm_weight
        e.configure_motion(motion_gain=motion_gain,**motion_overrides)
        e.airborne_reward=bool(airborne_reward)
    for e in envs+[evaluation]: e.configure_video_controller(video_controller or {})
    policy=ActorCritic(joint_assist=joint_assist,joint_assist_gain=joint_assist_gain).to(device)
    # 小残差初始化，避免随机网络破坏已经可用的平衡先验。
    torch.nn.init.zeros_(policy.actor[-1].weight); torch.nn.init.zeros_(policy.actor[-1].bias)
    policy.log_std.data.fill_(-1.7)
    optimizer=torch.optim.Adam(policy.parameters(),lr=1e-4)
    start_iteration=0
    if resume:
        state=load_checkpoint(resume,map_location=device,weights_only=True)
        if state.get('obs_dim')!=188 or state.get('action_dim')!=29 or state.get('joint_names')!=envs[0].names or state.get('algorithm')!='balanced_ppo':
            raise ValueError('恢复权重与balanced_ppo环境、观测或关节顺序不兼容')
        policy=ActorCritic.from_checkpoint(state,eval_mode=False).to(device)
        optimizer=torch.optim.Adam(policy.parameters(),lr=1e-4)
        try: optimizer.load_state_dict(state['optimizer'])
        except (ValueError,KeyError): pass
        start_iteration=state['iteration']
        video_controller=state.get('video_controller',video_controller)
        for e in envs+[evaluation]: e.configure_video_controller(video_controller or {})
        leg_weight=float(state.get('leg_weight',leg_weight)); arm_weight=float(state.get('arm_weight',arm_weight))
        saved_motion=state.get('motion_gain')
        for e in envs+[evaluation]:
            e.video_leg_weight=leg_weight; e.video_arm_weight=arm_weight
            e.airborne_reward=bool(state.get('airborne_reward',airborne_reward))
            if saved_motion is None: e.configure_motion(legacy=True)
            else:
                motion_values=dict(state.get('motion') or {})
                if 'arm_stability_floor' not in motion_values:
                    motion_values.update(arm_stability_floor=1.0,arm_stability_power=1.0)
                    e.configure_motion(motion_gain=float(saved_motion),arm_priority=False,**motion_values)
                else:
                    e.configure_motion(motion_gain=float(saved_motion),**motion_values)
        motion_gain=1.0 if saved_motion is None else float(saved_motion)
        airborne_reward=bool(state.get('airborne_reward',airborne_reward))
    observations=np.array([e.reset(seed=seed+i)[0] for i,e in enumerate(envs)])
    best=-np.inf
    if (output/'best.json').is_file():
        old=json.loads((output/'best.json').read_text())
        best=(100 if old['completed'] else 0)+old['steps']*.001+old['mean_reward']
    base=score(policy,evaluation,device)
    (output/'baseline.json').write_text(json.dumps(base,indent=2))
    curriculum=None
    if curriculum_target is not None:
        from .curriculum import LegCurriculum
        if not leg_weight<=curriculum_target<=2: raise ValueError('课程目标应在初始权重与2之间')
        relaxed=motion_gain>1.
        curriculum=LegCurriculum(weight=leg_weight,target=curriculum_target,
            max_root_xy_error=.20 if relaxed else .15,
            max_root_speed=1.15 if relaxed else 1.0,
            max_joint_mse=.07 if relaxed else .05)
    for iteration in range(start_iteration+1,start_iteration+iterations+1):
        if should_stop and should_stop(): break
        xs=[]; actions=[]; probs=[]; vals=[]; rewards=[]; nextvals=[]; dones=[]
        for t in range(horizon):
            if should_stop and should_stop():
                for e in envs+[evaluation]: e.close()
                duration=time.monotonic()-wall_started
                print(json.dumps({'event':'training_stopped','duration_seconds':round(duration,3),
                    'duration_text':format_duration(duration)},ensure_ascii=False),flush=True)
                return {'duration_seconds':round(duration,3),'duration_text':format_duration(duration),
                        'stopped':True}
            x=torch.as_tensor(observations,device=device)
            with torch.no_grad():
                dist=policy.distribution(x); a=dist.sample(); lp=dist.log_prob(a).sum(-1); value=policy.critic(x).squeeze(-1)
            next_obs=[]; rr=[]; ff=[]; ee=[]
            for env,action in zip(envs,a.cpu().numpy()):
                o,r,f,tr,info=env.step(action); next_obs.append(o); rr.append(r); ff.append(f); ee.append(f or tr)
            with torch.no_grad(): nv=policy.critic(torch.as_tensor(np.array(next_obs),device=device)).squeeze(-1).cpu().numpy()
            nv[np.array(ff)]=0
            xs.append(observations.copy()); actions.append(a.cpu().numpy()); probs.append(lp.cpu().numpy()); vals.append(value.cpu().numpy())
            rewards.append(rr); nextvals.append(nv); dones.append(ee)
            for i,e in enumerate(envs):
                if ee[i]: next_obs[i]=e.reset()[0]
            observations=np.array(next_obs)
        values=np.array(vals); rewards=np.array(rewards); nv=np.array(nextvals); done=np.array(dones)
        advantages=np.zeros_like(rewards); gae=np.zeros(num_envs)
        for t in reversed(range(horizon)):
            gae=rewards[t]+.99*nv[t]-values[t]+.99*.95*(1-done[t])*gae; advantages[t]=gae
        returns=advantages+values
        advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
        X=torch.as_tensor(np.array(xs).reshape(-1,188),device=device)
        A=torch.as_tensor(np.array(actions).reshape(-1,29),device=device)
        L=torch.as_tensor(np.array(probs).reshape(-1),device=device)
        G=torch.as_tensor(advantages.reshape(-1),device=device,dtype=torch.float32)
        R=torch.as_tensor(returns.reshape(-1),device=device,dtype=torch.float32)
        n=len(X)
        for epoch in range(4):
            for batch in torch.randperm(n,device=device).split(256):
                dist=policy.distribution(X[batch]); logp=dist.log_prob(A[batch]).sum(-1)
                ratio=(logp-L[batch]).exp()
                loss=-torch.minimum(ratio*G[batch],ratio.clamp(.8,1.2)*G[batch]).mean()+.5*(policy.critic(X[batch]).squeeze(-1)-R[batch]).square().mean()-.0005*dist.entropy().sum(-1).mean()
                if not torch.isfinite(loss): raise ValueError('non-finite loss')
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(),.5); optimizer.step()
        record={'iteration':iteration,'mean_reward':float(rewards.mean()),'steps':iteration*num_envs*horizon,'loss':float(loss.detach().cpu()),
            'joint_assist':bool(policy.joint_assist_enabled),
            'rl_pd_influence':float(envs[0].motion.rl_pd_influence),
            'hand_rl_pd_influence':envs[0].motion.effective_hand_influence(),
            'foot_rl_pd_influence':envs[0].motion.effective_foot_influence()}
        checkpoint={'policy':{k:v.detach().cpu() for k,v in policy.state_dict().items()},'optimizer':optimizer.state_dict(),'iteration':iteration,
            'joint_names':envs[0].names,'reference':str(Path(reference).resolve()),'obs_dim':188,'action_dim':29,'algorithm':'balanced_ppo','leg_weight':leg_weight,'arm_weight':arm_weight,'motion_gain':motion_gain,'motion':envs[0].motion.as_dict(),'max_episode_seconds':envs[0].max_episode_seconds,'video_controller':video_controller or {},'seed':seed,
            'joint_assist':bool(policy.joint_assist_enabled),'joint_assist_gain':policy.joint_assist_gain,'joint_assist_residual':policy.joint_assist.residual,'airborne_reward':bool(airborne_reward)}
        save_checkpoint(checkpoint,output/'latest.pt')
        if iteration==1 or iteration%5==0 or iteration==start_iteration+iterations:
            report=score(policy,evaluation,device); record['evaluation']=report
            # 必须完成片段，再比较奖励；保存的最佳策略不被后期退化覆盖。
            quality=(100 if report['completed'] else 0)+report['steps']*.001+report['mean_reward']
            if quality>best:
                save_checkpoint(checkpoint,output/'policy.pt')
                best=quality
                (output/'best.json').write_text(json.dumps(report,indent=2))
            if auto_adjust and iteration%10==0:
                from .self_check import diagnose
                check=diagnose(policy,evaluation,device,score,allow_model=model_review and iteration%30==0 and (report['fallen'] or report['max_root_xy_error_m']>.15 or report['leg_mse']>.05 or report['arm_mse']>.03))
                record['self_check']=check
                with (output/'self_checks.jsonl').open('a',encoding='utf-8') as log: log.write(json.dumps(check,ensure_ascii=False)+'\n')
                if check['changed']:
                    leg_weight=check['after']['leg_weight']; arm_weight=check['after']['arm_weight']
                    for e in envs+[evaluation]: e.video_leg_weight=leg_weight; e.video_arm_weight=arm_weight
                    observations=np.array([e.reset()[0] for e in envs]); best=-np.inf
                    if curriculum: curriculum.weight=leg_weight; curriculum.passes=0
                    report=check['selected_evaluation']; record['evaluation']=report
                    checkpoint.update(leg_weight=leg_weight,arm_weight=arm_weight)
                    save_checkpoint(checkpoint,output/'latest.pt')
                    quality=(100 if report['completed'] else 0)+report['steps']*.001+report['mean_reward']
                    save_checkpoint(checkpoint,output/'policy.pt')
                    best=quality; (output/'best.json').write_text(json.dumps(report,indent=2))
            if curriculum:
                stage=curriculum.update(report); record['curriculum']=stage
                if stage['promoted']:
                    leg_weight=stage['weight']; best=-np.inf
                    for e in envs+[evaluation]: e.video_leg_weight=leg_weight
                    observations=np.array([e.reset()[0] for e in envs])
        with (output/'metrics.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        if callback: callback(record)
        print(json.dumps(record),flush=True)
    for e in envs+[evaluation]: e.close()
    duration=time.monotonic()-wall_started
    print(json.dumps({'event':'training_finished','duration_seconds':round(duration,3),
        'duration_text':format_duration(duration)},ensure_ascii=False),flush=True)
    return {'duration_seconds':round(duration,3),'duration_text':format_duration(duration)}


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('reference'); p.add_argument('--output',default='runs/self_test1/training'); p.add_argument('--iterations',type=int,default=100); p.add_argument('--envs',type=int,default=4); p.add_argument('--horizon',type=int,default=128); p.add_argument('--device',default='cuda'); p.add_argument('--leg-weight',type=float,default=.15); p.add_argument('--arm-weight',type=float,default=1.); p.add_argument('--motion-gain',type=float,default=1.35); p.add_argument('--curriculum-target',type=float); p.add_argument('--max-episode-seconds',type=float); p.add_argument('--resume')
    p.add_argument('--joint-assist',action=argparse.BooleanOptionalAction,default=True,help='每关节神经网络辅助，默认开启')
    p.add_argument('--joint-assist-gain',type=float,default=.18,help='每关节辅助残差增益，0~2')
    p.add_argument('--airborne-reward',action=argparse.BooleanOptionalAction,default=True,help='腾空支撑凸包与放宽惩罚，默认开启')
    p.add_argument('--rl-pd-influence',type=float,default=1.0,help='RL-PD影响因子0~1')
    p.add_argument('--hand-rl-pd-range',type=float,nargs=2,help='手部收敛范围min max')
    p.add_argument('--foot-rl-pd-range',type=float,nargs=2,help='脚部收敛范围min max')
    a=p.parse_args(); train(a.reference,a.output,a.iterations,a.envs,a.horizon,a.device,leg_weight=a.leg_weight,arm_weight=a.arm_weight,motion_gain=a.motion_gain,curriculum_target=a.curriculum_target,max_episode_seconds=a.max_episode_seconds,resume=a.resume,joint_assist=a.joint_assist,joint_assist_gain=a.joint_assist_gain,airborne_reward=a.airborne_reward,rl_pd_influence=a.rl_pd_influence,hand_rl_pd_range=a.hand_rl_pd_range,foot_rl_pd_range=a.foot_rl_pd_range)
