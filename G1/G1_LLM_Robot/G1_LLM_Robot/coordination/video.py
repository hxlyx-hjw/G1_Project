"""视频跳跃事件标注：单目图像不能验证真实离地，自动事件仅作为候选。"""
from pathlib import Path
import json
import numpy as np
from runtime_io import atomic_write


def annotate(pose_path,output,mode='off',jump_times=None):
    if mode not in ('off','auto','explicit'): raise ValueError('jump_mode:off/auto/explicit')
    with np.load(pose_path,allow_pickle=False) as z:
        t=z['times'].astype(float); p=z['points'].astype(float); v=z['visibility'].astype(float)
    t=t-t[0]; times=[]; candidates=[]
    if mode=='explicit': times=list(jump_times or [])
    elif mode=='auto':
        def angle(a,b,c):
            u=p[:,a]-p[:,b]; w=p[:,c]-p[:,b]
            return np.arccos(np.clip(np.sum(u*w,axis=1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(w,axis=1)+1e-9),-1,1))
        bend=np.pi-(angle(23,25,27)+angle(24,26,28))/2
        # 先深屈膝，再快速伸展；只生成候选，不能据此证明腾空。
        for i in range(1,len(t)):
            start=max(0,i-5)
            if t[i]<.7 or t[i]>t[-1]-1.6: continue
            if np.min(v[start:i+1][:,[23,24,25,26,27,28]])<.7: continue
            if np.max(bend[start:i])>.6 and bend[i]<.3 and (bend[i]-bend[i-1])/(t[i]-t[i-1])<-.8:
                if not candidates or t[i]-candidates[-1]>2.5: candidates.append(float(t[i]))
        # 自动检测为候选，必须由显式时间或后续模型/用户核实才执行模板。
    for value in times:
        if not np.isfinite(value) or value<.7 or value>t[-1]-1.6: raise ValueError('起跳时刻需保留前0.7秒与后1.6秒，并相对于截取动作起点')
    times=sorted(float(x) for x in times)
    if any(b-a<2.5 for a,b in zip(times,times[1:])): raise ValueError('两次跳跃至少间隔2.5秒')
    result={'version':1,'mode':mode,'jump_times':times,'candidate_times':candidates,'duration':float(t[-1]),
        'time_origin':'trimmed_reference_start','auto_execution':False,
        'limitation':'auto仅输出屈膝-伸展候选，不把原参考的脚底接地标记当作跳跃真值。explicit使用保守跳跃模板，非视频原跳高恢复。'}
    atomic_write(output,lambda path:path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'))
    return result


def apply_motion(env,payload):
    """应用权重中的动作包络；旧权重保持原控制语义。"""
    motion_gain=payload.get('motion_gain')
    motion=payload.get('motion') or {}
    if motion_gain is None and not motion:
        env.configure_motion(legacy=True)
    else:
        # 上一轮checkpoint没有手臂稳定优先字段，保持当时的臂幅以免改变既有回放效果。
        if 'arm_stability_floor' not in motion:
            env.configure_motion(motion_gain=motion_gain,arm_priority=False,
                arm_stability_floor=1.0,arm_stability_power=1.0,**motion)
        else:
            env.configure_motion(motion_gain=motion_gain,**motion)


def restore_controller(env,payload):
    """所有训练和播放路径共享控制配置；旧权重保持原控制语义。"""
    env.video_leg_weight=payload.get('leg_weight',.15)
    env.video_arm_weight=payload.get('arm_weight',1.)
    apply_motion(env,payload)
    config=payload.get('video_controller')
    if config:
        env.configure_video_controller(config)
