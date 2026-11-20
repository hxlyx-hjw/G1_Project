"""基于物理对照评估的手腿权重自检；模型建议只能作为受限候选。"""
import json
import os
import numpy as np

LEG_BOUNDS=(0.,2.)
ARM_BOUNDS=(.25,1.5)


def quality(report):
    # 漂移、失败优先于高平均奖励；误差始终对原始视频参考计算。
    return (100 if report['completed'] and not report['fallen'] else 0)+min(report['steps'],10000)*.001-report['max_root_xy_error_m']*3-report['joint_mse']*5


def model_advice(report,leg,arm,settings):
    from openai import OpenAI
    key=os.environ.get('DEEPSEEK_API_KEY')
    if not key: raise RuntimeError('未提供DeepSeek配置，跳过模型建议')
    client=OpenAI(api_key=key,base_url=os.environ.get('G1_MODEL_BASE_URL','https://api.deepseek.com'),timeout=30,max_retries=0)
    response=client.chat.completions.create(model=os.environ.get('G1_MODEL_NAME','deepseek-flash'),temperature=0,
        extra_body={"thinking": {"type": "disabled"}},
        response_format={'type':'json_object'},messages=[{'role':'system','content':'根据G1物理诊断建议腿部参考权重leg_weight和手臂参考权重arm_weight，输出JSON和reason。只能建议权重，不能宣称训练成功。'},
        {'role':'user','content':json.dumps({'report':report,'leg_weight':leg,'arm_weight':arm,'bounds':{'leg':list(LEG_BOUNDS),'arm':list(ARM_BOUNDS)}},ensure_ascii=False)}])
    data=json.loads(response.choices[0].message.content)
    values=[float(data['leg_weight']),float(data['arm_weight'])]
    if not np.isfinite(values).all(): raise ValueError('模型返回非法权重')
    # 单次变化受限，防止模型建议跳到危险配置。
    return (float(np.clip(values[0],max(LEG_BOUNDS[0],leg-.1),min(LEG_BOUNDS[1],leg+.1))),
            float(np.clip(values[1],max(ARM_BOUNDS[0],arm-.1),min(ARM_BOUNDS[1],arm+.1))),str(data.get('reason',''))[:1000])


def diagnose(policy,env,device,score_fn,allow_model=False):
    original=(env.video_leg_weight,env.video_arm_weight)
    base=score_fn(policy,env,device)
    unstable=base['fallen'] or base['max_root_xy_error_m']>.15 or base['max_root_speed_mps']>1
    candidates=[]
    if unstable:
        candidates=[(max(LEG_BOUNDS[0],original[0]-.05),original[1]),(original[0],max(ARM_BOUNDS[0],original[1]-.1))]
    else:
        if base.get('leg_mse',0)>.01: candidates.append((min(LEG_BOUNDS[1],original[0]+.05),original[1]))
        if base.get('arm_mse',0)>.005: candidates.append((original[0],min(ARM_BOUNDS[1],original[1]+.05)))
    advice=None; error=None
    if allow_model:
        try:
            leg,arm,reason=model_advice(base,*original,{})
            candidates.append((leg,arm)); advice=reason
        except Exception as exc: error=type(exc).__name__+': '+str(exc)[:200]
    best=base; chosen=original; trials=[]
    try:
        for leg,arm in dict.fromkeys(candidates):
            env.video_leg_weight=leg; env.video_arm_weight=arm
            report=score_fn(policy,env,device)
            trials.append({'leg_weight':leg,'arm_weight':arm,'evaluation':report})
            if quality(report)>quality(best)+.005: best=report; chosen=(leg,arm)
    finally:
        env.video_leg_weight,env.video_arm_weight=chosen
    return {'before':{'leg_weight':original[0],'arm_weight':original[1]},'after':{'leg_weight':chosen[0],'arm_weight':chosen[1]},
        'changed':chosen!=original,'reason':'对照完整物理回放，以完成/漂移/原参考误差选择；缩幅不等于全幅学习',
        'baseline':base,'selected_evaluation':best,'trials':trials,'model_advice':advice,'model_error':error}
