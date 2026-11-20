"""训练技能库：只登记通过独立物理评估的策略。"""
from pathlib import Path
import json
import os
import uuid
from datetime import datetime
from runtime_io import atomic_write
from data_paths import SKILLS, rebase_paths

DB=SKILLS/'registry.json'


def _read():
    if not DB.exists(): return {'version':1,'skills':{}}
    try: data=json.loads(DB.read_text(encoding='utf-8'))
    except ValueError as exc: raise ValueError('技能库JSON损坏，拒绝覆盖，请先修复registry.json') from exc
    if not isinstance(data,dict) or not isinstance(data.get('skills'),dict): raise ValueError('技能库结构非法，拒绝覆盖')
    return rebase_paths(data)


def _write(data):
    atomic_write(DB,lambda path:path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8'))


def register(name,job_dir,report):
    required=('reference_completed','fallen','max_root_xy_error_m','max_root_speed_mps','joint_mse')
    missing=[k for k in required if k not in report]
    if missing: raise ValueError('评估缺少硬指标，不能登记技能：'+','.join(missing))
    if (not report.get('reference_completed') or report.get('fallen') or report['max_root_xy_error_m']>.15
            or report['max_root_speed_mps']>1.0 or report['joint_mse']>.05):
        raise ValueError('评估未达到技能门槛：完成、未摔倒、漂移≤0.15m、速度≤1m/s、关节MSE≤0.05')
    job_dir=Path(job_dir).resolve(); checkpoint=job_dir/'training/policy.pt'; reference=job_dir/'reference.npz'
    if not checkpoint.is_file() or not reference.is_file(): raise FileNotFoundError('技能缺少policy.pt或reference.npz')
    skill_id='skill_'+uuid.uuid4().hex[:10]
    try: source=json.loads((job_dir/'source.json').read_text(encoding='utf-8'))
    except (OSError,ValueError): source={}
    try: config=json.loads((job_dir/'config.json').read_text(encoding='utf-8'))
    except (OSError,ValueError): config={}
    motion_gain=None; motion={}; joint_assist=False; joint_assist_gain=None; airborne_reward=True
    try:
        import torch
        payload=__import__('runtime_io').load_checkpoint(checkpoint,map_location='cpu',weights_only=True)
        controller_config=payload.get('video_controller',{})
        arm_weight=payload.get('arm_weight',1.)
        leg_weight=payload.get('leg_weight',config.get('leg_weight'))
        if payload.get('motion_gain') is not None:
            motion_gain=payload['motion_gain']; motion=payload.get('motion') or {}
        joint_assist=bool(payload.get('joint_assist',False))
        joint_assist_gain=payload.get('joint_assist_gain')
        airborne_reward=bool(payload.get('airborne_reward',True))
    except Exception: leg_weight=config.get('leg_weight'); arm_weight=config.get('arm_weight',1.); controller_config={}
    data=_read(); data['skills'][skill_id]={'skill_id':skill_id,'name':str(name or job_dir.name),'created_at':datetime.now().isoformat(timespec='seconds'),
        'job_dir':str(job_dir),'checkpoint':str(checkpoint),'reference':str(reference),'evaluation':report,
        'video_controller':controller_config,'algorithm':'balanced_ppo','safe_to_run':True,'validation_scope':'single_reference_physical_rollout',
        'leg_weight':leg_weight,'arm_weight':arm_weight,'motion_gain':motion_gain,'motion':motion,'source':source,'action':config.get('action'),
        'joint_assist':joint_assist,'joint_assist_gain':joint_assist_gain,'airborne_reward':airborne_reward,
        'semantic_match_verified':False,'limitation':'物理评估通过不代表学会指定动作；就地根部、腿部可能缩幅，来源见source'}
    _write(data); return data['skills'][skill_id]


def list_skills(): return list(_read()['skills'].values())


def get(skill_id):
    item=_read()['skills'].get(skill_id)
    if not item: raise ValueError('未知技能：'+str(skill_id))
    return item


def remove(skill_id):
    data=_read(); item=data['skills'].pop(skill_id,None)
    if item is None: raise ValueError('未知技能：'+str(skill_id))
    _write(data); return item
