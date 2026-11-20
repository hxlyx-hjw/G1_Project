"""绝对空间关键点技能库，与节点策略库独立保存但可相互关联。"""
from pathlib import Path
import json
import uuid
import numpy as np
from runtime_io import atomic_write
from data_paths import SKILLS, rebase_paths
from .model import build_model
from .reference import Reference

DB=SKILLS/'spatial_registry.json'
KEYPOINT_BODIES=(
    'pelvis','torso_link',
    'left_shoulder_pitch_link','right_shoulder_pitch_link',
    'left_elbow_link','right_elbow_link',
    'left_wrist_yaw_link','right_wrist_yaw_link',
    'left_hip_pitch_link','right_hip_pitch_link',
    'left_knee_link','right_knee_link',
    'left_ankle_roll_link','right_ankle_roll_link',
)


def _read():
    if not DB.exists(): return {'version':1,'skills':{}}
    try: data=json.loads(DB.read_text(encoding='utf-8'))
    except ValueError as exc: raise ValueError('空间技能库JSON损坏，拒绝覆盖') from exc
    if not isinstance(data,dict) or not isinstance(data.get('skills'),dict): raise ValueError('空间技能库结构非法')
    return rebase_paths(data)


def _write(data):
    atomic_write(DB,lambda path:path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8'))


def capture_reference(reference_path,output_dir):
    """将参考qpos转换成每帧世界绝对关键点坐标。"""
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    model=build_model(output_dir/'spatial_model')
    data=__import__('mujoco').MjData(model)
    ref=Reference(reference_path,model)
    body_ids=[model.body(name).id for name in KEYPOINT_BODIES]
    positions=[]; roots=[]
    for t in ref.times:
        q,v=ref.sample(float(t)); data.qpos[:]=q; data.qvel[:]=v
        __import__('mujoco').mj_forward(model,data)
        positions.append(np.array([data.xpos[i] for i in body_ids],dtype=np.float64))
        roots.append(q[:7].copy())
    path=output_dir/'spatial_points.npz'
    np.savez_compressed(path,times=ref.times,positions=np.asarray(positions),root_qpos=np.asarray(roots),
        keypoint_names=np.asarray(KEYPOINT_BODIES),reference=str(Path(reference_path).resolve()),coordinate_system='world_absolute_g1_meters')
    return path


def register(name,job_dir,report,node_skill=None):
    required=('reference_completed','fallen','max_root_xy_error_m','max_root_speed_mps','joint_mse')
    if any(k not in report for k in required): raise ValueError('评估缺少空间技能硬指标')
    if report['fallen'] or not report['reference_completed'] or report['max_root_xy_error_m']>.15 or report['max_root_speed_mps']>1 or report['joint_mse']>.05:
        raise ValueError('评估未达到空间技能入库门槛')
    job_dir=Path(job_dir).resolve(); reference=job_dir/'reference.npz'
    if not reference.is_file(): raise FileNotFoundError(reference)
    points=capture_reference(reference,job_dir/'spatial')
    skill_id='spatial_'+uuid.uuid4().hex[:10]
    item={'skill_id':skill_id,'name':str(name or job_dir.name),'type':'absolute_spatial','created_at':__import__('datetime').datetime.now().isoformat(timespec='seconds'),
        'job_dir':str(job_dir),'reference':str(reference),'points':str(points),'keypoints':list(KEYPOINT_BODIES),
        'coordinate_system':'world_absolute_g1_meters','evaluation':report,'node_skill_id':node_skill,'safe_to_run':True,
        'limitation':'单目视频空间深度和全局位移可能不准确；空间参考通过物理片段验收不代表语义已核实'}
    data=_read(); data['skills'][skill_id]=item; _write(data); return item


def list_skills(): return list(_read()['skills'].values())

def get(skill_id):
    item=_read()['skills'].get(skill_id)
    if not item: raise ValueError('未知空间技能：'+str(skill_id))
    return item
