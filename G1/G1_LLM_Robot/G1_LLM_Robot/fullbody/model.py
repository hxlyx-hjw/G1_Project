"""隔离生成29关节力矩模型；不改变既有行走模型。"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from g1_walkfree.resources import find_menagerie, _SCENE_TEMPLATE


def build_model(output):
    root_dir = Path(find_menagerie())
    root = ET.parse(root_dir/'g1.xml').getroot()
    root.find('compiler').set('meshdir', (root_dir/'assets').as_posix())
    root.find('option').set('timestep', '.002')
    actuator = root.find('actuator')
    joints = {j.get('name'): j for j in root.findall('.//worldbody//joint')}
    for old in list(actuator):
        name = old.get('joint')
        limit = joints[name].get('actuatorfrcrange', '-50 50')
        new = ET.Element('motor',name=name,joint=name,ctrllimited='true',ctrlrange=limit)
        actuator.remove(old)
        actuator.append(new)
    # 保留足部球接触与地面碰撞，禁用其余网格自碰撞以稳定训练。
    for geom in root.findall('.//worldbody//geom'):
        if geom.get('class') != 'foot':
            geom.set('contype','0')
            geom.set('conaffinity','0')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    ET.ElementTree(root).write(output/'robot.xml',encoding='utf-8')
    (output/'scene.xml').write_text(_SCENE_TEMPLATE % {'model':'robot.xml'},encoding='utf-8')
    model = mujoco.MjModel.from_xml_path(str(output/'scene.xml'))
    if model.nu != 29 or model.nq != 36:
        raise ValueError('需要29自由度G1模型')
    return model


def joint_info(model):
    ids = model.actuator_trnid[:,0].astype(int)
    return [model.joint(int(j)).name for j in ids], model.jnt_qposadr[ids], model.jnt_dofadr[ids]


def nominal(model):
    from g1_walkfree.config import DEFAULT_POSE, POSE_NAMES, DEFAULT_ANGLES, LEG_ACTUATOR_NAMES
    names, addresses, _ = joint_info(model)
    values = dict(zip([n+'_joint' for n in LEG_ACTUATOR_NAMES],DEFAULT_ANGLES))
    values.update(zip([n+'_joint' for n in POSE_NAMES],DEFAULT_POSE))
    q = np.zeros(model.nq)
    q[2],q[3] = .79,1.
    q[addresses] = [values[n] for n in names]
    return q


def gains(names):
    kp = np.array([150 if 'knee' in n else 100 if 'hip' in n else 40 if 'ankle' in n else 80 if 'waist' in n else 35 for n in names],float)
    return kp, 2*np.sqrt(kp)*.2
