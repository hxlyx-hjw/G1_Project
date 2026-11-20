"""SMPL-X 模型加载与前向传播工具。

模型权重路径：D:\\SMPL\\models_smplx_v1_1\\models\\smplx

用法：
    from smplx_utils import load_smplx, forward_joints, G1_KEY_JOINTS
    model = load_smplx()
    joints = forward_joints(model)          # (55, 3) 世界坐标关节
"""
import torch
from smplx import SMPLX

try:
    from video_motion.smplx_fit import model_dir as _discover_model_dir
    from video_motion.smplx_fit import resolve_gender as _resolve_gender
except Exception:  # 作为独立脚本运行时仍保留固定路径回退
    _discover_model_dir = None
    _resolve_gender = None

MODEL_DIR = str(_discover_model_dir() or r"E:\TEST_RECORD\G1_Project\G1\models\SMPL\models_smplx_v1_1\models\smplx")

# SMPL-X v1.1 的 55 个关节命名（标准顺序）
JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'jaw', 'left_eye_smplhf', 'right_eye_smplhf',
    'left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2',
    'left_middle3', 'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1',
    'left_ring2', 'left_ring3', 'left_thumb1', 'left_thumb2', 'left_thumb3',
    'right_index1', 'right_index2', 'right_index3', 'right_middle1', 'right_middle2',
    'right_middle3', 'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_ring1',
    'right_ring2', 'right_ring3', 'right_thumb1', 'right_thumb2', 'right_thumb3',
]

# 重定向到 G1 需要的身体关键关节（名称 -> 索引）
G1_KEY_JOINTS = {
    'pelvis': 0, 'left_hip': 1, 'right_hip': 2, 'left_knee': 4, 'right_knee': 5,
    'left_ankle': 7, 'right_ankle': 8, 'left_shoulder': 16, 'right_shoulder': 17,
    'left_elbow': 18, 'right_elbow': 19, 'left_wrist': 20, 'right_wrist': 21,
}


def load_smplx(gender='neutral', device='cpu'):
    """加载 SMPL-X 模型（默认中性体型，v1.1）。"""
    folder = str(_discover_model_dir() or MODEL_DIR)
    if _resolve_gender is not None:
        gender = _resolve_gender(folder, gender)
    return SMPLX(folder, gender=gender).to(device)


def forward_joints(model, betas=None, body_pose=None, global_orient=None,
                   transl=None, device='cpu'):
    """前向传播，返回 55 个关节的 3D 世界坐标 (55, 3)。

    参数可传 torch 张量（shape: betas [1,10] / body_pose [1,63] /
    global_orient [1,3] / transl [1,3]），不传则用零姿态。
    """
    if body_pose is None:
        body_pose = torch.zeros(1, 63, device=device)
    if global_orient is None:
        global_orient = torch.zeros(1, 3, device=device)
    if betas is None:
        betas = torch.zeros(1, 10, device=device)
    output = model(
        betas=betas,
        global_orient=global_orient,
        body_pose=body_pose,
        transl=transl,
        return_verts=True,
    )
    # joints: (1, 127, 3)，前 55 个是身体关节
    return output.joints[0, :55, :].detach().cpu().numpy()


if __name__ == '__main__':
    model = load_smplx()
    joints = forward_joints(model)
    print('模型加载成功，关节形状:', joints.shape)
    print('关键关节坐标:')
    for name, idx in G1_KEY_JOINTS.items():
        print('  %-16s idx=%2d  (%+.3f, %+.3f, %+.3f)'
              % (name, idx, joints[idx][0], joints[idx][1], joints[idx][2]))
