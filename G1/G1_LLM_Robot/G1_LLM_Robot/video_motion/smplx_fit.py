"""SMPL-X 关节拟合后端。

使用 D:\\SMPL 提供的 SMPL-X v1.1 模型，把 MediaPipe 世界关键点作为观测，
逐帧优化 SMPL-X 的 global_orient/body_pose/transl，得到运动学一致的 3D
关节。该模块是离线拟合，不是 4D-Humans/WHAM 的端到端图像回归；下游
reference.py 仍用拟合后的关节做 IK，因此不会破坏现有控制链路。
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np

MODEL_DIR_ENV = 'G1_SMPLX_MODEL_DIR'
DEFAULT_MODEL_DIRS = (
    Path(r'E:\TEST_RECORD\G1_Project\G1\models\SMPL\models_smplx_v1_1\models\smplx'),
    Path(r'D:\SMPL'),
    Path(r'D:\SMPL\models_smplx_v1_1\models'),
    Path(r'D:\SMPL\models_smplx_v1_1\models\smplx'),
    Path('models/smplx'),
    Path.home() / '.smplx',
)
GENDERS = ('neutral', 'male', 'female')

# MediaPipe 33 点 -> SMPL-X 127 关节输出。SMPL-X 前 55 个关节见
# smplx.joint_names.JOINT_NAMES；脚趾/脚跟和手指在 55 之后。
MEDIAPIPE_TO_SMPLX = {
    0: 55, 1: 57, 2: 57, 3: 57, 4: 56, 5: 56, 6: 56,
    7: 59, 8: 58, 9: 55, 10: 55,
    11: 16, 12: 17, 13: 18, 14: 19, 15: 20, 16: 21,
    17: 70, 18: 75, 19: 67, 20: 72, 21: 66, 22: 71,
    23: 1, 24: 2, 25: 4, 26: 5, 27: 7, 28: 8,
    29: 62, 30: 65, 31: 60, 32: 63,
}

BODY_MEDIAPIPE = frozenset((11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32))

# (MediaPipe 点, SMPL-X 关节, 权重)
FIT_PAIRS = (
    (23, 1, 2.0), (24, 2, 2.0),
    (25, 4, 2.0), (26, 5, 2.0),
    (27, 7, 2.0), (28, 8, 2.0),
    (29, 62, 1.0), (30, 65, 1.0),
    (31, 60, 1.0), (32, 63, 1.0),
    (11, 16, 2.0), (12, 17, 2.0),
    (13, 18, 2.0), (14, 19, 2.0),
    (15, 20, 2.0), (16, 21, 2.0),
    (0, 55, 1.0),
)


def _unit(values):
    values = np.asarray(values, float)
    norm = float(np.linalg.norm(values))
    if norm < 1e-8:
        raise ValueError('退化的人体骨架')
    return values / norm


def _body_bases(points, visibility):
    count = len(points)
    bases = np.tile(np.eye(3), (count, 1, 1))
    valid = np.zeros(count, dtype=bool)
    last = np.eye(3)
    for index, point in enumerate(points):
        try:
            left = _unit(point[23] - point[24])
            up = _unit((point[11] + point[12] - point[23] - point[24]) / 2)
            forward = _unit(np.cross(left, up))
            up = _unit(np.cross(forward, left))
            last = np.column_stack((forward, left, up))
            bases[index] = last
            needed = [11, 12, 23, 24, 25, 26, 27, 28]
            valid[index] = np.min(visibility[index, needed]) >= .35
        except (ValueError, IndexError):
            bases[index] = last
    return bases, valid


def _folders(base):
    """展开常见 SMPL-X 下载布局，兼容直接传入 ``D:\\SMPL`` 根目录。"""
    base = Path(base)
    if base.is_file():
        if base.name.upper().startswith('SMPLX_'):
            return [base.parent]
        return []
    return [
        base, base / 'smplx', base / 'models' / 'smplx',
        base / 'models_smplx_v1_1' / 'models',
        base / 'models_smplx_v1_1' / 'models' / 'smplx',
    ]


def _available_genders(folder):
    return [gender for gender in GENDERS
            if (Path(folder) / ('SMPLX_%s.npz' % gender.upper())).is_file()]


def model_dir(path=None):
    """返回包含 SMPLX_*.npz 的 smplx 目录；找不到时返回 None。"""
    candidates = []
    for value in (path, os.environ.get(MODEL_DIR_ENV)):
        if value:
            candidates.append(Path(value))
    candidates.extend(DEFAULT_MODEL_DIRS)
    for base in candidates:
        for folder in _folders(base):
            try:
                if folder.is_dir() and _available_genders(folder):
                    return folder.resolve()
            except OSError:
                continue
    return None


def resolve_gender(folder, gender):
    """返回实际可用的性别权重；中性权重缺失时回退到其它已下载权重。"""
    folder = Path(folder)
    requested = str(gender).lower()
    if requested not in GENDERS:
        raise ValueError('SMPL-X gender 必须为 neutral/male/female')
    if (folder / ('SMPLX_%s.npz' % requested.upper())).is_file():
        return requested
    available = _available_genders(folder)
    if not available:
        raise FileNotFoundError('目录中没有 SMPLX_*.npz：%s' % folder)
    if 'neutral' in available:
        return 'neutral'
    return available[0]


def _model_file(folder, gender):
    return Path(folder) / ('SMPLX_%s.npz' % resolve_gender(folder, gender).upper())


def available(path=None):
    return importlib.util.find_spec('smplx') is not None and model_dir(path) is not None


def describe(path=None):
    folder = model_dir(path)
    info = {
        'available': False,
        'package': importlib.util.find_spec('smplx') is not None,
        'model_dir': (str(folder) if folder is not None else None),
        'model_file': None,
        'default_gender': None,
        'genders': [],
        'cuda': False,
    }
    if folder is not None:
        info['genders'] = _available_genders(folder)
        info['default_gender'] = resolve_gender(folder, 'neutral') if info['genders'] else None
        info['model_file'] = (str(_model_file(folder, info['default_gender']))
                              if info['default_gender'] else None)
        info['available'] = bool(info['package'] and info['genders'])
    try:
        import torch
        info['cuda'] = bool(torch.cuda.is_available())
    except Exception:
        pass
    return info


def _resolve_device(device):
    import torch
    value = str(device or 'auto').lower()
    if value == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if value not in ('cpu', 'cuda'):
        raise ValueError('SMPL-X device 必须为 auto/cpu/cuda')
    if value == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('请求了 CUDA，但当前 PyTorch 不可用')
    return torch.device(value)


def _targets(canonical, visibility, frame, scale):
    targets, indices, weights = [], [], []
    for mediapipe, smplx, weight in FIT_PAIRS:
        if visibility[frame, mediapipe] < .5:
            continue
        targets.append(canonical[frame, mediapipe] / scale)
        indices.append(smplx)
        weights.append(weight)
    if min(visibility[frame, 23], visibility[frame, 24]) >= .5:
        targets.append((canonical[frame, 23] + canonical[frame, 24]) / 2 / scale)
        indices.append(0)
        weights.append(2.0)
    if min(visibility[frame, 11], visibility[frame, 12]) >= .5:
        targets.append((canonical[frame, 11] + canonical[frame, 12]) / 2 / scale)
        indices.append(12)
        weights.append(1.0)
    return targets, indices, weights


def fit_sequence(points, visibility, times, model_dir_path=None, gender='neutral',
                 device='auto', iterations=60, betas=None, progress=None):
    """把 MediaPipe 世界关键点拟合成 SMPL-X 关节序列。

    ``progress(frame,total,error)`` 可选；error 单位为米，无效帧为 None。
    返回的 joints 在 SMPL-X 规范坐标系（x 前、y 左、z 上）中，单位为米。
    """
    points = np.asarray(points, float)
    visibility = np.asarray(visibility, float)
    times = np.asarray(times, float)
    count = len(times)
    if points.shape != (count, 33, 3) or visibility.shape != (count, 33):
        raise ValueError('SMPL-X 拟合需要 (T,33,3) 关键点和 (T,33) 可见度')
    if count < 2:
        raise ValueError('SMPL-X 拟合至少需要两帧')
    iterations = int(iterations)
    if not 1 <= iterations <= 500:
        raise ValueError('SMPL-X iterations 应在 1~500')
    folder = model_dir(model_dir_path)
    if folder is None:
        raise FileNotFoundError('未找到 SMPL-X 模型；请设置 G1_SMPLX_MODEL_DIR 或 D:\\SMPL')
    gender = resolve_gender(folder, gender)
    import torch
    import smplx
    from smplx.lbs import batch_rodrigues
    from scipy.spatial.transform import Rotation

    resolved = _resolve_device(device)
    layer = smplx.SMPLXLayer(
        str(folder), model_type='smplx', gender=gender,
        num_betas=10, use_pca=False, flat_hand_mean=True).to(resolved).eval()
    for parameter in layer.parameters():
        parameter.requires_grad_(False)

    bases, valid = _body_bases(points, visibility)
    if not valid.any():
        raise ValueError('没有可用于 SMPL-X 拟合的完整全身帧')
    reference_index = int(np.flatnonzero(valid)[0])
    basis = bases[reference_index]
    pelvis_ref = (points[reference_index, 23] + points[reference_index, 24]) / 2
    canonical = np.einsum('ji,tkj->tki', basis, points - pelvis_ref)

    # SMPL-X 原始坐标 -> 规范坐标（x 前、y 左、z 上）。
    rotation0 = torch.tensor(
        [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]],
        dtype=torch.float32, device=resolved)
    axis_angle0 = torch.tensor(
        Rotation.from_matrix(rotation0.cpu().numpy()).as_rotvec(),
        dtype=torch.float32, device=resolved)
    betas_tensor = torch.zeros(1, 10, dtype=torch.float32, device=resolved)
    if betas is not None:
        values = torch.as_tensor(np.asarray(betas, np.float32).reshape(-1)[:10],
                                 dtype=torch.float32, device=resolved)
        betas_tensor[0, :len(values)] = values
    with torch.no_grad():
        rest = layer(
            betas=betas_tensor,
            global_orient=rotation0.view(1, 1, 3, 3),
            body_pose=torch.eye(3, device=resolved).view(1, 1, 3, 3).expand(1, 21, -1, -1),
            transl=torch.zeros(1, 3, device=resolved),
            return_verts=False).joints[0]
    rest_leg = float((rest[1] - rest[4]).norm() + (rest[4] - rest[7]).norm())
    target_leg = float(np.mean([
        np.linalg.norm(points[reference_index, 25] - points[reference_index, 23]) +
        np.linalg.norm(points[reference_index, 27] - points[reference_index, 25]),
        np.linalg.norm(points[reference_index, 26] - points[reference_index, 24]) +
        np.linalg.norm(points[reference_index, 28] - points[reference_index, 26]),
    ]))
    scale = float(np.clip(target_leg / max(rest_leg, 1e-6), .5, 2.0))

    joints = np.full((count, 127, 3), np.nan, np.float32)
    global_orient = np.full((count, 3), np.nan, np.float32)
    body_pose = np.full((count, 63), np.nan, np.float32)
    transl = np.full((count, 3), np.nan, np.float32)
    errors = np.full(count, np.nan, np.float32)
    previous_orient = axis_angle0.clone()
    previous_pose = torch.zeros(63, dtype=torch.float32, device=resolved)
    previous_transl = torch.zeros(3, dtype=torch.float32, device=resolved)

    def evaluate(go, bp, tr):
        orient = batch_rodrigues(go.view(-1, 3)).view(1, 1, 3, 3)
        pose = batch_rodrigues(bp.view(-1, 3)).view(1, 21, 3, 3)
        return layer(betas=betas_tensor, global_orient=orient, body_pose=pose,
                     transl=tr.view(1, 3), return_verts=False).joints[0]

    with torch.enable_grad():
        for frame in range(count):
            targets, indices, weights = _targets(canonical, visibility, frame, scale)
            if len(targets) < 6:
                if progress:
                    progress(frame + 1, count, None)
                continue
            target = torch.as_tensor(np.asarray(targets, np.float32), device=resolved)
            weight = torch.as_tensor(np.asarray(weights, np.float32), device=resolved)
            orient = previous_orient.clone().requires_grad_(True)
            pose = previous_pose.clone().requires_grad_(True)
            translation = previous_transl.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([orient, pose, translation], lr=.03)
            steps = iterations + (30 if frame == reference_index else 0)
            for _ in range(steps):
                optimizer.zero_grad()
                output = evaluate(orient, pose, translation)
                loss = ((output[indices] - target) ** 2).sum(-1).mul(weight).mean()
                loss = loss + 2e-4 * (pose ** 2).mean()
                loss = loss + 3e-3 * ((pose - previous_pose) ** 2).mean()
                loss = loss + 3e-3 * ((orient - previous_orient) ** 2).mean()
                loss = loss + 3e-3 * ((translation - previous_transl) ** 2).mean()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                output = evaluate(orient, pose, translation)
                residual = (output[indices] - target).pow(2).sum(-1).mul(weight)
                errors[frame] = scale * float(residual.mean().sqrt())
            joints[frame] = output.detach().cpu().numpy()
            global_orient[frame] = orient.detach().cpu().numpy()
            body_pose[frame] = pose.detach().cpu().numpy()
            transl[frame] = translation.detach().cpu().numpy()
            previous_orient = orient.detach()
            previous_pose = pose.detach()
            previous_transl = translation.detach()
            if progress:
                progress(frame + 1, count, float(errors[frame]))

    return {
        'joints': joints,
        'global_orient': global_orient,
        'body_pose': body_pose,
        'transl': transl,
        'betas': betas_tensor.detach().cpu().numpy().reshape(-1),
        'scale': scale,
        'basis': basis,
        'pelvis_ref': pelvis_ref,
        'errors': errors,
        'valid': valid,
        'model_dir': str(folder),
        'gender': gender,
        'device': str(resolved),
        'iterations': iterations,
    }


def map_joints_to_mediapipe(fit, source_visibility=None):
    """把 SMPL-X 关节映射回原相机坐标的 MediaPipe 33 点布局。"""
    joints = np.asarray(fit['joints'], float)
    scale = float(fit['scale'])
    basis = np.asarray(fit['basis'], float)
    pelvis = np.asarray(fit['pelvis_ref'], float)
    errors = np.asarray(fit['errors'], float)
    valid = np.asarray(fit['valid'], bool)
    count = len(joints)
    points = np.zeros((count, 33, 3), np.float32)
    for mediapipe, smplx in MEDIAPIPE_TO_SMPLX.items():
        points[:, mediapipe] = pelvis + (scale * joints[:, smplx]) @ basis.T
    points[~valid] = 0.
    visibility = np.zeros((count, 33), np.float32)
    if source_visibility is not None:
        source = np.asarray(source_visibility, float)
        if source.shape == visibility.shape:
            visibility[:] = source
    confidence = np.where(valid & np.isfinite(errors),
                          np.clip(1.2 * np.exp(-errors / .10), .5, 1.0), 0.)
    for mediapipe in BODY_MEDIAPIPE:
        visibility[:, mediapipe] = np.maximum(visibility[:, mediapipe], confidence)
    visibility[~valid] = 0.
    return points, visibility.astype(np.float32)
