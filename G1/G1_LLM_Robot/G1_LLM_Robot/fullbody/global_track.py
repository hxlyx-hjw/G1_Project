"""单目关键点的足端接触约束全局轨迹估计（WHAM-lite）。

WHAM 需要额外的视频全局运动模型。当前环境没有安装它，因此这里用已有的
MediaPipe 世界关键点做保守恢复：支撑脚在世界中保持不动时，根节点必须向
相反方向移动。这样可以恢复行走/横向移动的水平位移和转向，而不再强制
``in_place``。空中阶段没有可靠的绝对高度观测，根高度保持上一帧，避免凭空
生成弹道；跳跃离地由足端接触标签和姿态表达。
"""
from __future__ import annotations

import numpy as np


def _unit(values):
    values = np.asarray(values, float)
    norm = float(np.linalg.norm(values))
    if norm < 1e-8:
        raise ValueError('退化的人体骨架')
    return values / norm


def body_bases(points, visibility=None):
    """返回每帧人体局部到相机的旋转矩阵，列向量为 前/左/上。"""
    count = len(points)
    bases = np.tile(np.eye(3), (count, 1, 1))
    valid = np.ones(count, dtype=bool)
    last = np.eye(3)
    for index, point in enumerate(points):
        try:
            left = _unit(point[23] - point[24])
            up = _unit((point[11] + point[12] - point[23] - point[24]) / 2)
            forward = _unit(np.cross(left, up))
            up = _unit(np.cross(forward, left))
            last = np.column_stack((forward, left, up))
            bases[index] = last
        except ValueError:
            bases[index] = last
            valid[index] = False
    if visibility is not None:
        needed = np.min(np.asarray(visibility)[:, [11, 12, 23, 24]], axis=1)
        valid &= needed >= 0.35
    return bases, valid


def estimate_root_trajectory(points, visibility, times, robot_leg_length=0.62,
                             robot_height=0.79, max_speed=2.5):
    """估计接触约束下的根部轨迹、足端接触和朝向。

    返回字典：
        root       (T,3) 世界/相机坐标系下的根位置（首帧 xy=0）
        contacts   (T,2) 左/右足是否接触
        yaw        (T,)  相对首帧的朝向角
        scale      人体到 G1 的腿长比例
        valid      (T,)  该帧骨架是否可信
    """
    points = np.asarray(points, float)
    visibility = np.asarray(visibility, float)
    times = np.asarray(times, float)
    count = len(times)
    if points.shape != (count, 33, 3) or visibility.shape != (count, 33):
        raise ValueError('关键点维度必须为 (T,33,3) 和 (T,33)')
    if count < 3 or np.any(np.diff(times) <= 0):
        raise ValueError('时间戳必须递增且至少三帧')

    bases, valid = body_bases(points, visibility)
    reference_index = int(np.flatnonzero(valid)[0]) if valid.any() else 0
    reference = bases[reference_index]
    # 把每帧骨架转换到“首帧机器人坐标”：x 前、y 左、z 上。
    relative = np.einsum('ji,tjk->tik', reference, bases)
    pelvis = (points[:, 23] + points[:, 24]) / 2
    leg_lengths = []
    for frame in range(count):
        if not valid[frame]:
            continue
        leg_lengths.append(np.mean([
            np.linalg.norm(points[frame, 25] - points[frame, 23]) +
            np.linalg.norm(points[frame, 27] - points[frame, 25]),
            np.linalg.norm(points[frame, 26] - points[frame, 24]) +
            np.linalg.norm(points[frame, 28] - points[frame, 26]),
        ]))
    human_leg = float(np.median(leg_lengths)) if leg_lengths else 0.0
    scale = float(np.clip(robot_leg_length / human_leg, 0.35, 1.8)) if human_leg > 0.1 else 1.0

    foot_indices = [27, 28]
    local = np.zeros((count, 2, 3), float)
    foot_valid = np.zeros((count, 2), bool)
    for frame in range(count):
        if not valid[frame]:
            continue
        for side, foot in enumerate(foot_indices):
            local[frame, side] = scale * (bases[frame].T @ (points[frame, foot] - pelvis[frame]))
            foot_valid[frame, side] = visibility[frame, foot] >= 0.35
    local[:, :, 2] = np.minimum(local[:, :, 2], 0.0)

    usable = foot_valid & np.isfinite(local).all(axis=2)
    foot_robot = np.einsum('tij,tfj->tfi', relative, local)
    lowest = np.min(np.where(usable, foot_robot[:, :, 2], np.inf), axis=1)
    finite_lowest = lowest[np.isfinite(lowest)]
    if not len(finite_lowest):
        return None
    extended = float(np.percentile(finite_lowest, 20))
    threshold = extended + max(0.06, 0.12 * float(robot_leg_length))
    contacts = usable & (foot_robot[:, :, 2] <= threshold)

    root = np.zeros((count, 3), float)
    root[:, 2] = float(robot_height)
    contact_frames = np.flatnonzero(contacts.any(axis=1))
    if len(contact_frames):
        first = int(contact_frames[0])
        first_side = int(np.argmin(np.where(contacts[first], foot_robot[first, :, 2], np.inf)))
        raw_z = float(-foot_robot[first, first_side, 2])
        z_offset = float(robot_height) - raw_z
    else:
        z_offset = float(robot_height)
    velocity = np.zeros(2, float)
    for frame in range(1, count):
        dt = max(1e-3, float(times[frame] - times[frame - 1]))
        candidates = []
        for side in range(2):
            if not contacts[frame, side]:
                continue
            previous_world = root[frame - 1] + relative[frame - 1] @ local[frame - 1, side]
            candidates.append(previous_world - relative[frame] @ local[frame, side])
        if candidates:
            target_xy = np.mean(candidates, axis=0)[:2]
        else:
            target_xy = root[frame - 1, :2] + velocity * dt
        delta = target_xy - root[frame - 1, :2]
        distance = float(np.linalg.norm(delta))
        limit = float(max_speed) * dt
        if distance > limit and distance > 0:
            delta *= limit / distance
        root[frame, :2] = root[frame - 1, :2] + delta
        velocity = 0.65 * velocity + 0.35 * delta / dt
        if contacts[frame].any():
            side = int(np.argmin(np.where(contacts[frame], foot_robot[frame, :, 2], np.inf)))
            target_z = z_offset - float(foot_robot[frame, side, 2])
            root[frame, 2] = 0.45 * target_z + 0.55 * root[frame - 1, 2]
        else:
            root[frame, 2] = root[frame - 1, 2]

    forward = relative[:, :2, 0]
    lengths = np.linalg.norm(forward, axis=1)
    yaw = np.zeros(count, float)
    good = lengths > 1e-6
    yaw[good] = np.arctan2(forward[good, 1], forward[good, 0])
    if good.any():
        last_yaw = 0.0
        for frame in range(count):
            if good[frame]:
                yaw[frame] = np.unwrap([last_yaw, yaw[frame]])[-1]
                last_yaw = yaw[frame]
            else:
                yaw[frame] = last_yaw
        if count >= 5 and float(good.mean()) >= .8:
            from scipy.signal import savgol_filter
            window = min(7, count if count % 2 else count - 1)
            if window >= 5:
                yaw = savgol_filter(yaw, window, 2, mode='interp')
        yaw -= yaw[0]
    return {'root': root, 'contacts': contacts, 'yaw': yaw, 'scale': scale,
            'valid': valid, 'contact_threshold': threshold,
            'method': 'mediapipe_foot_contact'}


def yaw_quaternion(yaw):
    """返回绕 z 轴的 (w,x,y,z) 四元数数组。"""
    yaw = np.asarray(yaw, float)
    quat = np.zeros((len(yaw), 4), float)
    quat[:, 0] = np.cos(yaw / 2)
    quat[:, 3] = np.sin(yaw / 2)
    return quat

