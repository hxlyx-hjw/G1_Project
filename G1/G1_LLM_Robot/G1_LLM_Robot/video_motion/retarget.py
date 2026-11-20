"""人体局部坐标 → G1 肩肘腕 IK；腿部继续使用原 RL 策略。"""
import time
import numpy as np
import mujoco
from scipy.optimize import least_squares
from .pose import load


def unit(v):
    n = np.linalg.norm(v)
    if n < 1e-6:
        raise ValueError('退化的人体骨架')
    return v / n


def arm_directions(points, side):
    left = unit(points[11] - points[12])
    up = unit((points[11] + points[12] - points[23] - points[24]) / 2)
    # G1 使用右手坐标：+X 前、+Y 左、+Z 上，因此 前 = 左 × 上。
    # 不可使用 上 × 左：那会产生左手基底，将动作前后镜像。
    forward = unit(np.cross(left, up))
    up = unit(np.cross(forward, left))
    basis = np.column_stack((forward, left, up))
    s, e, w = (11, 13, 15) if side == 'left' else (12, 14, 16)
    return basis.T @ unit(points[e] - points[s]), basis.T @ unit(points[w] - points[e])


class Retargeter:
    def __init__(self, controller):
        self.model = controller.model
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = controller.data.qpos
        self.names = [f'{side}_{j}' for side in ('left', 'right') for j in
                      ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow')]
        self.addresses = [self.model.joint(n + '_joint').qposadr[0] for n in self.names]
        self.bounds = np.array([controller.pose_limits()[n] for n in self.names])
        self.previous = np.clip([controller.joint(n) for n in self.names], *self.bounds.T)
        self.torso = self.model.body('torso_link').id

    def solve(self, points, vis):
        q = self.previous.copy()
        for offset, side in ((0, 'left'), (4, 'right')):
            indices = [11,12,23,24] + ([13,15] if side == 'left' else [14,16])
            if np.min(vis[indices]) < .5:
                continue
            try:
                upper, lower = arm_directions(points, side)
            except ValueError:
                continue
            shoulder = self.model.body(side + '_shoulder_pitch_link').id
            elbow = self.model.body(side + '_elbow_link').id
            wrist = self.model.body(side + '_wrist_yaw_link').id
            previous = self.previous[offset:offset+4]
            def residual(angles):
                self.data.qpos[self.addresses[offset:offset+4]] = angles
                mujoco.mj_forward(self.model, self.data)
                rotation = self.data.xmat[self.torso].reshape(3,3)
                a = rotation.T @ unit(self.data.xpos[elbow] - self.data.xpos[shoulder])
                b = rotation.T @ unit(self.data.xpos[wrist] - self.data.xpos[elbow])
                return np.r_[a-upper, b-lower, .04*(angles-previous)]
            q[offset:offset+4] = least_squares(
                residual, previous, bounds=tuple(self.bounds[offset:offset+4].T),
                loss='soft_l1', f_scale=.05, max_nfev=35).x
        self.previous = q
        return q


def prepare(controller, path):
    times, points, visibility = load(path)
    converter = Retargeter(controller)
    angles = np.array([converter.solve(p,v) for p,v in zip(points,visibility)])
    # 限制目标角变化速度，防止遮挡恢复后跳变。
    for i in range(1,len(angles)):
        limit = 3.5 * (times[i]-times[i-1])
        angles[i] = np.clip(angles[i], angles[i-1]-limit, angles[i-1]+limit)
    if len(angles)>=7:
        from scipy.signal import savgol_filter
        angles=savgol_filter(angles,7,2,axis=0)
    return times, angles, converter.names, visibility


def play(session, path, speed=1.):
    if not np.isfinite(speed) or not .25 <= speed <= 2:
        raise ValueError('speed 必须在 0.25~2 之间')
    # 预处理不推进仿真；执行时直接在锁内步进，避免后台线程跨过视频帧。
    with session._lock:
        if session._active:
            raise RuntimeError('请等待当前机器人动作完成后再播放视频')
        times, angles, names, vis = prepare(session.ctrl, path)
        c = session.ctrl
        initial = c.status()
        origin = np.array([initial['x'], initial['y']])
        start = c.data.time
        initial_angles = np.array([c.joint(n) for n in names])
        wall_start = time.monotonic()
        duration = .6 + times[-1] / speed
        max_drift, max_error, fallen = 0., 0., False
        try:
            session._cmd = [0.,0.,0.]
            while c.data.time-start < duration:
                if session._stop_evt.is_set():
                    raise RuntimeError('机器人会话已关闭')
                if time.monotonic()-wall_start > max(60, duration*5):
                    raise TimeoutError('动作回放超过墙上时间限制')
                elapsed = c.data.time-start
                if elapsed < .6:
                    target = initial_angles + (angles[0]-initial_angles)*elapsed/.6
                else:
                    t = (elapsed-.6)*speed
                    target = np.array([np.interp(t,times,angles[:,j]) for j in range(len(names))])
                for n,v in zip(names,target):
                    c.set_joint(n,v)
                state = c.status()
                delta = origin - np.array([state['x'],state['y']])
                yaw = np.deg2rad(state['yaw'])
                vx,vy = np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]]) @ delta
                c.set_command(float(np.clip(1.5*vx,-.35,.35)),float(np.clip(1.5*vy,-.28,.28)),float(np.clip(-.6*np.deg2rad(state['yaw']-initial['yaw']),-.55,.55)))
                c.step()
                session._maybe_capture()
                if session._viewer is not None and round((c.data.time-start) / c.model.opt.timestep) % 25 == 0:
                    session._viewer.sync()
                max_drift = max(max_drift,float(np.linalg.norm(delta)))
                measured = np.array([c.data.qpos[c.model.joint(n+'_joint').qposadr[0]] for n in names])
                max_error = max(max_error,float(np.max(np.abs(measured-target))))
                if c.status()['fallen']:
                    fallen = True
                    break
                if session.realtime:
                    delay = c.data.time-start-(time.monotonic()-wall_start)
                    if delay > 0:
                        time.sleep(min(delay,.02))
        finally:
            session._cmd = [0.,0.,0.]
            c.stop()
        return {'completed': not fallen, 'fallen': fallen, 'simulation_seconds': float(c.data.time-start),
                'max_drift_m': max_drift, 'max_joint_error_rad': max_error,
                'frames': len(times), 'low_visibility_frames': int(np.sum(np.min(vis[:,[11,12,13,14,15,16,23,24]],axis=1)<.5)),
                'scope': '上半身姿态重定向，腿部由已有RL策略维持平衡'}
