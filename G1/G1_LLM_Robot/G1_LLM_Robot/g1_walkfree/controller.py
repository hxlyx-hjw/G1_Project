"""核心控制器：RL 策略驱动腿部 + 位置伺服驱动腰/手臂。

控制链路
--------
::

    每 1 步 (500Hz)   腿部: tau = kps*(q_target - q) - kds*dq  -> <motor>
                      腰臂: data.ctrl[12:29] = 目标角          -> <position>
    每 10 步 (50Hz)   组装 47 维观测 -> 策略 -> 12 维动作 -> 新的 q_target

其中 ``q_target = action * ACTION_SCALE + DEFAULT_ANGLES``，
也就是策略输出的是**相对默认站姿的偏移**。

"自由可控"体现在三个输入上
--------------------------
* :meth:`set_command`  速度指令 (vx, vy, vyaw) —— 控制往哪走、走多快、怎么转
* :meth:`set_joint`    腰/手臂 17 个关节目标角 —— 控制上半身姿态
* :meth:`reset`        摔倒后一键复位

平衡不需要我们管：策略本身就是为了"不摔"训练的。
"""

from __future__ import annotations

import os
import tempfile

import mujoco
import numpy as np

from . import config as C
from . import resources


class WalkController:
    """G1 行走控制器。构造一次，之后循环调用 :meth:`step`。"""

    # ------------------------------------------------------------------
    # 构造 / 复位
    # ------------------------------------------------------------------
    def __init__(self,
                 menagerie: str | None = None,
                 policy: str | None = None,
                 out_dir: str | None = None,
                 torso_pitch: float = 0.0,
                 verbose: bool = False):
        """加载模型与策略。

        参数
            menagerie    手动指定 ``unitree_g1`` 目录（None = 自动查找）
            policy       手动指定 ``motion.pt``（None = 自动查找）
            out_dir      改造后模型的输出目录（None = 系统临时目录）
            torso_pitch  腰部俯仰的初始偏置（rad），0 为默认站姿
            verbose      是否打印加载信息
        """
        # ---- 1) 定位资源 ----
        self.menagerie_dir = resources.find_menagerie(menagerie)
        self.policy_path = resources.find_policy(policy)
        # 改造后的模型默认生成到系统临时目录（不污染用户的 menagerie 安装）
        self.out_dir = out_dir or os.path.join(tempfile.gettempdir(), C.TMP_SUBDIR)

        # ---- 2) 生成并加载混合执行器模型 ----
        self.scene_path = resources.build_walkarm_model(self.menagerie_dir, self.out_dir)
        self.model = mujoco.MjModel.from_xml_path(self.scene_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = C.SIM_DT
        if self.model.nu != len(C.POSE_NAMES) + C.NUM_ACTIONS:
            raise RuntimeError(
                "执行器数量异常：期望 %d 个（12 腿 + 17 腰臂），实际 %d 个。"
                % (len(C.POSE_NAMES) + C.NUM_ACTIONS, self.model.nu))

        # ---- 3) 载入 RL 策略（TorchScript）----
        # 延迟 import：只做 headless 单测时能更早暴露 torch 缺失
        import torch
        self._torch = torch
        self.policy = torch.jit.load(self.policy_path, map_location="cpu")
        self.policy.eval()

        # ---- 4) 上半身姿态目标（actuator 12..28）----
        self.pose = C.DEFAULT_POSE.copy()
        self.pose[2] = float(torso_pitch)          # 腰部俯仰

        # ---- 5) 速度指令 ----
        self.cmd = np.zeros(3, dtype=np.float64)   # (vx, vy, vyaw)

        # ---- 6) 复位到初始状态 ----
        self._reset_state()

        if verbose:
            print("menagerie : %s" % self.menagerie_dir)
            print("policy    : %s" % self.policy_path)
            print("scene     : %s" % self.scene_path)

    def _reset_state(self) -> None:
        """把仿真状态恢复到"站立、腿在默认站姿、速度为零"。"""
        d, m = self.data, self.model
        mujoco.mj_resetData(m, d)
        d.qpos[:] = 0.0
        d.qpos[2] = C.RESET_HEIGHT                 # 骨盆离地高度
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]         # 姿态四元数 = 单位元
        d.qpos[7:7 + C.NUM_ACTIONS] = C.DEFAULT_ANGLES
        d.ctrl[:C.NUM_ACTIONS] = 0.0
        d.ctrl[C.NUM_ACTIONS:] = self.pose
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

        # 策略相关的运行时缓存
        self._action = np.zeros(C.NUM_ACTIONS, dtype=np.float32)
        self._target = C.DEFAULT_ANGLES.copy()
        self._obs = np.zeros(C.NUM_OBS, dtype=np.float32)
        self._counter = 0
        self._start_xy = d.qpos[:2].copy()

    def reset(self) -> None:
        """摔倒后复位。注意：速度指令会保留，便于连续测试。"""
        self._reset_state()

    # ------------------------------------------------------------------
    # 外部输入：速度指令
    # ------------------------------------------------------------------
    def set_command(self, vx: float | None = None,
                    vy: float | None = None,
                    vyaw: float | None = None) -> None:
        """设置速度指令。三项都可以单独更新（None = 保持不变），自动限幅。"""
        if vx is not None:
            self.cmd[0] = float(np.clip(vx, -C.MAX_VX, C.MAX_VX))
        if vy is not None:
            self.cmd[1] = float(np.clip(vy, -C.MAX_VY, C.MAX_VY))
        if vyaw is not None:
            self.cmd[2] = float(np.clip(vyaw, -C.MAX_WZ, C.MAX_WZ))

    def stop(self) -> None:
        """急停：三个速度指令全部归零（机器人会原地站住）。"""
        self.cmd[:] = 0.0

    # ------------------------------------------------------------------
    # 外部输入：腰 / 手臂姿态
    # ------------------------------------------------------------------
    def set_joint(self, name: str, value: float) -> None:
        """设置单个腰/手臂关节的目标角（rad）。名字见 ``config.POSE_NAMES``。"""
        if name in C.POSE_NAMES:
            self.pose[C.POSE_NAMES.index(name)] = float(value)

    def joint(self, name: str) -> float:
        """读取单个腰/手臂关节的当前目标角。"""
        return float(self.pose[C.POSE_NAMES.index(name)]) if name in C.POSE_NAMES else 0.0

    def nudge_joint(self, name: str, delta: float) -> None:
        """在当前目标角基础上增减（键盘微调用），限制在 ±3 rad。"""
        if name in C.POSE_NAMES:
            i = C.POSE_NAMES.index(name)
            self.pose[i] = float(np.clip(self.pose[i] + delta, -3.0, 3.0))

    def set_arm(self, side: str, **kwargs: float) -> None:
        """批量设置某只手臂，例如 ``set_arm("left", shoulder_pitch=1.0)``。"""
        for j, v in kwargs.items():
            self.set_joint("%s_%s" % (side, j), v)

    def apply_arm_preset(self, name: str) -> None:
        """套用 ``config.ARM_PRESETS`` 里的预设（左右肩 roll 自动取反）。"""
        preset = C.ARM_PRESETS[name]
        for side, sign in (("left", 1.0), ("right", -1.0)):
            for j, v in preset.items():
                self.set_joint("%s_%s" % (side, j),
                               v * sign if j == "shoulder_roll" else v)

    # ------------------------------------------------------------------
    # 策略观测
    # ------------------------------------------------------------------
    @staticmethod
    def _gravity_orientation(quat: np.ndarray) -> np.ndarray:
        """由基座四元数算"投影重力方向"（即重力在机体坐标系下的方向）。

        这是策略判断"身体有没有歪"的主要输入，官方公式，不能改。
        """
        qw, qx, qy, qz = quat
        return np.array([2.0 * (-qz * qx + qw * qy),
                         -2.0 * (qz * qy + qw * qx),
                         1.0 - 2.0 * (qw * qw + qz * qz)], dtype=np.float32)

    def _build_obs(self) -> np.ndarray:
        """组装 47 维观测（顺序与官方 deploy_mujoco.py 完全一致）::

            [0:3]   角速度 × ANG_VEL_SCALE
            [3:6]   投影重力方向
            [6:9]   速度指令 × CMD_SCALE
            [9:21]  (关节角 - 默认站姿) × DOF_POS_SCALE
            [21:33] 关节角速度 × DOF_VEL_SCALE
            [33:45] 上一步策略输出的动作
            [45:47] 相位 sin/cos
        """
        d = self.data
        phase = (self._counter * C.SIM_DT) % C.PHASE_PERIOD / C.PHASE_PERIOD
        obs = self._obs
        obs[:3] = d.qvel[3:6] * C.ANG_VEL_SCALE
        obs[3:6] = self._gravity_orientation(d.qpos[3:7])
        obs[6:9] = self.cmd.astype(np.float32) * C.CMD_SCALE
        obs[9:9 + C.NUM_ACTIONS] = (d.qpos[7:7 + C.NUM_ACTIONS]
                                    - C.DEFAULT_ANGLES) * C.DOF_POS_SCALE
        obs[9 + C.NUM_ACTIONS:9 + 2 * C.NUM_ACTIONS] = \
            d.qvel[6:6 + C.NUM_ACTIONS] * C.DOF_VEL_SCALE
        obs[9 + 2 * C.NUM_ACTIONS:9 + 3 * C.NUM_ACTIONS] = self._action
        obs[9 + 3 * C.NUM_ACTIONS:] = [np.sin(2 * np.pi * phase),
                                       np.cos(2 * np.pi * phase)]
        return obs

    # ------------------------------------------------------------------
    # 主循环（每个物理步调用一次）
    # ------------------------------------------------------------------
    def step(self) -> None:
        """推进一步仿真：发力矩 -> 物理步进 -> 到点跑策略。"""
        d, m = self.data, self.model

        # ---- 1) 腿部：策略给的位置目标 -> PD 变成力矩 ----
        tau = ((self._target - d.qpos[7:7 + C.NUM_ACTIONS]) * C.KPS
               + (0.0 - d.qvel[6:6 + C.NUM_ACTIONS]) * C.KDS)
        d.ctrl[:C.NUM_ACTIONS] = tau

        # ---- 2) 腰 / 手臂：直接写位置目标（模型里是 <position> 执行器）----
        d.ctrl[C.NUM_ACTIONS:] = self.pose

        # ---- 3) 物理 ----
        mujoco.mj_step(m, d)

        # ---- 4) 每 CONTROL_DECIMATION 步（50Hz）跑一次策略 ----
        self._counter += 1
        if self._counter % C.CONTROL_DECIMATION == 0:
            obs = self._build_obs()
            with self._torch.inference_mode():          # 关闭梯度，省时间
                out = self.policy(self._torch.from_numpy(obs).unsqueeze(0))
            self._action = out.numpy().squeeze()        # (12,)
            # 动作是"相对默认站姿的偏移"
            self._target = self._action * C.ACTION_SCALE + C.DEFAULT_ANGLES

    # ------------------------------------------------------------------
    # 状态查询（给界面/日志用）
    # ------------------------------------------------------------------
    def status(self) -> dict:
        """返回一帧摘要状态，供界面显示与测试断言使用。"""
        d = self.data
        qw, qx, qy, qz = d.qpos[3:7]

        # 偏航角（绕世界 z 轴），用于显示"当前朝向"
        yaw = float(np.degrees(np.arctan2(2.0 * (qw * qz + qx * qy),
                                          1.0 - 2.0 * (qy * qy + qz * qz))))
        # 骨盆自身 z 轴与世界 z 轴的点积：1=正立，越接近 0 越倒
        upright = float(np.dot(d.xmat[1].reshape(3, 3)[:, 2], [0.0, 0.0, 1.0]))

        vx, vy = float(d.qvel[0]), float(d.qvel[1])
        return dict(
            time=float(d.time),
            x=float(d.qpos[0]), y=float(d.qpos[1]), z=float(d.qpos[2]),
            yaw=yaw,
            vx=vx, vy=vy, speed=float(np.hypot(vx, vy)),
            dx=float(d.qpos[0] - self._start_xy[0]),
            dy=float(d.qpos[1] - self._start_xy[1]),
            cmd=(float(self.cmd[0]), float(self.cmd[1]), float(self.cmd[2])),
            fallen=bool(upright < C.FALLEN_UPRIGHT or d.qpos[2] < C.FALLEN_HEIGHT),
            upright=upright, height=float(d.qpos[2]),
        )

    # ------------------------------------------------------------------
    @property
    def pelvis_id(self) -> int:
        """骨盆 body 的 id（视角跟随用）。"""
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    def pose_limits(self) -> dict[str, tuple[float, float]]:
        """返回腰/手臂各关节的 (下限, 上限)，供界面滑杆设定量程。

        优先用执行器的 ctrlrange（menagerie 里是 ``inheritrange="1"``，
        等于关节 range）；取不到就退回关节自身的 range。
        """
        limits: dict[str, tuple[float, float]] = {}
        for i, name in enumerate(C.POSE_NAMES):
            ai = C.NUM_ACTIONS + i
            lo, hi = (float(self.model.actuator_ctrlrange[ai][0]),
                      float(self.model.actuator_ctrlrange[ai][1]))
            if not (hi > lo):                       # ctrlrange 没设，退回关节 range
                jid = int(self.model.actuator_trnid[ai][0])
                lo, hi = (float(self.model.jnt_range[jid][0]),
                          float(self.model.jnt_range[jid][1]))
            if not (hi > lo):                       # 还是没有（无限位关节），给个安全范围
                lo, hi = -2.6, 2.6
            limits[name] = (lo, hi)
        return limits

