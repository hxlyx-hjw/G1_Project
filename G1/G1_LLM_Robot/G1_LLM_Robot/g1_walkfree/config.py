"""集中存放所有常量与默认参数。

设计原则：**除了这里，其它模块不写"魔法数字"**。
想调策略手感、改姿态、改速度上限，只改这一个文件即可。

参数来源说明
------------
* 策略超参（KPS/KDS/DEFAULT_ANGLES/各 scale）来自官方
  ``unitree_rl_gym/deploy/deploy_mujoco/configs/g1.yaml``，
  必须与训练时保持一致，否则策略输入分布会跑偏。
* 观测维度 47 / 动作维度 12 也是官方定义，不能改。
"""

from __future__ import annotations

import numpy as np

# ======================================================================
# 一、策略与仿真超参（来自官方 g1.yaml，请勿随意改动）
# ======================================================================

#: 仿真步长（秒）。官方 500Hz，策略以 1/50Hz 推理。
SIM_DT: float = 0.002

#: 策略推理间隔：每 N 个仿真步跑一次网络 -> 500/10 = 50Hz
CONTROL_DECIMATION: int = 10

#: 12 个腿部关节的 PD 刚度。顺序 = 左腿(髋p/髋r/髋y/膝/踝p/踝r) + 右腿(同)
KPS = np.array([100, 100, 100, 150, 40, 40,
                100, 100, 100, 150, 40, 40], dtype=np.float64)

#: 12 个腿部关节的 PD 阻尼
KDS = np.array([2, 2, 2, 4, 2, 2,
                2, 2, 2, 4, 2, 2], dtype=np.float64)

#: 训练时使用的"默认站姿"（髋前摆 -0.1 / 膝 0.3 / 踝 -0.2）。
#: 策略输出的动作是**相对该站姿的偏移**，不是绝对角度。
DEFAULT_ANGLES = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                           -0.1, 0.0, 0.0, 0.3, -0.2, 0.0], dtype=np.float64)

# --- 观测归一化系数 -------------------------------------------------
ANG_VEL_SCALE: float = 0.25   # 躯干角速度的前置缩放
DOF_POS_SCALE: float = 1.0    # 关节角偏差缩放
DOF_VEL_SCALE: float = 0.05   # 关节角速度缩放

#: 策略输出的动作 -> 关节位置目标的缩放（0.25 rad 满量程）
ACTION_SCALE: float = 0.25

#: 速度指令的缩放，对应 (vx, vy, vyaw)
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)

#: 观测 / 动作维度（官方定义，不可改）
NUM_ACTIONS: int = 12
NUM_OBS: int = 47

#: 观测里的相位信号周期（秒）。官方用 0.8s 的正弦/余弦对。
PHASE_PERIOD: float = 0.8


# ======================================================================
# 二、模型改造相关
# ======================================================================

#: 需要从 <position> 改成 <motor> 的 12 个腿部执行器名（去掉 _joint 后缀）
LEG_ACTUATOR_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
]

#: 29 自由度模型里，腰 + 手臂共 17 个位置伺服关节（actuator 下标 12..28）。
#: 顺序必须与 MJCF 里 <actuator> 的书写顺序一致。
POSE_NAMES = [
    "waist_yaw", "waist_roll", "waist_pitch",                          # 12,13,14
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",  # 15,16,17
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",   # 18..21
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",     # 22,23,24
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",  # 25..28
]

#: 腰/手臂的默认目标角（= menagerie `stand` 关键帧里 ctrl 的后 17 项）
DEFAULT_POSE = np.array([
    0.0, 0.0, 0.0,                          # 腰
    0.2, 0.2, 0.0, 1.28, 0.0, 0.0, 0.0,     # 左臂
    0.2, -0.2, 0.0, 1.28, 0.0, 0.0, 0.0,    # 右臂
], dtype=np.float64)


# ======================================================================
# 三、手臂预设
# ======================================================================

#: 单侧手臂的 7 个关节名（不含 left_/right_ 前缀）
ARM_JOINTS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw"]

#: 界面上显示用的中文名（关节名本身要留给 MuJoCo，所以另建一张对照表）
ARM_JOINT_LABELS = {
    "shoulder_pitch": "肩 前后摆",
    "shoulder_roll": "肩 外张",
    "shoulder_yaw": "肩 旋转",
    "elbow": "肘 弯曲",
    "wrist_roll": "腕 旋转",
    "wrist_pitch": "腕 俯仰",
    "wrist_yaw": "腕 偏转",
}

#: 腰部 3 个关节（含中文名）
WAIST_JOINTS = ["waist_yaw", "waist_roll", "waist_pitch"]
WAIST_JOINT_LABELS = {
    "waist_yaw": "腰 左右转",
    "waist_roll": "腰 侧倾",
    "waist_pitch": "腰 前后俯仰",
}


#: 一键预设。注意 shoulder_roll 左右手符号相反，应用时会自动取反。
#:
#: ⚠ **肩关节符号约定（容易搞反）**
#:     ``left_shoulder_pitch_joint`` 的轴是 ``[0, 1, 0]``，
#:     实测（骨盆为原点，x 正 = 身体前方 = 行走前进方向）::
#:
#:         shoulder_pitch = -2.0  ->  手在 前上方 (x=+0.10, z=+0.55)
#:         shoulder_pitch = -0.8  ->  手在 正前方 (x=+0.28, z=+0.29)  <-- 前平举
#:         shoulder_pitch = +0.5  ->  手在 正下方 (x=+0.08, z=+0.03)
#:         shoulder_pitch = +1.5  ->  手在 后方   (x=-0.19, z=+0.08)  <-- 手甩到背后
#:
#:     **负值 = 手臂向前，正值 = 手臂向后**（与直觉相反）。
#:     下面的数值都是用数值 IK 反解 + 侧视渲染确认过的，不要再凭感觉改符号。
#:
#: 提醒：手臂摆得越夸张，行走速度越会被质心偏移带偏（策略按固定手臂姿态训练）。
ARM_PRESETS: dict[str, dict[str, float]] = {
    # 手在身侧、略前、髋部高度
    "自然下垂（推荐）": dict(shoulder_pitch=0.05, shoulder_roll=0.15, shoulder_yaw=0.0,
                      elbow=1.40, wrist_roll=0.0, wrist_pitch=0.0, wrist_yaw=0.0),
    # 手在正前方、与肩同高（前进方向）
    "双手前平举": dict(shoulder_pitch=-0.95, shoulder_roll=0.30, shoulder_yaw=0.0,
                  elbow=0.35, wrist_roll=0.0, wrist_pitch=0.0, wrist_yaw=0.0),
    # 手举到头前上方
    "双手上举": dict(shoulder_pitch=-2.00, shoulder_roll=0.45, shoulder_yaw=0.0,
                 elbow=0.20, wrist_roll=0.0, wrist_pitch=0.0, wrist_yaw=0.0),
    # 手向体侧平伸
    "张开双臂": dict(shoulder_pitch=0.25, shoulder_roll=1.55, shoulder_yaw=0.0,
                 elbow=0.45, wrist_roll=0.0, wrist_pitch=0.0, wrist_yaw=0.0),
    # 屈肘把手收到胸前
    "抱胸": dict(shoulder_pitch=-0.55, shoulder_roll=0.35, shoulder_yaw=0.0,
               elbow=1.45, wrist_roll=0.0, wrist_pitch=0.0, wrist_yaw=0.0),
}


# ======================================================================
# 四、速度指令上限（保护用，防止指令过大导致摔倒）
# ======================================================================
MAX_VX: float = 1.2      # 前后 (m/s)
MAX_VY: float = 0.8      # 左右 (m/s)
MAX_WZ: float = 1.2      # 转向 (rad/s)

#: 键盘每按一次的速度增量
KEY_STEP_VX: float = 0.10
KEY_STEP_VY: float = 0.10
KEY_STEP_WZ: float = 0.20

#: 手臂关节用键盘微调的增量（rad）
KEY_STEP_ARM: float = 0.05


# ======================================================================
# 五、复位与摔倒判定
# ======================================================================
RESET_HEIGHT: float = 0.80      # 复位时骨盆离地高度 (m)
FALLEN_HEIGHT: float = 0.30     # 骨盆低于此高度视为摔倒 (m)
FALLEN_UPRIGHT: float = 0.60    # 骨盆 z 轴与世界 z 轴点积低于此值视为摔倒


# ======================================================================
# 六、资源定位候选路径
# ======================================================================
#: menagerie 里 G1 目录相对某个根目录的名字
MENAGERIE_SUBDIR = "unitree_g1"
#: 相对用户主目录的候选 menagerie 位置
MENAGERIE_CANDIDATES = [
    r"E:\TEST_RECORD\G1_Project\G1\assets\mujoco_menagerie",
    "mujoco_menagerie",
    "Documents/mujoco_menagerie",
    "repos/mujoco_menagerie",
    "code/mujoco_menagerie",
]
#: 相对用户主目录的候选策略权重位置
POLICY_CANDIDATES = [
    r"E:\TEST_RECORD\G1_Project\G1\assets\motion.pt",
    "g1-mujoco-stand/vendor/unitree_rl_gym/deploy/pre_train/g1/motion.pt",
    "unitree_rl_gym/deploy/pre_train/g1/motion.pt",
    "vendor/unitree_rl_gym/deploy/pre_train/g1/motion.pt",
]
#: 可通过环境变量指定
ENV_MENAGERIE = "MUJOCO_MENAGERIE"
ENV_POLICY = "G1_POLICY_PATH"

#: 改造后的模型写到哪个子目录（相对系统临时目录）
TMP_SUBDIR = "g1_walk_free"
