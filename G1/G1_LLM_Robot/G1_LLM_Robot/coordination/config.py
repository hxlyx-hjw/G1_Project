"""跳跃/协调控制参数；独立于现有行走策略。"""
from dataclasses import dataclass


@dataclass(frozen=True)
class JumpConfig:
    dt: float=.002
    crouch_seconds: float=.65
    thrust_seconds: float=.16
    settle_seconds: float=1.5
    crouch_hip: float=-.35
    crouch_knee: float=.70
    crouch_ankle: float=-.35
    thrust_hip: float=-.04
    thrust_knee: float=.08
    thrust_ankle: float=-.04
    max_flight_seconds: float=.8
    max_correction_rad: float=.08

KEY_BODIES=('pelvis','torso_link','left_shoulder_pitch_link','right_shoulder_pitch_link',
    'left_elbow_link','right_elbow_link','left_wrist_yaw_link','right_wrist_yaw_link',
    'left_hip_pitch_link','right_hip_pitch_link','left_knee_link','right_knee_link',
    'left_ankle_roll_link','right_ankle_roll_link')
