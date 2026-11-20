"""平衡感知的动作包络：放大参考动作，同时按稳定性自动收回幅度。"""
from dataclasses import asdict, dataclass, fields

import numpy as np


@dataclass
class MotionConfig:
    """动作放大与平衡保护的统一参数。"""

    max_gain: float = 1.45
    min_gain: float = 0.80
    leg_action_scale: float = 0.16
    arm_action_scale: float = 0.28
    waist_action_scale: float = 0.22
    max_leg_reference_delta: float = 0.90
    max_arm_reference_delta: float = 1.30
    guard: bool = True
    tilt_floor: float = 0.55
    tilt_span: float = 0.40
    speed_free: float = 0.80
    speed_span: float = 0.80
    com_free: float = 0.08
    com_span: float = 0.12
    # 手臂优先保证机体稳定：失衡时比腿部更早收回参考动作。
    arm_stability_floor: float = 0.35
    arm_stability_power: float = 1.0
    # RL-PD 影响能力：0 更偏向神经/参考控制，1 更偏向 RL-PD 平衡与默认姿态。
    # 手部和脚部分别用范围裁剪全局因子；默认值保持旧控制语义。
    rl_pd_influence: float = 1.0
    hand_rl_pd_range: tuple[float, float] = (0.0, 0.0)
    foot_rl_pd_range: tuple[float, float] = (1.0, 1.0)

    def validate(self):
        if not 0.0 < self.min_gain <= self.max_gain <= 3.0:
            raise ValueError('动作包络增益须满足 0 < min_gain <= max_gain <= 3')
        for name in ('leg_action_scale', 'arm_action_scale', 'waist_action_scale',
                     'max_leg_reference_delta', 'max_arm_reference_delta'):
            if getattr(self, name) < 0:
                raise ValueError('%s 不能为负数' % name)
        for name in ('tilt_span', 'speed_span', 'com_span'):
            if getattr(self, name) <= 0:
                raise ValueError('%s 必须为正数' % name)
        if not 0.0 <= self.arm_stability_floor <= 1.0:
            raise ValueError('arm_stability_floor 应在 0~1 之间')
        if self.arm_stability_power < 0:
            raise ValueError('arm_stability_power 不能为负数')
        if not 0.0 <= float(self.rl_pd_influence) <= 1.0:
            raise ValueError('rl_pd_influence 应在 0~1 之间')
        for name in ('hand_rl_pd_range', 'foot_rl_pd_range'):
            value = getattr(self, name)
            if isinstance(value, list):
                value = tuple(value)
                setattr(self, name, value)
            if (not isinstance(value, (tuple, list)) or len(value) != 2
                    or not 0.0 <= float(value[0]) <= float(value[1]) <= 1.0):
                raise ValueError('%s 必须是 [min,max] 且满足 0<=min<=max<=1' % name)
        return self

    def effective_hand_influence(self):
        return float(np.clip(self.rl_pd_influence, *self.hand_rl_pd_range))

    def effective_foot_influence(self):
        return float(np.clip(self.rl_pd_influence, *self.foot_rl_pd_range))

    def as_dict(self):
        return asdict(self)

    @classmethod
    def legacy(cls):
        """旧权重的控制语义：不做放大，保持原残差尺度。"""
        return cls(max_gain=1.0, min_gain=1.0, leg_action_scale=0.12,
                   arm_action_scale=0.20, waist_action_scale=0.20,
                   max_leg_reference_delta=0.50,
                   max_arm_reference_delta=10.0, guard=False,
                   arm_stability_floor=0.0, arm_stability_power=1.0).validate()

    @classmethod
    def from_mapping(cls, values=None, legacy=False):
        if isinstance(values, cls):
            return values.validate()
        base = cls.legacy() if legacy else cls()
        values = dict(values or {})
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError('未知动作包络参数：' + ','.join(sorted(unknown)))
        data = base.as_dict()
        data.update(values)
        return cls(**data).validate()


def balance_margin(upright, root_speed, com_error, config):
    """返回 0~1 的稳定性裕度；1 表示可放大幅度，0 表示应收回幅度。"""
    tilt = np.clip((float(upright) - config.tilt_floor) / config.tilt_span, 0.0, 1.0)
    speed = np.exp(-max(0.0, float(root_speed) - config.speed_free) / config.speed_span)
    com = np.exp(-max(0.0, float(com_error) - config.com_free) / config.com_span)
    return float(np.clip(0.55 * tilt + 0.25 * speed + 0.20 * com, 0.0, 1.0))


def amplitude(margin, config):
    """把稳定性裕度映射为动作放大系数。"""
    if not config.guard:
        return float(config.max_gain)
    ratio = float(np.clip(margin, 0.0, 1.0))
    return float(config.min_gain + (config.max_gain - config.min_gain) * ratio)


def arm_amplitude(margin, config, base_gain=None):
    """手臂在通用增益上再乘稳定优先系数；稳定时与腿部同幅，失衡时先收臂。"""
    if not config.guard:
        return float(config.max_gain if base_gain is None else base_gain)
    ratio = float(np.clip(margin, 0.0, 1.0))
    gain = amplitude(ratio, config) if base_gain is None else float(base_gain)
    priority = config.arm_stability_floor + (1.0 - config.arm_stability_floor) * ratio ** config.arm_stability_power
    return float(gain * priority)
