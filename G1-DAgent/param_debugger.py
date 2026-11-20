"""Bounded, explainable parameter proposals for the next G1 training run."""
from __future__ import annotations

from copy import deepcopy


BOUNDS = {
    'motion_gain': (1.0, 2.0),
    'leg_weight': (0.0, 2.0),
    'arm_weight': (0.25, 1.5),
    'rl_pd_influence': (0.0, 1.0),
    'joint_assist_gain': (0.0, 2.0),
}


def _clamp(name, value):
    low, high = BOUNDS[name]
    return max(low, min(high, round(float(value), 4)))


def _number(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _range(value, default):
    try:
        values = list(value)
        if len(values) != 2:
            return list(default)
        low, high = float(values[0]), float(values[1])
        return [max(0.0, min(1.0, low)), max(0.0, min(1.0, high))]
    except (TypeError, ValueError):
        return list(default)


class ParameterDebugger:
    """Apply conservative, auditable rules; never edits an existing job."""

    def base_parameters(self, record):
        config = record.get('config') or {}
        evaluation = record.get('evaluation') or {}
        motion = evaluation.get('motion_config') or {}
        return {
            'motion_gain': _number(config.get('motion_gain'),
                                   _number(motion.get('max_gain'), 1.35)),
            'leg_weight': _number(config.get('leg_weight'), .15),
            'arm_weight': _number(config.get('arm_weight'), 1.0),
            'rl_pd_influence': _number(
                config.get('rl_pd_influence'),
                _number(motion.get('rl_pd_influence'), 1.0)),
            'joint_assist_gain': _number(config.get('joint_assist_gain'), .18),
            'hand_rl_pd_range': _range(
                config.get('hand_rl_pd_range',
                           motion.get('hand_rl_pd_range')),
                [0.0, 0.0]),
            'foot_rl_pd_range': _range(
                config.get('foot_rl_pd_range',
                           motion.get('foot_rl_pd_range')),
                [1.0, 1.0]),
        }

    def propose(self, record):
        current = self.base_parameters(record)
        proposed = deepcopy(current)
        changes = []

        def change(name, delta, reason):
            old = float(proposed[name])
            new = _clamp(name, old + delta)
            if abs(new - old) > 1e-9:
                proposed[name] = new
                changes.append({
                    'parameter': name, 'old': old, 'new': new,
                    'delta': round(new - old, 4), 'reason': reason,
                })

        evaluation = record.get('evaluation') or {}
        fallen = bool(evaluation.get('fallen'))
        completion = float(evaluation.get('completion_fraction') or 0.0)
        drift = float(evaluation.get('max_root_xy_error_m') or 0.0)
        joint_mse = float(evaluation.get('joint_mse') or 0.0)
        reason = evaluation.get('termination_reason') or record.get('failure_reason')

        if record.get('outcome') in ('running', 'queued'):
            direction = ['训练尚未结束，先读取最新状态，不要提前判定成功。']
            confidence = 'low'
        elif fallen:
            confidence = 'medium'
            change('motion_gain', -.10, '评估摔倒，先降低参考动作幅度')
            change('leg_weight', -.05, '摔倒时优先收回腿部参考')
            change('arm_weight', -.10, '摔倒时优先收回手臂参考')
            change('rl_pd_influence', .05, '提高 RL-PD/默认姿态平衡权重')
            if joint_mse > .12:
                change('joint_assist_gain', .02, '关节跟踪误差较大，小幅增加逐关节辅助')
            direction = [
                '先复位并确认站立稳定；下一轮以 1~5 轮烟测验证，不要直接跑满长训练。',
                '检查 termination_reason；若持续 root_height/pelvis_tilt，优先缩短参考和降低 motion_gain。',
                '如果调参后仍摔倒，再考虑降低 num_envs 或提高已有平衡策略权重。',
            ]
        elif completion < .5:
            confidence = 'medium'
            change('motion_gain', -.05, '参考完成度偏低，先缩小动作幅度')
            change('leg_weight', -.03, '降低参考腿部残差以换取更长时间稳定')
            direction = [
                '检查训练参考是否包含镜头切换、脚部不可见或动作截断。',
                '先增加评估步数/训练轮数，再考虑提高动作幅度。',
            ]
        elif drift > .25:
            confidence = 'medium'
            change('motion_gain', -.05, '平面漂移偏大')
            change('rl_pd_influence', .03, '提高平衡先验权重')
            direction = [
                '检查根轨迹和接触标签；漂移不等于动作幅度不足。',
                '先验证原地/短距离片段，再恢复全局位移。',
            ]
        elif joint_mse > .08:
            confidence = 'medium'
            change('leg_weight', .05, '未摔倒但关节跟踪误差偏大')
            change('arm_weight', .05, '逐步提高手臂参考跟踪')
            direction = [
                '优先检查参考动作与机器人关节限位是否匹配。',
                '用短片段逐步增加 leg_weight/arm_weight，持续检查摔倒与漂移。',
            ]
        else:
            confidence = 'high' if evaluation.get('reference_completed') else 'medium'
            direction = [
                '当前评估没有明显失稳；继续增加轮数并保持参数，避免无依据大幅改动。',
                '每次只改变一个参数，保留旧 checkpoint 做 A/B 物理对照。',
            ]

        proposed['hand_rl_pd_range'] = _range(
            proposed.get('hand_rl_pd_range'), [0.0, 0.0])
        proposed['foot_rl_pd_range'] = _range(
            proposed.get('foot_rl_pd_range'), [1.0, 1.0])
        if not changes:
            direction.append('没有符合规则的参数修改建议；保持当前配置并继续采样。')

        reference = record.get('reference')
        command = None
        if reference:
            command = (
                'python -m fullbody.train_balanced "%s" '
                '--output "runs/dagent/%s_retry" '
                '--iterations 500 --envs 4 --device auto '
                '--motion-gain %.2f --leg-weight %.2f --arm-weight %.2f '
                '--rl-pd-influence %.2f --joint-assist-gain %.2f '
                '--hand-rl-pd-range %.2f %.2f --foot-rl-pd-range %.2f %.2f'
                % (
                    reference, record.get('job_id'),
                    proposed['motion_gain'], proposed['leg_weight'],
                    proposed['arm_weight'], proposed['rl_pd_influence'],
                    proposed['joint_assist_gain'],
                    proposed['hand_rl_pd_range'][0], proposed['hand_rl_pd_range'][1],
                    proposed['foot_rl_pd_range'][0], proposed['foot_rl_pd_range'][1],
                )
            )
        return {
            'job_id': record.get('job_id'),
            'outcome': record.get('outcome'),
            'failure_reason': reason,
            'confidence': confidence,
            'current': current,
            'proposed': proposed,
            'changes': changes,
            'directions': direction,
            'requires_user_confirmation': True,
            'launch_command': command,
            'note': '仅生成建议和命令，不修改已有训练任务或权重。',
        }
