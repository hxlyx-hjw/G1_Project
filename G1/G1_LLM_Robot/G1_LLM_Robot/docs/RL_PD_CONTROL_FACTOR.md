# RL-PD / 神经网络控制调节因子

## 定义

`rl_pd_influence` 是 0~1 的全局因子，表示 RL-PD 相对神经网络/参考动作的控制能力：

- `0`：更偏向神经网络与参考动作控制。
- `1`：更偏向预训练 RL-PD 平衡和默认关节姿态。

手部和脚部分别用收敛范围裁剪这个因子：

```text
hand_influence = clip(rl_pd_influence, hand_rl_pd_range[0], hand_rl_pd_range[1])
foot_influence = clip(rl_pd_influence, foot_rl_pd_range[0], foot_rl_pd_range[1])
```

例如全局因子 `0.7`、手部范围 `[0.2,0.6]`、脚部范围 `[0.5,0.9]` 时：

```text
hand_influence = 0.6
foot_influence = 0.7
```

## 控制公式

脚部/腿部：

```text
target_leg = DEFAULT + foot_influence * 0.25 * balance_action
             + gain * (video_leg_weight * reference_delta + leg_action_scale * action)
```

手部/手臂：

```text
target_arm = default_arm
             + (1 - hand_influence) * arm_gain
               * (arm_weight * reference_delta + arm_action_scale * action)
```

因此 `foot_influence` 越高，RL-PD 平衡策略对腿的控制越强；
`hand_influence` 越高，手臂越倾向于默认 PD 姿态，越低则越追随参考动作。

## 默认值和兼容性

为了保持旧 checkpoint 的逐位控制语义，默认值为：

```text
rl_pd_influence = 1.0
hand_rl_pd_range = [0, 0]
foot_rl_pd_range = [1, 1]
```

新训练由 DeepSeek 工具要求用户提供手部/脚部范围，因此默认值只用于旧权重和直接 Python/CLI 调用。

## DeepSeek 交互要求

每次启动视频、AMASS 或 Docker 训练前，DeepSeek 必须先询问：

```text
1. 全局 RL-PD 影响因子 rl_pd_influence（0~1）
2. 手部收敛范围 hand_rl_pd_range=[min,max]
3. 脚部收敛范围 foot_rl_pd_range=[min,max]
```

未得到明确回答时不得调用训练工具。`robot_amass_train` 在缺少手部/脚部范围时会直接拒绝启动。

## CLI 示例

```powershell
python -m fullbody.train_balanced `
  "E:\TEST_RECORD\G1_Project\G1\motions\amass\ACCAD\Male2General_c3d\A1-Stand_poses_120_jpos.npz" `
  --output E:\TEST_RECORD\G1_Project\G1\runs\amass_factor `
  --iterations 100 --envs 4 --device cuda `
  --rl-pd-influence 0.7 `
  --hand-rl-pd-range 0.2 0.6 `
  --foot-rl-pd-range 0.5 0.9
```

因子会写入 `motion` checkpoint、评估 `motion_config` 和技能库元数据，并在
`evaluate`、GIF 导出和训练渲染中恢复。
