# 质心网络与逐关节网络检测

检测日期：2026-10-22。

## 检测对象

- 质心协调网络：`coordination.network.Coordinator`
- 逐关节辅助网络：`fullbody.ppo.JointAssist`
- 训练入口：`fullbody.train_balanced.train`
- 仿真入口：`fullbody.ppo.evaluate` / `BalancedEnv`

检测使用两个现有 checkpoint：

| 名称 | checkpoint | 说明 |
| --- | --- | --- |
| 1iter | `runs/chat_training/20261022_213707_7bb753/training/policy.pt` | AMASS A1-Stand，1 轮，逐关节网络开启 |
| 420iter | `runs/chat_training/20261020_131832_dbb48a/training/policy.pt` | test6，现有 checkpoint 记录为 420 轮 |

## 1. 训练和仿真是否调用

已确认调用：

- 训练时 `ActorCritic.distribution()` 使用 `mean()`，而 `mean()` 在
  `joint_assist_enabled=True` 时叠加逐关节网络输出。
- 评估时 `ActorCritic.from_checkpoint()` 恢复 `joint_assist`、
  `joint_assist_gain` 和 `joint_assist_residual`，`evaluate()` 使用
  `policy.mean()`。
- 训练时 `job_worker` 生成 `video_controller.coordinator_checkpoint`，
  `BalancedEnv.configure_video_controller()` 加载质心网络。
- 仿真评估从 checkpoint 的 `video_controller` 恢复同一质心网络。
- `BalancedEnv.control_target()` 在足部有接触时把质心网络修正叠加到
  29 维关节目标；腾空时不叠加。

真实评估路径的 `evaluation.json` 已确认包含：

```text
policy_used=true
joint_assist=true
video_controller.coordinator_checkpoint=...
airborne_steps
airborne_fraction
min_support_margin_m
airborne_support_compliance
```

## 2. 逐关节网络

两个 checkpoint 的 29 个逐关节输出头均有非零权重，29/29 个关节都产生了
非零动作增量。

| checkpoint | 动作增量最大绝对值 | 关节目标增量均值 | 关节目标增量最大 | 非零关节 |
| --- | ---: | ---: | ---: | ---: |
| 1iter | `5.44e-5` | `1.03e-5 rad` | `2.06e-5 rad` | 29/29 |
| 420iter | `4.49e-3` | `2.94e-4 rad` | `2.15e-3 rad` | 29/29 |

结论：逐关节网络确实开启并参与控制，但当前增益和作用幅度很小。1 轮权重
基本可视为噪声级；420 轮虽已增大一个数量级，仍远小于质心网络和主策略。

## 3. 质心网络

在接触帧上，质心网络修正全部 29 个关节均有非零影响：

| checkpoint | 接触帧 | 关节目标修正均值 | 最大修正 |
| --- | ---: | ---: | ---: |
| 1iter | 148/149 | `0.0078 rad` | `0.066 rad` |
| 420iter | 145/150 | `0.0180 rad` | `0.080 rad` |

结论：质心网络比逐关节网络影响明显，但只在接触阶段生效；腾空阶段不会
叠加该修正。

## 4. 训练记录与判据

记录字段会随启用模块增加：

- `metrics.jsonl` 增加 `joint_assist`。
- 评估增加 `airborne_steps`、`airborne_fraction`、`min_support_margin_m`、
  `airborne_support_compliance`、`coordinator_used`、`motion_gain` 等。
- checkpoint 增加 `joint_assist`、`joint_assist_gain`、`joint_assist_residual`、
  `airborne_reward`、`motion`、`video_controller`。

但技能登记的硬门槛没有随网络数量自动变化，仍是：

```text
reference_completed=true
fallen=false
max_root_xy_error_m <= 0.15
max_root_speed_mps <= 1.0
joint_mse <= 0.05
```

课程判据只在 `motion_gain > 1` 时放宽为 `0.20 m / 1.15 m/s / 0.07`；
自检只调整 `leg_weight` 和 `arm_weight`。因此“记录字段会增长”，但
“硬验收标准不会仅因为网络变多而自动放宽”。

## 5. 完整系统与纯 RL-PD 平衡对照

纯 RL-PD 定义：零速度命令、默认关节目标、无参考动作残差、无质心网络、
无逐关节网络，仅使用预训练平衡网络加 PD。

同一参考、同一初始相位、150 步上限：

| checkpoint | 配置 | 存活步数 | 摔倒 | 最大平面漂移 | 平均关节 MSE |
| --- | --- | ---: | --- | ---: | ---: |
| 1iter | 完整系统 | 149 | 否 | 0.085 m | 0.0200 |
| 1iter | 纯 RL-PD | 149 | 否 | 0.060 m | 0.0516 |
| 420iter | 完整系统 | 150 | 否 | 0.796 m | 0.4790 |
| 420iter | 纯 RL-PD | 150 | 否 | 0.268 m | 0.6437 |

对照表明：

- 完整系统更能跟踪参考关节轨迹，但在这两个参考上平面漂移更大。
- 纯 RL-PD 更接近保持原地平衡，但不跟踪参考动作。
- 1iter 的逐关节网络开关几乎不改变结果；420iter 的差异主要来自参考
  包络和质心网络，而不是逐关节网络。
- 420iter 去掉质心网络后最大漂移从 `0.796 m` 降到 `0.264 m`，关节 MSE
  基本不变（`0.479` 对 `0.469`）。当前质心网络并非在所有参考上都能改善
  平衡，需要继续训练或按任务调节。

以上是现有 checkpoint 的实测结果，不是对所有训练轮数和所有动作的普遍结论。
