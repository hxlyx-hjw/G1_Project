# 视频训练整合：跳跃与质心协调

已将原独立小跳和关键节点—质心网络接入全身视频PPO训练，保留旧权重的原控制语义。

## 聊天入口

```text
训练 ../vedio_self/test1.mp4，开启 coordination=true，使用GPU并保持渲染；普通动作不添加跳跃。
```

已知参考起点后1.2秒需要起跳时：

```text
训练这个视频，coordination=true，jump_mode=explicit，jump_times=[1.2]，使用GPU。报告实际腾空、落地以及技能入库是否达标。
```

参数已同时加入 `robot_train_video` 和 `robot_learn_action`：

- `coordination=true`：新聊天训练默认启用。每个任务复制自己的网络文件，缺失时用当前设备训练教师网络。
- `jump_mode=off`：默认，按视频动作训练，不凭空添加跳跃。
- `jump_mode=auto`：识别可信全身骨架中“屈膝→快速伸展”的候选时间，只记入 `jump_events.json`，不自动执行。单目骨架与当前贴地参考不能可靠证明真实跳跃。
- `jump_mode=explicit`：按 `jump_times` 叠加保守跳跃模板，与肩臂视频参考和PPO残差联合运行。
- `jump_times`：相对于最长连续全身片段起点，而不是原视频起点。每次起跳前至少0.7秒、后至少1.6秒，两次相隔至少2.5秒。

如果视频本身没有跳跃，explicit相当于人为编排一个跳跃动作，不应称为恢复了视频跳高。

## 运行链路

视频提取 → 全身重定向/支撑修正 → 候选或显式事件标注 → 复制/训练质心网络 → 配置接触驱动跳跃模板 → PPO视频残差训练 → 独立评估 → 满足原有硬门槛后入库。

接触时质心网络输出受限关节修正，腾空时禁用支撑纠偏；肩肘腕继续跟踪视频。显式跳跃在下蹲和蹬伸阶段接管腿部目标，飞行阶段收腿，真实落地后恢复平衡先验和视频跟踪。PPO控制物理推进，没有根部外力或瞬移。

这是“保守跳跃模板+视频残差”的组合，不是已经学会任意视频跳跃的通用策略。原有奖励没有专门优化跳高、飞行时间和落地冲量，新机制主要通过控制模板执行跳跃并用事件验收检查。下一步若要精确模仿视频跳跃，应加入根部高度轨迹、接触相位条件观测、飞行姿态与落地奖励，以及多种子扰动训练。

## 权重、回放与技能

checkpoint新增 `video_controller` 字段，保存协调网络路径、跳跃时刻及模式。任务文件：

```text
runs/chat_training/<job>/com_graph.pt
runs/chat_training/<job>/jump_events.json
runs/chat_training/<job>/video_controller.json
```

训练、独立评估、训练实时查看器、GIF和原节点技能回放统一恢复此配置。旧checkpoint没有字段时不启用新控制器。节点技能登记同时记录配置。空间技能仍是参考关键点回放，不能据此展示策略真实跳跃轨迹。

报告包含 `coordinator_used`、跳跃阶段、实际起跳/落地时刻、连续腾空时间与 `all_scheduled_jumps_landed`。显式事件没有真实完成腾空落地时，即使参考时间结束也不判定参考完成。单次触地仍需参考结束前持续未倒地来验证稳定性。

## Start.py 入口

先用原有全身重定向生成参考：

```powershell
python Start.py --mode fullbody-retarget --video "你的动作.mp4" --reference motions/my/reference.npz
```

训练网络后将其用于视频训练：

```powershell
python -m coordination train --device cuda
python Start.py --mode fullbody-train --reference motions/my/reference.npz --output runs/my_video --coordinator runs/coordination/com_graph.pt --device cuda --iterations 100
```

需要显式起跳时再加 `--jump-times 1.2`。不带 `--coordinator` 的原有训练入口保持旧算法。查看结果：

```powershell
python Start.py --mode fullbody-eval --reference motions/my/reference.npz --output runs/my_video
```

## 实测

- 全套34项测试通过，包含配置恢复、物理推进、事件范围与自动检测不擅自执行。
- `runs/video_coordination_smoke`：启用网络的CUDA短训练完成6.68秒片段，未摔倒；不代表网络必然改善所有视频。
- `runs/video_jump_smoke`：人为指定参考1.2秒起跳，独立评估真实离地约0.14秒、落地后完成后续片段且未倒地；最大漂移约0.108m，但最大根速度约1.076m/s，超过原1m/s技能门槛，不能入库。
- 训练内评估和独立评估数值有差异，以独立报告为验收依据。未放宽阈值，任务结束不代表学会视频原动作。
