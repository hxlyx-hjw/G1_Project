# 空间绝对位置技能库

项目现在有两类技能库：

| 类型 | 文件 | 内容 | 运行方式 |
|---|---|---|---|
| 节点权重技能 | `skills/registry.json` | `.pt` 策略、参考动作、训练参数和物理评估 | `robot_skill_run` |
| 空间位置技能 | `skills/spatial_registry.json` | 每帧绝对世界关键点坐标、时间戳、根部位姿、来源和评估 | `robot_spatial_skill_run` |

## 空间数据

训练任务独立物理评估通过后，会自动生成：

```text
runs/chat_training/<job_id>/spatial/spatial_points.npz
```

其中包括14个G1关键点：骨盆、躯干、双肩、双肘、双腕、双髋、双膝、双踝。坐标单位为米，坐标系为 `world_absolute_g1_meters`，包含：

- `times`：相对参考起点的时间。
- `positions`：`帧 × 关键点 × 3` 的世界绝对坐标。
- `root_qpos`：每帧根部位置和姿态。
- `keypoint_names`：关键点名称。

空间技能登记在 `skills/spatial_registry.json`，并通过 `node_skill_id` 关联原有节点权重技能。两库独立保存，空间库损坏不会覆盖节点库。

## DeepSeek工具

- `robot_spatial_skill_list`：列出空间技能。
- `robot_spatial_skill_get`：查看关键点、时间轴、来源、关联节点技能和评估。
- `robot_spatial_skill_run`：运行空间技能。

聊天示例：

```text
列出空间技能。
运行 spatial_xxxxxxxxxx 的绝对空间动作，打开仿真窗口。
运行 spatial_xxxxxxxxxx 的关联节点策略。
```

`mode`有三种：

- `spatial`：按绝对关键点轨迹回放，并计算空间误差。
- `node`：调用关联的原有节点权重策略，使用节点策略的物理评估。
- `linked`：运行空间参考并保留关联节点技能ID，用于对照；当前运行结果以空间参考为主。

## 与训练通路的关系

训练任务满足原有硬门槛后，同时登记：

```text
skills/registry.json              节点技能
skills/spatial_registry.json     空间技能
```

硬门槛仍为：参考完成、未摔倒、最大平面漂移≤0.15m、最大根部速度≤1m/s、关节MSE≤0.05。未通过任务不会写入任何可运行技能。

## 当前边界

空间库保存的是重定向后G1模型的世界绝对坐标；单目视频的真实深度、相机运动和全局位移不能自动恢复。当前空间运行使用登记的绝对关键点与参考姿态基线做空间误差回放，节点模式才使用训练策略。空间技能通过物理验收不代表视频动作语义已核实，也不代表可以直接部署实机。

回归：空间库格式、工具schema、原有技能库和全项目测试均已验证。当前机器人工具总数为25个。
