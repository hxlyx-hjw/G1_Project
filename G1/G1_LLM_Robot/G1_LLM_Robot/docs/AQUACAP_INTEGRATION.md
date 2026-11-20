# AquaCap 思路在 G1 项目中的落地

参考文档：`AquaCap: A Training-Free Underwater Embodied Agent with Code-as-Policy`。

本次只迁移与当前仿真控制直接相关的机制，不引入水下感知模型或新的大模型训练。

## 1. 动作后置验证

`RobotToolkit.dispatch()` 对物理动作工具自动执行后置检查，覆盖：

- `robot_goto`：实际位置误差、朝向误差
- `robot_turn`：实际朝向误差
- `robot_walk`：预期位移与实际位移
- `robot_sequence` / 播放 / 手臂 / 腰部：摔倒和错误文本

工具返回末尾追加：

```text
[VERIFY] {"success":true,"outcome":"success","reason":...,"evidence":...,"correction":...}
```

“工具执行完成”不再等同于“目标达成”。失败会输出 `partial` 或 `failed` 及物理证据。

## 2. 结构化失败记忆

新增 `failure_memory.py`，把失败保存到：

```text
logs/failure_memory.jsonl
```

每条记录包含工具名、参数、结果、失败原因、物理证据、前后状态和修正建议。
DeepSeek 可调用 `robot_recall_failures` 检索同类历史修正。

记忆只作为建议；当前 `robot_status` 仍是事实来源。不会修改模型参数或技能权重。

## 3. 结构化状态可靠性

`RobotSession.status()` 新增：

```text
upright
height
reliability.stability
reliability.height_quality
reliability.quality
reliability.source = simulated_ground_truth
```

`describe()` 增加一段中文可靠性摘要，便于模型区分“位置接近目标但姿态不稳”等情况。

## 4. 未直接迁移的部分

- AquaCap 的水下 VLM/SAM3/ICP 感知没有照搬；当前视觉链路使用 MediaPipe + SMPL-X/双视角 DLT。
- 双层 Policy/Coding Agent 暂未拆成独立模型进程；先通过“计划工具 + 动作验证”降低局部失败的影响。
- 失败记忆不做向量数据库，使用可审计的 JSONL 和确定性检索。

## 5. 使用约定

DeepSeek 在执行物理动作后应检查 `[VERIFY]`：

- `success=true`：可以推进下一动作。
- `partial`：先读取状态，优先缩小动作幅度或拆分动作。
- `failed`：先复位/重读状态，再调用 `robot_recall_failures` 获取历史修正。

这些字段不会替代现有训练评估；它们只服务于动作级闭环。
