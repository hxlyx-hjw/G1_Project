# DeepSeek 对话启动视频训练与常开渲染

运行 `python Start.py --mode chat`，或在Visual Studio从Start菜单选择AI聊天。训练工具会自动注册到DeepSeek Function Calling循环：

| 工具 | 作用 |
|---|---|
| `robot_train_video` | 给定本地视频，立即启动后台提取、重定向、PPO训练并返回job_id |
| `robot_training_status` | 查询阶段、设备、轮次、指标、独立评估、渲染状态 |
| `robot_proactive_report` | 配置/立即发送主动训练报告，可设置10分钟等间隔 |
| `robot_training_stop` | 请求在检查点停止，保留权重 |
| `robot_training_viewer` | 打开或重新打开任务渲染窗口 |

聊天示例：

```text
用 ../vedio_self/test1.mp4 开始训练100轮，自动使用GPU，保持训练渲染窗口打开。告诉我腿部动作是否缩幅。
查询刚才的训练进度。
每5分钟主动给我发一次训练报告；现在先发一份。
停止刚才的训练，保留已经保存的权重。
重新打开刚才任务的渲染窗口。
```

启动参数：`video` 必填；`iterations=100`、`device=auto`、`render=true`、`leg_weight=0.15`、`arm_weight=1.0`、`motion_gain=1.35`、`max_seconds=60`、`fps=15`、`num_envs=4` 为默认值。`leg_weight` 可到2、`arm_weight` 可到1.5，`motion_gain` 为1~2；`max_seconds` 可到180。`max_episode_seconds` 省略时按参考时长自动放宽，避免长动作被固定10秒截断。`device`支持auto/cuda/cpu，auto优先可用CUDA；指定cuda但不可用时明确失败。相对视频路径相对于项目根目录。新增 `joint_assist=true`（29关节各一套辅助残差网络）、`joint_assist_gain=0.18` 和 `airborne_reward=true`（腾空足部支撑凸包、降低模仿/动作惩罚）。

## 腿部课程与GIF

现已增加可选 `curriculum_target` 分阶段扩大腿部权重，并支持 `robot_export_gif` / `robot_gif_status` 后台导出策略物理回放；完整设计和聊天示例见 [LEG_RANGE_AND_GIF.md](LEG_RANGE_AND_GIF.md)。

## 调用轮数与控制台解析

DeepSeek在一次回答中调用训练、训练状态、停止、渲染或GIF工具后，该次工具循环不再受原12轮上限限制，直到模型返回最终文本或用户中断。普通机器人工具仍默认12轮；Python接口传 `max_tool_rounds=None` 可对全部工具取消轮数限制。不限轮数不等于自动永久轮询，也不改变DeepSeek API的上下文长度或服务端速率限制。

默认控制台逐次输出中文解析及完整原始JSON：任务阶段、设备、训练轮次、奖励、课程、渲染权重轮次、物理回放结果和文件路径。原始结果仍完整回传模型。设置verbose=False（CLI的--quiet）时不打印解析。解析按规则读取实际字段，不额外调用大模型。completed会解释为任务结束，摔倒会明确标注失稳。

## 训练计时与主动报告

每个后台训练任务都会记录 `started_at`、`elapsed_seconds/elapsed_text`（从任务创建到当前）、`training_seconds/training_text`（纯PPO训练阶段）和结束时的 `duration_seconds/duration_text`（总耗时）。训练状态工具、控制台解析、训练监视窗口标题和主动报告都会显示这些字段；训练中还会给出 `eta_text` 预计剩余时间。直接运行 `fullbody` 训练时也会打印 `training_finished` 计时记录。

主动报告默认在训练启动后每10分钟发送一次，训练完成、失败或停止时再发送最终报告。报告会打印到聊天控制台、写入对话历史，并保存到：

```text
runs/chat_training/<job_id>/proactive_reports/report_YYYYmmdd_HHMMSS.md
runs/chat_training/<job_id>/proactive_reports/latest.md
```

配置方式：

```text
/proactive              查看状态
/proactive 5            改为每5分钟
/proactive off          关闭
/proactive on           开启
/proactive now          立即发送
```

也可以让DeepSeek调用 `robot_proactive_report`，参数为 `enabled`、`interval_minutes`、`job_id`、`send_now`。环境变量 `G1_PROACTIVE_REPORT_SECONDS=600` 可改默认间隔，`G1_PROACTIVE_REPORT=0` 可默认关闭。报告由实测状态生成，若API可用会再附加不超过120字的DeepSeek摘要；摘要失败不影响结构化报告。

启动时也可使用 `python Start.py --mode chat --proactive-interval-minutes 5`，或用 `--no-proactive` 关闭；`deepseek_robot.py` 对应参数为 `--proactive-interval-minutes`、`--no-proactive`、`--proactive-no-model`。

## 骨架提取进度

视频提取分为MediaPipe骨架检测和SMPL-X运动学拟合两个阶段。命令行会逐级显示：

```text
[骨架检测] 12/60（20%）
  SMPL-X 12/60（20%），关节误差 0.054 m
```

后台训练的 `status.json` 同步写入 `extraction` 字段（阶段、当前帧、总帧数、百分比），因此 `robot_training_status`、控制台解析和10分钟主动报告都能看到实时进度。MediaPipe仍使用独立工作进程，进度行通过标准输出实时回传，不改变原有检测结果格式。

## Docker 训练工具

DeepSeek 新增 5 个 Docker 工具，训练在 `g1-llm-robot:cu128` 容器中执行，不占用宿主机 Python 环境：

| 工具 | 作用 |
|---|---|
| `robot_docker_train` | 在指定宿主机目录训练；目录含 `reference.npz` 时直接训练，否则自动选择视频并完成提取/重定向 |
| `robot_docker_status` | 查询容器状态、阶段、轮次、计时和评估 |
| `robot_docker_summary` | 生成任务目录下的 `summary.md`，返回耗时、指标、评估和权重路径 |
| `robot_docker_export_video` | 从 `policy.pt` 导出 MP4/GIF 物理回放，输出必须位于项目 `runs` 内 |
| `robot_docker_stop` | 停止容器并保留已保存权重 |

示例对话：

```text
用 D:\videos\wave 目录在 Docker 里后台训练，400轮，CUDA。
查询刚才 Docker 训练的进度，并生成训练总结。
把刚才的训练策略导出为 MP4，20秒，15fps。
```

DeepSeek 会把宿主机目录只读挂载到容器的 `/data`，把项目 `runs` 挂载到容器内；
API Key 只通过进程环境传入，不写入 config/status。

回归验证包含连续25轮训练状态工具调用后正常返回最终答复；离线全套测试见下方“验证”。

## 任务与渲染行为

启动工具只创建后台进程，立即返回任务ID，聊天不会等待训练。预处理包括选取最长连续全身可见片段、全身IK及足部支撑修正；首次处理可能需要数十秒或更久。所有输出保存在唯一的 `runs/chat_training/<job_id>/`，不会覆盖其他训练。

持续窗口显示阶段、设备、轮次、奖励及画面使用的权重轮次。窗口按屏幕宽度自适应放大（约94%，最高1920×1080），图像区域同步提升离屏渲染分辨率（最高1600×900），并以机器人正面偏右的3/4轴测视角跟随播放，便于观察更大的训练活动范围。参考生成前显示等待状态；生成后运行独立物理回放，约每两秒检查最新权重。窗口显示的是**最近保存策略的独立演示**，并非训练采样世界本身；片段结束或摔倒会重置。训练结束后仍持续播放，用户可自行关闭。

关闭窗口不会停止训练，可再次用viewer工具打开。停止工具写入取消标记，训练在采样检查点退出；耗时的IK正在执行时，要等当前阶段结束检查取消。聊天退出时请求停止由该会话启动的训练；窗口仍保留，已保存权重不删除。若系统强制终止聊天进程则不能保证发送停止请求。

同一聊天会话限制一个训练任务同时运行；状态管理作用于当前会话。重启聊天后旧任务文件保留，但本版不会接管上一会话的进程。

## GPU与环境

沿用项目环境：torch、MuJoCo、SciPy、MediaPipe、OpenCV、Pillow和Windows Python的Tk界面。GPU用于PPO网络及批量观测更新，MuJoCo物理与姿态提取仍在CPU。不是Warp GPU物理。既有环境已满足，无需额外安装；缺包时查看worker.log并按项目requirements安装。依赖失败会由status返回failed，不会假装训练完成。

## 质量边界

当前算法为预训练G1腿部平衡先验加29维视频残差，并在其上增加平衡感知动作包络：躯干直立、根部速度和质心支撑偏差都在安全区时，`motion_gain` 放大参考和残差动作；任一指标变差时自动收回幅度，优先保平衡。手臂在失衡时比腿部更早收回；腾空阶段用足部支撑凸包约束质心，并放宽模仿与动作惩罚。默认 `leg_weight=0.15`、`motion_gain=1.35` 已比旧版更大幅度；`leg_weight=1.0~2.0` 属于高幅度尝试，仍可能漂移或摔倒。根部仍按就地参考，不能声称完整复现横向步伐。任务状态completed仅表示训练及评估流程已结束，应同时检查 `evaluation.fallen`、`reference_completed`、轨迹和动作相似度。技能库入库门槛仍要求完成、未摔倒、漂移≤0.15m、速度≤1m/s、关节MSE≤0.05，避免高幅度策略绕过质量检查。

## 验证

- 真实工具分发启动：CUDA、1轮、1环境，视频前2秒；预处理、训练、独立评估成功。
- 训练后渲染状态为rendering，已加载checkpoint_iteration=1，训练完成后窗口保持运行。
- 较长首轮测试超过测试等待时限后请求停止，正确变为cancelled。
- 工具schema与分发器一致；新增 `robot_proactive_report` 后仍保持 schema 与 handler 一一对应。
- 真实后台任务验证：状态依次经历 extracting（骨架/SMPL-X 百分比）→ training → evaluating → completed；总耗时41秒、纯训练2秒均正确写入；任务启动回调成功触发主动报告监听。

离线测试：`python -m unittest discover -s tests -v`（当前62项）。
真实工具链测试（会打开持续窗口并执行训练，不调用DeepSeek API）：`python -m examples.training_tool_smoke`。

实现：`fullbody/jobs.py`任务管理，`job_worker.py`后台流程，`training_viewer.py`渲染，`tools.py` schema；`robot_tools.py`分发，`deepseek_robot.py`负责注册和退出停止。
