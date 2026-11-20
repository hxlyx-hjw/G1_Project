# G1_LLM_Robot 项目详细文档与学习指南

> 生成时间：2026-10-19
> 本文基于对项目全部源码、文档、测试的通读整理，力求详尽，并附递进式学习顺序。

---

## 一、项目概述

这是一个**宇树 G1 人形机器人**的综合项目，把四件事整合到一个程序里：

1. **视频动作模仿**：从单人视频提取人体骨架，重定向到 G1 的肩肘腕/全身关节
2. **MuJoCo 物理仿真**：行走、摆臂、全身模仿、独立跳跃的物理回放
3. **DeepSeek 大模型控制**：Function Calling 闭环，用自然语言直接"开"机器人
4. **模仿学习**：视频参考 → 全身重定向 → PPO 训练 → 物理评估 → 技能库

统一启动入口是 `Start.py`，Visual Studio 项目是 `G1Robot.pyproj`。当前机器人工具 **25 个**，技能库**双轨**（节点权重技能 + 空间位置技能）。

**已验证能力**：上半身/全身动作重定向、RL 行走 + 手臂同时控制、DeepSeek 闭环工具调用、联网"搜视频→训练→入技能库"全链路、独立保守小跳（有协调网络时腾空 0.13s 落地稳定）、空间关键点技能回放。

**当前边界**：官方示例短训练仍失稳、视频全幅横向步伐未复现、跳跃是保守模板非通用跳跃策略、单目视频全局位移/深度不可靠。

---

## 二、系统架构（分层）

```
┌─────────────────────────────────────────────────────────┐
│  决策层：DeepSeek LLM（Function Calling 循环）           │
│    deepseek_robot.py · multi_agent.py                    │
├─────────────────────────────────────────────────────────┤
│  工具层：robot_tools.py（25 个工具 schema + 分发）       │
│    chat_store · file_parser · harness_search             │
├─────────────────────────────────────────────────────────┤
│  会话层：robot_bridge.py（后台仿真线程 + 线程安全 API）  │
├─────────────────────────────────────────────────────────┤
│  控制层：                                               │
│    g1_walkfree/   行走（RL 策略 + PD）                  │
│    video_motion/  上半身动作重定向                       │
│    fullbody/      全身训练 + 空间技能                    │
│    coordination/  质心协调网络 + 独立跳跃                │
├─────────────────────────────────────────────────────────┤
│  物理层：MuJoCo（500Hz 仿真）                            │
├─────────────────────────────────────────────────────────┤
│  基础设施：app_logging · runtime_io · environment_check  │
└─────────────────────────────────────────────────────────┘
```

---

## 三、文件结构（树状图）

```
G1_LLM_Robot/
├── self_video_replay.gif          训练回放动画
├── vedio_self/                    用户自有视频（test1.mp4）
└── G1_LLM_Robot/                  ← 主项目
    ├── Start.py                   统一启动入口（menu/video/chat/fullbody-*）
    ├── Start_G1.bat               Windows 双击启动
    ├── G1Robot.pyproj             Visual Studio 项目
    ├── README.md                  总览
    ├── .gitignore
    ├── api_key.txt / tavily_api_key.txt
    ├── requirements-video.txt / requirements-fullbody.txt
    │
    ├── deepseek_robot.py          DeepSeek 对话 + Function Calling
    ├── robot_bridge.py            会话桥接（后台仿真线程）
    ├── robot_tools.py             工具声明与分发（25 工具）
    ├── chat_store.py              SQLite 对话历史
    ├── multi_agent.py             多智能体协作
    ├── tool_output.py             工具结果控制台解析
    ├── file_parser.py             文件解析
    ├── harness_search.py          统一 Tavily 搜索
    ├── app_logging.py             日志
    ├── runtime_io.py              Windows 安全写入
    ├── environment_check.py       只读环境诊断
    │
    ├── g1_walkfree/               行走控制器
    │   ├── config.py / resources.py / controller.py / runners.py
    │   └── ui/  theme.py / widgets.py / panel.py / keymap.py
    │
    ├── video_motion/              视频动作
    │   ├── pose.py / retarget.py / tools.py / __main__.py / assets/
    │
    ├── fullbody/                  全身训练 + 空间技能
    │   ├── model.py / reference.py / refine.py / prepare_self.py
    │   ├── env.py / balanced.py / ppo.py / train_balanced.py
    │   ├── self_check.py / curriculum.py / jobs.py / job_worker.py
    │   ├── training_viewer.py / gif_export.py / skills.py / tools.py
    │   ├── web_video.py / web_pipeline.py
    │   ├── spatial_skills.py / spatial_run.py / tools_spatial.py
    │   └── __init__.py / __main__.py
    │
    ├── coordination/              质心协调 + 独立跳跃
    │   ├── config.py / network.py / jump.py
    │   ├── video.py / video_jump.py / tools.py / __main__.py
    │
    ├── docs/                      专项文档（15 篇）
    ├── tests/                     离线回归（14 个）
    ├── examples/                  演示/端到端（12 个）
    ├── skills/                    registry.json + spatial_registry.json
    └── motions/ runs/ outputs/ vedio/ logs/
```

---

## 四、核心模块详解

### 4.1 g1_walkfree/ — 行走控制器（一切的基础）

**职责**：让 G1 边走边摆臂，平衡由预训练 RL 策略保证。

| 文件 | 详解 |
|---|---|
| `config.py` | 集中所有常量。策略超参（KPS/KDS/DEFAULT_ANGLES/scale 来自官方 g1.yaml，**不可改**）、观测 47 维/动作 12 维、`ARM_PRESETS` 手臂预设、速度上限、摔倒判定、资源候选路径 |
| `resources.py` | 资源定位（显式参数 > 环境变量 > 主目录候选）+ **模型改造**：把 menagerie 29 自由度 MJCF 改成"腿 12 个 `<motor>` + 腰臂 17 个 `<position>`"混合模型 |
| `controller.py` | `WalkController` 核心：500Hz 物理步进（腿 PD 力矩 + 臂位置目标），50Hz 跑 TorchScript 策略组装 47 维观测。`q_target = action×0.25 + DEFAULT_ANGLES` |
| `runners.py` | `run_headless`（回归/扫参）与 `run_view`（3D 窗口） |
| `ui/` | 图形控制台：`theme`（尺寸/字体，`compute_layout` 屏幕自适应）、`widgets`（大滑杆/状态卡/滚动容器）、`panel`（主面板+定时器）、`keymap`（键盘映射） |

**关键设计**：模型改造只动 `<actuator>` 段，运动学树/质量/惯量/网格全不变，所以 RL 策略能直接迁移。

### 4.2 video_motion/ — 视频动作提取与上半身重定向

| 文件 | 详解 |
|---|---|
| `pose.py` | MediaPipe Tasks 视频推理（mediapipe 1.x 用 Tasks API），自动下载姿态模型，输出 `times/points/visibility` 到 npz |
| `retarget.py` | 人体坐标 → G1 肩肘 IK（`Retargeter.solve` 用 least_squares）；`play` 回放（锁会话推进、0.6s 过渡、角速度限幅、就地位置闭环） |
| `tools.py` | `robot_learn_video` / `robot_play_motion` 工具 schema |
| `__main__.py` | `setup / extract / play / learn` 命令 |

**关键**：右手坐标 `前 = 左 × 上`（旧代码用"上×左"导致前后镜像，已修正）。

### 4.3 deepseek_robot.py + robot_bridge.py + robot_tools.py — LLM 控制链

| 文件 | 详解 |
|---|---|
| `deepseek_robot.py` | `RobotChatBot`：`chat()` → `_run_until_final_text()` 循环（调模型 → 执行 tool_calls → role=tool 回传 → 直到最终文本）。默认模型 `deepseek-flash` + 思考模式（`reasoning_effort="medium"` + `extra_body thinking`）。含 `--selftest`、`--list-tools`、`--task` |
| `robot_bridge.py` | `RobotSession`：后台线程跑 MuJoCo，对外暴露线程安全动作 API。**核心是"空闲暂停仿真"**（否则 LLM 网络往返期间仿真自己跑掉）+ `@_needs_sim` 装饰器 + `_wait_sim` 自动激活。含离屏 GIF 录制 |
| `robot_tools.py` | `robot_tool_schemas()` 生成 JSON Schema；`RobotToolkit.dispatch()` 统一入口并兜底异常。挂载 25 个工具 |

**关键设计**（都是踩过的坑）：
- 仿真空闲必须暂停，否则坐标读数前后矛盾
- `_wait_sim` 必须自动激活仿真，否则与"空闲暂停"死锁
- 动作方法统一 `@_needs_sim` 装饰器，保证一个不漏
- 思考模式带 tools 必须回传 `reasoning_content`，否则 400

### 4.4 fullbody/ — 全身训练 + 空间技能（核心学习模块）

**职责**：视频参考 → 全身重定向 → 物理环境 → PPO 训练 → 评估 → 技能库。

| 文件 | 详解 |
|---|---|
| `model.py` | `build_model` 生成 29 关节全力矩模型（保留足部球接触、禁用其余网格碰撞） |
| `reference.py` | 全身 IK（`convert`）：基于骨段方向 + 足底约束，浮动根部就地；`Reference` 类相位采样 |
| `refine.py` | 足部落点/足底水平/支撑重心约束 + `savgol_filter` 平滑 |
| `prepare_self.py` | `trim` 提取最长连续全身可见区间 |
| `env.py` | `ImitationEnv`：50Hz 控制/500Hz 物理，29 维残差，188 维观测，奖励函数 |
| `balanced.py` | `BalancedEnv`：**预训练腿部平衡先验 + 29 维视频残差**（非直接全身 PD），腿按 `leg_weight` 缩幅 |
| `ppo.py` | 原生 PPO（GAE/裁剪目标/熵/梯度限幅）+ `train` + `evaluate` |
| `train_balanced.py` | 批量 CUDA PPO + 多 CPU 环境，小残差初始化，保存最优策略 |
| `self_check.py` | 手腿权重自检（物理对照评估，模型建议受限候选） |
| `curriculum.py` | 腿部权重课程（连续 3 次整段达标才升 0.05） |
| `jobs.py` / `job_worker.py` | 训练任务管理 + 后台进程状态机 |
| `training_viewer.py` | Tk 实时渲染窗口 |
| `gif_export.py` | 策略 GIF 导出 |
| `skills.py` | 节点权重技能库（`registry.json`） |
| `web_video.py` / `web_pipeline.py` | 联网搜索→下载→筛选 |
| `spatial_skills.py` | 空间绝对关键点技能库（`spatial_registry.json`） |
| `spatial_run.py` | 空间技能回放（spatial/node/linked 三模式） |
| `tools_spatial.py` | 空间技能工具 schema |

### 4.5 coordination/ — 质心协调网络 + 独立跳跃（进阶）

| 文件 | 详解 |
|---|---|
| `config.py` | `JumpConfig`（下蹲/蹬伸/腾空/落地参数）+ `KEY_BODIES` 14 关键节点 |
| `network.py` | `CoordinatorNet`：以质心为中心的星形图网络，14 节点 → 共享编码 → 汇聚 → 29 关节修正（±0.08rad）。教师为质心 Jacobian 约束控制（非端到端 RL） |
| `jump.py` | 独立跳跃状态机（stand/crouch/thrust/flight/landing/recover），仅关节力矩，落地后切平衡策略 |
| `video.py` | 视频跳跃事件标注（off/auto/explicit） |
| `video_jump.py` | 视频训练跳跃模板（`VideoJumpController`） |
| `tools.py` | `CoordinationJobs` 后台进程 + `robot_coordination_start/status` 工具 |
| `__main__.py` | CLI：`train` / `jump` |

### 4.6 基础设施（根目录新增）

| 文件 | 详解 |
|---|---|
| `app_logging.py` | 日志 `RotatingFileHandler`，项目不可写回退 LocalAppData |
| `runtime_io.py` | Windows 安全写入：唯一临时文件 + 短重试 + 原子替换（解决 WinError 5） |
| `environment_check.py` | 只读环境诊断（不泄密钥，不改数据） |
| `Start_G1.bat` | Windows 双击启动，固定目录 + 错误码透传 |

---

## 五、关键技术要点（易踩坑）

1. **右手坐标**：`+X 前、+Y 左、+Z 上`，`前 = 左 × 上`（不可写反）
2. **手臂符号坑**：`shoulder_pitch` 轴 `[0,1,0]`，**负值 = 手臂向前**（与直觉相反）
3. **仿真空闲暂停**：LLM 网络往返期间仿真必须冻结，否则机器人自己漂走
4. **模型改造只动 actuator**：运动学树不变，RL 策略才能迁移
5. **平衡先验 + 残差**：腿部是平衡先验主导，视频动作只按 `leg_weight`（默认 0.15）缩幅叠加
6. **completed ≠ 学会**：任务状态 completed 只表示流程走完，必须看 `evaluation.fallen`、`reference_completed`、漂移、动作相似度
7. **思考模式**：带 tools 必须回传 `reasoning_content`；思考模式不支持 `temperature`

---

## 六、控制理论图谱

见下方五个层次的反馈控制框图（详见《项目文档_v2.md》第二节）：

1. **系统级**：LLM → 工具分发 → 会话 → 控制器 → MuJoCo（结果回传闭环）
2. **腿部行走闭环**：速度指令 → RL 策略（控制器）→ PD（内环）→ G1 动力学（被控对象）→ 状态反馈
3. **全身模仿闭环**：参考 → PPO 残差 → 平衡先验 + 协调修正 → PD → G1 → 188 维观测反馈
4. **协调网络质心闭环**：质心误差 → CoordinatorNet → 关节修正 → 新质心（腾空时禁用）
5. **跳跃状态机**：阶段调度（开环）→ 落地切平衡策略（闭环）

---

## 七、学习顺序（递进式，从入门到进阶）

### 阶段 0：环境准备（先能跑）

1. 读 `README.md` 第 1~3 节（快速运行、Visual Studio、环境资源）
2. 装依赖：`python -m pip install -r requirements-video.txt`，聊天再加 `openai`
3. 资源定位：menagerie、`motion.pt`、姿态模型（`python -m video_motion setup`）
4. 跑环境诊断：`python environment_check.py`（只读，看缺什么）
5. 双击 `Start_G1.bat` 或 `python Start.py` 看菜单

### 阶段 1：先跑起来（建立全局印象）

1. `python Start.py --mode video` —— 看视频动作回放
2. `python Start.py --mode chat` —— 看大模型控制机器人
3. 通读树状图 + `docs/DIRECTORY.md`，知道每个目录干什么

### 阶段 2：行走控制（理解机器人怎么动）

**这是理解整个项目的地基。** 按顺序读：

1. `g1_walkfree/config.py` —— 先懂常量：策略超参、关节名、手臂预设、坐标约定
2. `g1_walkfree/resources.py` —— 懂"模型改造"（腿 motor + 臂 position）
3. `g1_walkfree/controller.py` —— 懂核心控制链：RL 策略 → PD → 力矩
4. 对照理解两个"坑"：右手坐标、手臂负值=向前

**掌握标准**：能说清"从速度指令到关节力矩"的完整链路。

### 阶段 3：视频动作（理解动作从哪来）

1. `video_motion/pose.py` —— MediaPipe 怎么提取骨架
2. `video_motion/retarget.py` —— 人体坐标怎么 IK 到 G1 肩肘
3. 理解"前 = 左 × 上"的右手坐标修正

**掌握标准**：能说清"视频 → npz 骨架 → 关节角"的转换。

### 阶段 4：LLM 控制链（理解怎么被大模型控制）

1. `robot_bridge.py` —— 先懂会话桥接（**重点是"空闲暂停"为什么必要**）
2. `robot_tools.py` —— 懂工具 schema 和分发
3. `deepseek_robot.py` —— 懂 Function Calling 循环 + 思考模式

**掌握标准**：能说清"用户一句话 → 工具调用 → 仿真执行 → 结果回传"的闭环。

### 阶段 5：全身训练（理解怎么学动作）

这是最核心也最复杂的部分，按数据流读：

1. `fullbody/model.py` —— 29 关节力矩模型
2. `fullbody/reference.py` + `refine.py` + `prepare_self.py` —— 参考动作怎么来（重定向→支撑修正→选段）
3. `fullbody/env.py` + `balanced.py` —— 训练环境（**重点：平衡先验 + 残差**）
4. `fullbody/ppo.py` + `train_balanced.py` —— PPO 怎么训
5. `fullbody/skills.py` —— 技能库怎么登记

**掌握标准**：能说清"视频 → 参考 → 环境 → PPO → 权重 → 技能"的完整训练链路。

### 阶段 6：协调与跳跃（进阶）

1. `coordination/network.py` —— 质心协调网络（星形图，14 节点）
2. `coordination/jump.py` —— 跳跃状态机
3. `coordination/video.py` / `video_jump.py` —— 怎么接入视频训练
4. 读 `docs/JUMP_AND_COM_NETWORK.md`、`docs/VIDEO_JUMP_COORDINATION.md`

### 阶段 7：空间技能库（进阶）

1. `fullbody/spatial_skills.py` —— 绝对关键点怎么存
2. `fullbody/spatial_run.py` —— 三种回放模式
3. 读 `docs/SPATIAL_SKILL_LIBRARY.md`

---

## 八、快速命令参考

```powershell
# 环境诊断（只读）
python environment_check.py

# 视频动作回放
python Start.py --mode video --seconds 8 --headless

# 聊天控制
python Start.py --mode chat

# 全身训练全流程
python Start.py --mode fullbody-retarget --seconds 8 --fps 10
python Start.py --mode fullbody-preview
python Start.py --mode fullbody-train --iterations 100 --steps 256
python Start.py --mode fullbody-eval --headless

# 质心协调网络
python -m coordination train --device cuda
python -m coordination jump --viewer --gif

# 离线回归测试（14 个）
python -m unittest discover -s tests -v
```

---

## 附：文档导航

| 想了解 | 看哪篇 |
|---|---|
| 总览 | `README.md` |
| 目录导航 | `docs/DIRECTORY.md` |
| 全身训练 | `docs/FULLBODY_TRAINING.md` |
| 聊天训练 | `docs/CHAT_TRAINING.md` |
| 技能库 | `docs/SKILL_LIBRARY.md` |
| 空间技能库 | `docs/SPATIAL_SKILL_LIBRARY.md` |
| 协调与跳跃 | `docs/JUMP_AND_COM_NETWORK.md`、`docs/VIDEO_JUMP_COORDINATION.md` |
| 腿部动作/GIF | `docs/LEG_RANGE_AND_GIF.md` |
| 联网学习 | `docs/WEB_ACTION_LEARNING.md` |
| 自检/搜索 | `docs/SELF_CHECK_AND_SEARCH.md` |
| 故障排查 | `docs/ERROR_405.md`、`docs/WINDOWS_PERMISSIONS.md`、`docs/TAVILY_CONFIG.md` |
