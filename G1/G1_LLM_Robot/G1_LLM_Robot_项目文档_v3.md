# G1_LLM_Robot 项目文档（2026-10-20 最新版）

> 生成时间：2026-10-20
> 基于对项目全部源码、文档、测试的通读整理。相比上一版，本项目已集成 SMPL-X 姿态拟合、全局轨迹恢复、动作包络、跳跃接管、主动报告、Docker 部署等能力。

---

## 一、项目概述

这是一个**宇树 G1 人形机器人**的综合项目，核心能力分四层：

1. **视频动作模仿**：单目视频 → MediaPipe/SMPL-X 姿态提取 → 全局轨迹恢复 → 全身重定向
2. **MuJoCo 物理仿真**：行走、摆臂、全身模仿、独立跳跃、跳跃后交接
3. **DeepSeek 大模型控制**：Function Calling 闭环，自然语言操作机器人（27+ 工具）
4. **模仿学习**：参考 → 平衡先验 + 残差 PPO + 关节辅助 + 协调网络 → 评估 → 技能库

**本次重大升级**（相比上一版）：

| 升级 | 说明 |
|---|---|
| **SMPL-X 姿态拟合** | `video_motion/smplx_fit.py`，用 MediaPipe 关键点逐帧优化 SMPL-X 参数，输出 127 关节 |
| **全局轨迹恢复** | `fullbody/global_track.py`（WHAM-lite），用支撑脚约束恢复根部位移 |
| **动作包络** | `fullbody/motion.py`，放大动作 + 按稳定性自动收回幅度 |
| **支撑凸包 + 关节辅助** | `fullbody/support.py`，腾空阶段平衡约束 |
| **跳跃接管** | `coordination/handoff.py`，跳跃落地后交接回行走控制器 |
| **主动报告** | `proactive.py`，后台线程定时报告训练状态 |
| **Docker 部署** | 完整镜像（CUDA 12.8.1 + torch 2.11 cu128 + RTX 50 支持） |
| **自主训练** | `auto_train.py`，`docker run -d` 后台自主训练 |

**参考重定向模式**：`root_mode=in_place / contact_grounded / auto`，默认 `contact_grounded`（保留全局轨迹）。

---

## 二、文件树状图

```
G1_LLM_Robot/                                  ← 外层
├── Dockerfile / .dockerignore                 Docker 镜像
├── docker_build.ps1 / docker_run.ps1          Docker 构建/运行脚本
├── docker_entrypoint.sh                       Docker 入口
├── DOCKER_README.md                           Docker 说明
├── vcxsrv_start.ps1                           VcXsrv 窗口启动
├── G1_LLM_Robot_详细文档.md / _项目文档_v2.md   历史文档
├── vedio_self/                                用户视频
└── G1_LLM_Robot/                              ← 主项目
    ├── Start.py                               统一入口（含 Ollama 选项）
    ├── Start_G1.bat                           Windows 双击启动
    ├── G1Robot.pyproj                         Visual Studio 项目
    ├── README.md                              总览
    ├── .gitignore / api_key.txt / tavily_api_key.txt
    ├── requirements-video.txt / requirements-fullbody.txt
    │
    ├── auto_train.py                          ★自主训练（docker 后台）
    ├── proactive.py                           ★训练主动报告
    ├── viewer_camera.py                       ★统一轴测相机 + 无显示降级
    ├── motion_logger.py                       ★运动日志（位置/速度/加速度+绘图）
    ├── smplx_utils.py                         ★SMPL-X 模型加载
    ├── app_logging.py / runtime_io.py / environment_check.py   基础设施
    │
    ├── deepseek_robot.py                      DeepSeek 对话（含 Ollama 支持 + 思考模式）
    ├── robot_bridge.py                        会话桥接（含运动日志）
    ├── robot_tools.py                         工具分发（27+ 工具）
    ├── chat_store.py / multi_agent.py / tool_output.py
    ├── file_parser.py / harness_search.py
    │
    ├── g1_walkfree/                           行走控制器
    │   ├── config.py / resources.py / controller.py / runners.py
    │   └── ui/  theme / widgets / panel / keymap
    │
    ├── video_motion/                          视频动作（升级版）
    │   ├── pose.py                            提取入口（backend=auto/smplx/mediapipe）
    │   ├── smplx_fit.py                       ★SMPL-X 运动学拟合
    │   ├── mediapipe_backend.py               ★MediaPipe 后端
    │   ├── mediapipe_worker.py                ★MediaPipe 子进程（防阻塞）
    │   ├── retarget.py / tools.py / __main__.py / assets/
    │
    ├── fullbody/                              全身训练（升级版，26 文件）
    │   ├── model.py / reference.py / refine.py / prepare_self.py
    │   ├── global_track.py                    ★全局轨迹（WHAM-lite）
    │   ├── motion.py                          ★动作包络
    │   ├── support.py                         ★支撑凸包 + 关节辅助
    │   ├── env.py / balanced.py / ppo.py / train_balanced.py
    │   ├── self_check.py / curriculum.py / jobs.py / job_worker.py
    │   ├── training_viewer.py / gif_export.py / skills.py / tools.py
    │   ├── web_video.py / web_pipeline.py
    │   ├── spatial_skills.py / spatial_run.py / tools_spatial.py
    │   └── __init__.py / __main__.py
    │
    ├── coordination/                          协调 + 跳跃（9 文件）
    │   ├── config.py / network.py / jump.py
    │   ├── handoff.py                         ★跳跃落地交接
    │   ├── video.py / video_jump.py / tools.py / __main__.py
    │
    ├── docs/                                  专项文档（17 篇）
    │   ├── DIRECTORY / CHAT_TRAINING / FULLBODY_TRAINING
    │   ├── JUMP_AND_COM_NETWORK / VIDEO_JUMP_COORDINATION
    │   ├── JUMP_HANDOFF / VIDEO_TRACKING_UPGRADE
    │   ├── LEG_RANGE_AND_GIF / SELF_CHECK_AND_SEARCH
    │   ├── SELF_VIDEO_TRAINING / SKILL_LIBRARY / SPATIAL_SKILL_LIBRARY
    │   ├── TAVILY_CONFIG / VIDEO_MOTION / WEB_ACTION_LEARNING
    │   ├── ERROR_405 / WINDOWS_PERMISSIONS
    │
    ├── tests/                                 离线回归（19 个，61+ 项）
    │   ├── test_video_motion / test_fullbody / test_curriculum
    │   ├── test_global_track / test_jump_handoff / test_motion_envelope
    │   ├── test_proactive / test_support_and_assist / test_viewer_camera
    │   ├── test_self_check / test_skills / test_skill_run / test_spatial_skills
    │   ├── test_tool_loop / test_training_tools / test_web_action
    │   ├── test_coordination / test_video_coordination / test_config_review
    │
    ├── examples/                              演示（13 个，含 jump_handoff_smoke）
    ├── skills/                                registry.json + spatial_registry.json
    └── motions/ runs/ outputs/ vedio/ logs/
```

---

## 三、核心架构（分层）

```
┌─────────────────────────────────────────────────────────┐
│ 决策层：DeepSeek LLM（含 Ollama 本地模型切换）           │
│   deepseek_robot.py · multi_agent.py · proactive.py     │
├─────────────────────────────────────────────────────────┤
│ 工具层：robot_tools.py（27+ 工具）                       │
│   chat_store · file_parser · harness_search             │
├─────────────────────────────────────────────────────────┤
│ 会话层：robot_bridge.py（后台仿真 + 运动日志）           │
├─────────────────────────────────────────────────────────┤
│ 控制层：                                                │
│   g1_walkfree/   行走（RL 策略 + PD）                   │
│   video_motion/   姿态提取（MediaPipe/SMPL-X）          │
│   fullbody/       全身训练（全局轨迹 + 动作包络 + 辅助） │
│   coordination/   协调网络 + 跳跃 + 落地交接             │
├─────────────────────────────────────────────────────────┤
│ 物理层：MuJoCo（500Hz，viewer_camera 统一相机）          │
├─────────────────────────────────────────────────────────┤
│ 基础设施：app_logging · runtime_io · environment_check  │
│ 部署：Docker（CUDA 12.8.1 + torch 2.11 cu128）          │
└─────────────────────────────────────────────────────────┘
```

---

## 四、核心模块详解（重点新增）

### 4.1 视频动作升级（video_motion/）

| 文件 | 功能 |
|---|---|
| `pose.py` | 提取入口，`backend=auto/smplx/mediapipe`，默认优先 SMPL-X 失败回退 MediaPipe |
| `smplx_fit.py` | ★SMPL-X 运动学拟合：MediaPipe 关键点作观测，逐帧优化 `global_orient/body_pose/transl`，输出 127 关节 |
| `mediapipe_backend.py` | MediaPipe 后端封装（lite/full/heavy 三档精度） |
| `mediapipe_worker.py` | MediaPipe 子进程（解决 Windows 关闭阻塞 40 秒问题） |
| `retarget.py` | 肩肘 IK 重定向 + 回放 |

**姿态精度**：`G1_POSE_QUALITY=lite|full|heavy`，默认 full；SMPL-X 用 `E:\TEST_RECORD\G1_Project\G1\models\SMPL\models_smplx_v1_1\models\smplx` 权重。

### 4.2 全身训练升级（fullbody/）

| 文件 | 功能 |
|---|---|
| `global_track.py` | ★WHAM-lite：支撑脚不动约束反推根部位移/转向/接触标签 |
| `motion.py` | ★动作包络：放大动作（max_gain 1.45）+ 按稳定性自动收回 |
| `support.py` | ★足部凸包 + 质心裕度，腾空阶段平衡约束 |
| `reference.py` | 重定向，`root_mode=in_place/contact_grounded/auto` |
| `refine.py` | 稳健 IK + 脚底水平 + 支撑/质心约束 + 根轨迹平滑 |
| `env.py` | 训练环境，真实接触标签 + 腾空比例 + 关键点 RMSE |
| `balanced.py` | 平衡先验 + 残差 + 关节辅助 + 腾空奖励 |
| `ppo.py` / `train_balanced.py` | PPO 训练 |

**核心变化**：参考从 `in_place`（就地）升级为 `contact_grounded`（全局轨迹），恢复了行走/横向移动的位移。

### 4.3 协调与跳跃（coordination/）

| 文件 | 功能 |
|---|---|
| `network.py` | CoordinatorNet 质心协调网络 |
| `jump.py` | 独立跳跃状态机 |
| `handoff.py` | ★跳跃落地快照保存 + 交接回行走控制器 |
| `tools.py` | 协调任务 + 跳跃交接工具 |

**跳跃接管流程**：跳跃落地稳定 → 保存 `handoff.npz` → `robot_jump_handoff` 导入行走会话 → 后续 walk/goto 继续。

### 4.4 主动报告与自主训练

| 文件 | 功能 |
|---|---|
| `proactive.py` | 训练计时 + 主动报告（后台线程定期发送 + 保存 Markdown） |
| `auto_train.py` | 自主训练启动器（动作描述 → 搜视频 → 训练 → 等待完成） |

### 4.5 统一相机与部署

| 文件 | 功能 |
|---|---|
| `viewer_camera.py` | 统一轴测相机（azimuth 135 / elevation -24）+ 无显示环境降级 + X11 容错 |
| `Dockerfile` 等 | 完整 Docker 镜像（CUDA 12.8.1 + Ubuntu 24.04 + Python 3.12 + torch 2.11 cu128） |

---

## 五、关键技术要点

1. **右手坐标**：`+X 前、+Y 左、+Z 上`，`前 = 左 × 上`
2. **手臂符号**：`shoulder_pitch` 负值 = 向前（与直觉相反）
3. **仿真空闲暂停**：LLM 网络往返期间仿真必须冻结
4. **平衡先验 + 残差**：腿部平衡先验主导，视频动作按权重缩幅叠加
5. **全局轨迹**：`contact_grounded` 模式恢复位移，替代旧的 `in_place`
6. **动作包络**：放大动作的同时按倾斜/速度/质心裕度自动收回，手臂优先保证稳定
7. **跳跃交接**：跳跃落地快照 → 关节名映射 → 导入行走会话
8. **思考模式**：DeepSeek 思考模式需回传 `reasoning_content`；Ollama 不支持需关闭

---

## 六、Docker 部署

```powershell
# 构建镜像
.\docker_build.ps1                    # 镜像 g1-llm-robot:cu128，约 14.2GB

# 交互菜单
.\docker_run.ps1

# 无窗口训练
.\docker_run.ps1 --mode fullbody-train --iterations 10 --steps 256 --device cuda

# 后台自主训练
docker run -d --name g1-auto-train --gpus all --shm-size=4g `
  -e TAVILY_API_KEY="tvly-..." -e DEEPSEEK_API_KEY="sk-..." `
  -v "E:\TEST_RECORD\G1_Project\G1\models\SMPL\models_smplx_v1_1\models\smplx:/models/SMPL:ro" `
  -v "E:\TEST_RECORD\G1_Project\G1\models\WHAM:/models/WHAM:ro" `
  g1-llm-robot:cu128 python -u auto_train.py "挥手" --iterations 500 --device cuda

# 测试
docker run --rm --gpus all g1-llm-robot:cu128 python -m unittest discover -s tests -v
```

已验证：CUDA 可用（RTX 5070）、MuJoCo EGL 离屏渲染、61 项测试通过（3 项 Windows 路径用例跳过）。

---

## 七、命令参考

```powershell
# 环境诊断
python environment_check.py

# 视频动作（SMPL-X 后端）
python -m video_motion extract "D:\视频\动作.mp4" --output motions/wave.npz --backend smplx --quality full

# 全身重定向（默认 contact_grounded + refine）
python -m fullbody retarget motions/wave.npz --output motions/fullbody/reference.npz

# 聊天控制（DeepSeek / Ollama）
python Start.py --mode chat
python Start.py --mode ollama

# 协调网络 + 跳跃
python -m coordination train --device cuda
python -m coordination jump --viewer --gif

# 离线回归测试
python -m unittest discover -s tests -v
```

---

## 八、验证状态

| 验证项 | 结果 |
|---|---|
| 离线回归测试 | 61+ 项通过 |
| SMPL-X 端到端 | 19 帧，平均关节误差约 0.055m |
| 官方示例 | Full 模型 60 帧，contact_grounded + refine 正常 |
| 跳跃接管 | 窗口持续 + 落地状态导入 + 前进 0.5 秒未摔倒 |
| Docker CUDA | RTX 5070 识别，EGL 离屏渲染正常 |

**当前边界**：SMPL-X 是"关键点 + 运动学拟合"，不是 4D-Humans/WHAM 的图像级重建；空中阶段无可靠绝对高度；`betas` 固定 0；参考可执行性仍受 IK/关节限位/简化碰撞限制。
