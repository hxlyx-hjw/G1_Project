# G1_LLM_Robot 项目文档（含控制理论图谱）

> 生成时间：2026-10-19
> 基于对项目全部文本文件的通读整理，不修改任何源文件（多余文件清单见末尾）。

---

## 一、文件树状图（最新）

```
G1_LLM_Robot/                                    ← 外层（用户目录）
├── G1_Final_Guide.md               ⚠旧版说明（与内层 README 重复）
├── G1_Fullbody_Training.md         ⚠旧版（与 docs/FULLBODY_TRAINING.md 重复）
├── self_video_training.md          ⚠旧版（与 docs/SELF_VIDEO_TRAINING.md 重复）
├── self_video_replay.gif           训练回放动画
├── G1_LLM_Robot_项目文档.md         历史整理文档
├── G1_LLM_Robot_目录结构与文件说明.md   历史整理文档
├── G1_LLM_Robot_目录结构与文件说明2.md  ⚠与上一条完全重复
├── vedio_self/                     用户自有视频（test1.mp4）
└── G1_LLM_Robot/                   ← 主项目（内层）
    ├── Start.py                    统一启动入口
    ├── Start_G1.bat                Windows 双击启动脚本（固定目录+错误码）
    ├── G1Robot.pyproj              Visual Studio 项目
    ├── README.md                   总览说明
    ├── .gitignore                  忽略规则
    ├── api_key.txt / tavily_api_key.txt   本地密钥（已忽略）
    ├── requirements-video.txt / requirements-fullbody.txt
    │
    ├── app_logging.py              ★日志（RotatingFileHandler，回退 LocalAppData）
    ├── runtime_io.py               ★Windows 安全写入（原子写+重试）
    ├── environment_check.py        ★只读环境诊断（不泄密钥）
    │
    ├── deepseek_robot.py           DeepSeek 对话 + Function Calling（含思考模式）
    ├── robot_bridge.py             机器人会话桥接层（后台仿真线程）
    ├── robot_tools.py              工具声明与分发（25 个工具）
    ├── chat_store.py               SQLite 对话历史
    ├── multi_agent.py              多智能体协作
    ├── tool_output.py              工具结果控制台解析
    ├── file_parser.py              文件解析（txt/md/pdf/docx）
    ├── harness_search.py           统一 Tavily 搜索
    │
    ├── g1_walkfree/                行走控制器包
    │   ├── config.py / resources.py / controller.py / runners.py
    │   └── ui/  theme.py / widgets.py / panel.py / keymap.py
    │
    ├── video_motion/               视频动作包
    │   ├── pose.py / retarget.py / tools.py / __main__.py / assets/
    │
    ├── fullbody/                   全身训练 + 空间技能包
    │   ├── model.py / reference.py / refine.py / prepare_self.py
    │   ├── env.py / balanced.py / ppo.py / train_balanced.py
    │   ├── self_check.py / curriculum.py / jobs.py / job_worker.py
    │   ├── training_viewer.py / gif_export.py / skills.py / tools.py
    │   ├── web_video.py / web_pipeline.py
    │   ├── spatial_skills.py       ★空间绝对关键点技能库
    │   ├── spatial_run.py          ★空间技能回放+节点策略联动
    │   ├── tools_spatial.py        ★空间技能工具 schema
    │   ├── __init__.py / __main__.py
    │   ├── ppo.py.bak              ⚠备份（多余）
    │   └── training_viewer.py.bak  ⚠备份（多余）
    │
    ├── coordination/               ★关键节点-质心协调 + 独立跳跃包
    │   ├── config.py               JumpConfig + 14 关键节点
    │   ├── network.py              CoordinatorNet 星形图网络
    │   ├── jump.py                 独立跳跃状态机
    │   ├── video.py                视频跳跃事件标注
    │   ├── video_jump.py           视频训练跳跃模板
    │   ├── tools.py                协调任务进程接口 + schema
    │   ├── __init__.py / __main__.py
    │
    ├── docs/                       专项文档（15 篇）
    │   ├── DIRECTORY.md / README.md
    │   ├── CHAT_TRAINING.md / FULLBODY_TRAINING.md / LEG_RANGE_AND_GIF.md
    │   ├── SELF_CHECK_AND_SEARCH.md / SELF_VIDEO_TRAINING.md
    │   ├── SKILL_LIBRARY.md / SPATIAL_SKILL_LIBRARY.md ★
    │   ├── TAVILY_CONFIG.md / VIDEO_MOTION.md / WEB_ACTION_LEARNING.md
    │   ├── JUMP_AND_COM_NETWORK.md ★ / VIDEO_JUMP_COORDINATION.md ★
    │   ├── ERROR_405.md ★ / WINDOWS_PERMISSIONS.md ★
    │
    ├── tests/                      离线回归测试（14 个）
    │   ├── test_video_motion.py / test_fullbody.py / test_curriculum.py
    │   ├── test_self_check.py / test_skills.py / test_skill_run.py
    │   ├── test_tool_loop.py / test_training_tools.py / test_web_action.py
    │   ├── test_config_review.py
    │   ├── test_coordination.py ★ / test_spatial_skills.py ★ / test_video_coordination.py ★
    │
    ├── examples/                   演示/端到端脚本（12 个）
    │   ├── smoke_video_motion.py / demo_llm_actions.py / example_module_usage.py
    │   ├── e2e_test.py / test_bridge.py / training_tool_smoke.py
    │   ├── web_learning_smoke.py / gif_tool_smoke.py / render_self.py / sweep_self.py
    │   └── jump_sweep.py ★
    │
    ├── skills/                     registry.json + spatial_registry.json ★
    ├── motions/ runs/ outputs/ vedio/ logs/ bin/ .vs/ __pycache__/
```

> ★ = 本次新读取到的新增模块；⚠ = 建议清理的多余文件。

---

## 二、控制理论图谱

### 2.1 系统级控制架构

```
用户指令
   │
   ▼
┌─────────────────┐   tool_calls    ┌──────────────────┐
│  DeepSeek LLM   │───────────────► │  RobotToolkit     │
│ (决策控制器)     │                │  (工具分发)        │
└─────────────────┘                └────────┬─────────┘
        ▲                                   │ dispatch
        │  role=tool 结果                    ▼
        │                         ┌──────────────────┐
        │                         │  RobotSession     │
        │                         │  (后台仿真线程)    │
        │                         └────────┬─────────┘
        │                                  │
        │                    ┌─────────────┼─────────────┐
        │                    ▼             ▼             ▼
        │              [WalkController] [视频回放]   [协调/跳跃进程]
        │                    │             │             │
        │                    └─────────────┼─────────────┘
        │                                  ▼
        │                             MuJoCo (500Hz)
        └──────────────────────────────────┘
                    （结果回传形成闭环）
```

### 2.2 腿部行走闭环（RL 策略 + PD 内环）

```
              ┌──────────────────── 状态反馈（47 维观测）──────────────────┐
              │                                                            │
 vx,vy,ωz  ┌──▼───────┐  q_target   ┌─────────┐   τ   ┌──────────┐        │
 ─────────►│  RL 策略  │────────────►│ PD 控制  │──────►│ G1 动力学 │────────┤
  (参考输入) │ (控制器)  │  50Hz       │ (内环)   │       │ (被控对象) │        │
            └──────────┘             └─────────┘       └────┬─────┘        │
            q_target = a×0.25 + 默认站姿   τ=kp·(t−q)−kd·dq  │              │
                                                             ▼              │
                                                          q, dq ────────────┘
```

### 2.3 全身模仿闭环（参考跟踪 + 残差 PPO + 平衡先验 + 协调网络）

```
 q_ref(参考) ──► [⊕] ──► [PPO 策略] ──► a(29维残差) ──► [⊕] ──► [PD] ──► τ ──► [G1]
                  ▲                                    ▲                     │
                  │                                    │ 平衡先验 target       │
                  │                                    │ 协调修正 Δq          │
                  │                                    │                      │
                  └──────────── 188 维观测 ◄────────────┴──────────────────────┘
```

### 2.4 协调网络（质心闭环，星形图网络）

```
 质心误差 e_com = com[:2] − 支撑中心
        │
        ▼
┌─────────────────────────────┐
│  CoordinatorNet（星形图网络） │
│  14 关键节点 ──► 共享编码      │
│       │                      │
│       └──► 汇聚 ──► ⊕ 质心特征 │
│                    │         │
│                    ▼         │
│            29 关节修正 Δq     │
│            （限幅 ±0.08 rad） │
└──────────────┬──────────────┘
               │ Δq 叠加到关节目标
               ▼
         [PD] ──► [G1] ──► 新的 com（反馈）
               └─────────────────┘
   腾空时禁用（无支撑，反馈失去意义）
```

### 2.5 跳跃状态机（开环时序 + 落地闭环切换）

```
          ┌──────────────────────── 阶段调度 ────────────────────────┐
          │  stand → crouch → thrust → flight → landing/recover       │
          ▼                                                           │
   phase ──► target 关节角 ──► [⊕] ──► [PD] ──► τ ──► [G1] ──► q     │
                                ▲                     │               │
                                │ 协调修正 Δq（接触时）  │               │
                                └─────────────────────┘               │
                                                                      │
  落地后：phase=landing → 切换 [平衡策略 motion.pt] 接管腿部（闭环）     │
  腾空阶段：无支撑，禁用质心纠偏，仅收腿姿态控制（开环）                  │
```

---

## 三、模块说明（重点新增）

### coordination/ — 关键节点-质心协调 + 独立跳跃

| 文件 | 功能 |
|---|---|
| `config.py` | `JumpConfig`（下蹲/蹬伸/腾空/落地时间参数）+ `KEY_BODIES` 14 关键节点 |
| `network.py` | `CoordinatorNet`：以质心为中心的星形图网络，14 节点 → 共享编码 → 汇聚 → 29 关节修正（±0.08rad）。教师为质心 Jacobian 约束控制，非端到端 RL |
| `jump.py` | 独立跳跃状态机（stand/crouch/thrust/flight/landing/recover），仅关节力矩，无根部外力；落地后切平衡策略恢复 |
| `video.py` | 视频跳跃事件标注（off/auto/explicit），单目不能验证真实离地 |
| `video_jump.py` | 供视频 PPO 训练调用的接触驱动保守跳跃模板 |
| `tools.py` | `CoordinationJobs` 后台进程接口 + `robot_coordination_start/status` 工具 |
| `__main__.py` | CLI：`train`（训练网络）/ `jump`（独立小跳） |

### fullbody/ 新增空间技能

| 文件 | 功能 |
|---|---|
| `spatial_skills.py` | 空间绝对关键点技能库（`skills/spatial_registry.json`），14 关键点世界坐标，与节点库独立 |
| `spatial_run.py` | 空间技能回放（spatial/node/linked 三模式） |
| `tools_spatial.py` | `robot_spatial_skill_list/get/run` 工具 schema |

### 基础设施（根目录新增）

| 文件 | 功能 |
|---|---|
| `app_logging.py` | 日志（`RotatingFileHandler`，项目不可写回退 LocalAppData） |
| `runtime_io.py` | Windows 安全写入：唯一临时文件 + 短重试 + 原子替换 |
| `environment_check.py` | 只读环境诊断（不泄密钥，不改数据） |
| `Start_G1.bat` | Windows 双击启动，固定目录 + 错误码透传 |

### 工具总数与技能库

- 机器人工具总数：**25 个**（原 17 + 协调 2 + 空间 3 + 其他）
- 技能库双轨：`skills/registry.json`（节点权重技能）+ `skills/spatial_registry.json`（空间位置技能）

---

## 四、多余文件清单（建议删除）

| 文件 | 原因 |
|---|---|
| `fullbody/ppo.py.bak` | 修复 evaluate() 前的原始备份，无价值 |
| `fullbody/training_viewer.py.bak` | 修复 load_checkpoint 前的原始备份，无价值 |
| 外层 `G1_LLM_Robot_目录结构与文件说明2.md` | 与"目录结构与文件说明.md"完全重复 |
| 外层 `G1_Final_Guide.md` | 与内层 `README.md` 同源旧版 |
| 外层 `G1_Fullbody_Training.md` | 与 `docs/FULLBODY_TRAINING.md` 重复 |
| 外层 `self_video_training.md` | 与 `docs/SELF_VIDEO_TRAINING.md` 重复 |
| 外层 `G1_LLM_Robot_项目文档.md`、`目录结构与文件说明.md` | 历史整理文档，可归档或删除 |

## 五、缺失/建议补齐清单

| 项目 | 说明 |
|---|---|
| README.md | 未完整提及 `coordination/`、空间技能库、`app_logging`/`runtime_io` 等新模块，建议补充总览 |
| `docs/DIRECTORY.md` | 未收录 `coordination/`、`spatial_*`、`app_logging.py` 等新文件，建议更新 |
| 协调网络接入行走 | `JUMP_AND_COM_NETWORK.md` 已注明"未来接管步行需加入协调网络版本"，当前未实现 |
| 跳跃验收 | 空间技能硬门槛（根速度 ≤1m/s）对跳跃偏严，跳跃视频 smoke 因根速度 1.076m/s 无法入库 |
