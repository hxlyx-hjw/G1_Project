# G1 桌面控制台（PyQt5）

一体化交互界面，把原来分散在终端里的能力集中到一个窗口：

- **对话**：DeepSeek Function Calling，流式呈现工具调用、参数、耗时和结果；
- **机器人**：实时位置/朝向/速度/可靠性，行走、转向、手臂预设、急停；
- **训练**：后台训练任务列表、进度、日志尾部、主动报告；
- **环境**：SuperMimic / RAX 连通性、运行配置、全部 56 个工具声明。

## 启动

双击：

```text
E:\TEST_RECORD\G1_Project\G1\启动G1界面.bat
```

或命令行：

```powershell
cd E:\TEST_RECORD\G1_Project\G1
python -m G1_Desktop.app
```

常用参数：

| 参数 | 作用 |
| --- | --- |
| `--no-viewer` | 不打开 MuJoCo 3D 窗口（仿真仍运行） |
| `--no-robot` | 只启动对话，不加载 MuJoCo / torch |
| `--model NAME` | 模型名，默认 `deepseek-flash` |
| `--base-url URL` | OpenAI 兼容接口地址 |
| `--no-search` | 不启用 Tavily 联网搜索 |
| `--no-proactive` | 关闭主动训练报告 |
| `--check` | 只检查路径和 PyQt5 后退出 |

API Key 沿用原项目的解析顺序：`--key` → `DEEPSEEK_API_KEY` → `G1_LLM_Robot/.../api_key.txt`。

## 功能说明

### 对话

- 输入框 Enter 发送、Shift+Enter 换行；底部有常用快捷指令。
- 每个工具调用显示为一张卡片：名称、参数、耗时、结果摘要；可展开原始 JSON。
- 模型返回的 `reasoning_content` 放在「▸ 思考过程」里，默认折叠。
- 训练期间的主动报告以黄色气泡插入对话。

### 机器人

- 状态每 600ms 刷新（仿真时钟、位置、速度、可靠性）。
- 行走按钮通过 `robot_walk` 工具执行，步长和速度倍率可调；停止/复位走 `robot_stop` / `robot_reset`。
- 手臂按钮来自 `g1_walkfree/config.py` 的 `ARM_PRESETS`。
- 右上角「急停」（Ctrl+Shift+S）直接从界面线程清零速度指令，不排在后端队列里。
- 3D 窗口是独立的 MuJoCo 窗口，可随时打开/关闭，不影响仿真。

### 训练

- 任务列表显示 月-日 时分秒 + 状态胶囊，完整 ID 在详情和 tooltip 里。
- 详情包含阶段、已用/训练/预计剩余时间、轮次进度、输出目录。
- 「立即报告」「渲染窗口」「导出 GIF」「停止任务」直接调用原有 `TrainingJobs`。
- 日志每 2 秒读取 `worker.log` 尾部。

### 环境

- 「检查研究环境」异步执行 `robot_supermimic_status` / `robot_rax_status`。
- 工具目录支持按名称/描述筛选，右侧显示说明和参数 Schema。
- 路径过长时中间省略，鼠标悬停可看完整路径。

## 架构

```text
G1_Desktop/
  app.py              启动器：路径、参数、Qt 初始化、torch/MuJoCo 预加载
  backend.py          后台线程 + 命令队列 + Qt 信号桥接
  main_window.py      主窗口、导航、状态灯、轮询、退出清理
  theme.py            配色、字体、全局 QSS
  widgets.py          卡片/胶囊/气泡/工具卡/流式布局
  pages/
    chat_page.py      对话
    robot_page.py     机器人
    training_page.py  训练
    env_page.py       环境
  tests/test_smoke.py 离屏 GUI 冒烟测试
```

线程模型：

```text
UI 线程                        Backend 线程（单队列）
  │ 点击/输入                      │
  ├──────────── ask/tool ─────────►│  RobotChatBot.chat / robot.dispatch
  │                                │
  │◄──── Qt 信号（queued）─────────┤  tool_started / tool_finished / chat_finished …
  │
  ├─ QTimer 600ms 读取 session.status()（内部有锁，线程安全）
  └─ QTimer 1500ms 读取 training.status()（只读文件 + poll 子进程）
```

所有阻塞操作都在 Backend 线程执行，界面线程不会被模型请求或仿真动作卡住。

## 与原有 CLI 的关系

现有代码一行未改：

- `deepseek_robot.py`、`robot_tools.py`、`Start.py` 保持原样；
- `G1_DeepSeek_Extensions` 原样复用；
- `Start_G1_Unified.py` / `启动G1.bat` 仍走原来的命令行流程。

GUI 通过运行时子类化 `ExtendedRobotChatBot` 接管 `_log`、`_dispatch_tool`、
`_emit_proactive` 三个事件点，不侵入现有实现。

## 已知环境问题：Qt 与 MuJoCo/torch 的 DLL 顺序

这台机器（Windows + Python 3.14）实测：

```text
先 import PyQt5，再 import mujoco/torch
→ OSError: [WinError 1114] DLL 初始化例程失败
```

反过来先加载 `mujoco` + `torch`，再加载 Qt 就正常。因此 `app.py` 在创建
`QApplication` 之前调用 `preload_robot_stack()`；预加载失败时自动降级为纯对话模式，
并在界面上显示原因，不会白屏或闪退。

## 测试

```powershell
cd E:\TEST_RECORD\G1_Project\G1
python -m unittest discover -s G1_Desktop/tests -v
```

测试使用 `QT_QPA_PLATFORM=offscreen`，不连接 DeepSeek、不启动 MuJoCo。
注意：Qt 的 offscreen 平台在本机不渲染文字，截图检查请用真实窗口。
