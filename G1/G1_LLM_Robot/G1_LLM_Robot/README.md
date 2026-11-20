# G1 视频动作模仿与大模型控制：最终说明

跳跃后保留窗口并将稳定落地状态交接到行走仿真，见 [跳跃交接说明](docs/JUMP_HANDOFF.md)。

跳跃机制和质心协调网络已接入视频训练、渲染和权重回放，见 [视频整合说明](docs/VIDEO_JUMP_COORDINATION.md)。

新增独立 [小跳机制与关键节点—质心协调网络](docs/JUMP_AND_COM_NETWORK.md)，保留原有行走、训练和技能库控制接口。

新增 [空间绝对位置技能库](docs/SPATIAL_SKILL_LIBRARY.md)，训练通过后与节点权重技能同时登记，可独立运行或联动。

原Harness搜索已统一，训练支持手腿权重自检、自调和可选DeepSeek复核，见 [最新说明](docs/SELF_CHECK_AND_SEARCH.md)。

新增 [Harness联网搜索→视频训练→技能库通路](docs/WEB_ACTION_LEARNING.md)，可直接给出动作描述启动后台自主学习；公网搜索质量和动作语义仍需核实。

新增 [训练技能库](docs/SKILL_LIBRARY.md)：通过独立物理评估的策略会自动登记，DeepSeek可列出、查询并直接调用技能仿真。

目录已按用途整理，文件位置和辅助脚本命令见 [目录导航](docs/DIRECTORY.md)。DeepSeek现支持后台视频训练、GPU选择及常开渲染，见 [聊天训练工具说明](docs/CHAT_TRAINING.md)。离线回归统一执行 `python -m unittest discover -s tests -v`。

AMASS 的 G1 重定向动作已下载并接入现有全身训练参考格式，见 [AMASS 数据说明](docs/AMASS.md)。使用 `python -m fullbody amass-download` 下载，`python -m fullbody amass-export` 转换为 `motions/amass`。

本地多数据集目录、下载状态、AIST++ 双视角和 Motion-X 单视频训练入口见
[G1 本地数据集说明](docs/DATASETS_G1.md)。统一命令：`python -m fullbody.datasets inventory`。

RL-PD 与神经网络控制的混合因子、手部/脚部收敛范围及 DeepSeek 询问规则见 [控制因子说明](docs/RL_PD_CONTROL_FACTOR.md)。

双机位视频的 MediaPipe 2D + DLT 三角化、正交近似/相机标定和 SMPL-X 多视角拟合见 [双视角三维动作](docs/MULTIVIEW_VIDEO.md)。

动作后置验证、失败记忆和状态可靠性评分参考 AquaCap 的闭环思想，见 [AquaCap 落地说明](docs/AQUACAP_INTEGRATION.md)。

训练数据（`runs`、`outputs`、`motions`、`skills`、`logs`、`vedio`）已迁移到
`E:\TEST_RECORD\G1_Project\G1`，原路径为目录联接；检查或重建联接运行项目外层的
`setup_data_root.ps1`。

Docker 镜像 `g1-llm-robot:cu128`（CUDA 12.8 / Python 3.12 / PyTorch 2.11）已可用，
构建与运行方式见 [DOCKER_README.md](../../docker/DOCKER_README.md)。

本项目将本地视频姿态提取、G1 上半身动作重定向、MuJoCo 仿真回放及 DeepSeek 工具调用整合到同一程序。统一启动文件是 **Start.py**，Visual Studio 项目是 **G1Robot.pyproj**。

> 新增全身训练模块：参考重定向、29维残差控制、PPO训练与物理评估已整合到 Start 菜单3~6，完整命令和限制见 [全身训练说明](docs/FULLBODY_TRAINING.md)。参考重定向默认使用接触约束的全局根轨迹与足底支撑修正，可用 `--root-mode in_place` 回退旧行为。原有视频菜单1和DeepSeek视频工具继续使用肩肘回放。全身训练链路可运行，但示例短训练权重尚未稳定完成动作。

## 1. 快速运行

在包含 `Start.py` 的项目目录执行：

```powershell
python Start.py
```

菜单选项：

- 回车或 `1`：提取 `vedio/example.mp4` 的动作并打开 G1 三维回放，不需要 API Key。
- `2`：启动 DeepSeek 聊天控制，默认打开三维窗口。
- `0`：退出。

直接运行视频示例：

```powershell
python Start.py --mode video
```

使用自己的视频、放慢回放或录制 GIF：

```powershell
python Start.py --mode video --video "D:\视频\挥手.mp4" --seconds 10 --fps 15 --speed 0.7
python Start.py --mode video --gif vedio/result.gif
python Start.py --mode video --headless
python Start.py --mode chat
```

`--seconds` 是最多读取的视频秒数，范围 `(0,60]`；`--fps` 为提取采样率，范围 `1~60`；`--speed` 是回放倍率，范围 `0.25~2`。视频不足指定长度时读取到结尾。`--headless` 关闭三维窗口。

Start 将工作目录固定到自身所在目录，因此 IDE 从其他工作目录启动也可以定位资源。相对视频、GIF 路径都相对于该项目目录。

## 2. Visual Studio 启动

1. 安装 Visual Studio 的“Python 开发”工作负载。
2. 打开 `G1Robot.pyproj`，其中已指定 `Start.py` 为启动文件。
3. 选择安装依赖的 Python 解释器。本机验证用解释器为：

   ```text
   C:\Users\lenovo\AppData\Local\Python\pythoncore-3.14-64\python.exe
   ```

4. 按 **Ctrl+F5** 运行，或 **F5** 调试，在控制台选择启动模式。

也可以“打开文件夹”，右键 `Start.py` 并设置为启动文件。Visual Studio Code 可直接运行 `Start.py`，同样显示菜单。

可在 `Start.py` 顶部修改默认设置：

```python
DEFAULT_MODE = 'menu'  # menu / video / chat
DEFAULT_VIDEO = PROJECT_DIR / 'vedio' / 'example.mp4'
DEFAULT_SECONDS = 8.0
```

设为 `video` 后，点击运行即可直接播放示例。正常视频回放结束后保留三维窗口，关闭窗口退出；聊天退出或执行异常时释放机器人会话。

## 3. 环境与资源

本机已验证：Windows、Python 3.14、NumPy 2.5.2、SciPy 1.18.1、OpenCV 5.0.0、MediaPipe 1.1.1、smplx 0.1.28、MuJoCo 3.16.0、PyTorch 2.11.0+cu128。

检查环境并下载姿态模型：

```powershell
python -m video_motion setup
```

新环境安装视频处理依赖：

```powershell
python -m pip install -r requirements-video.txt
```

该依赖文件只覆盖视频处理和仿真；聊天还需要现有 DeepSeek 程序的依赖，例如 `openai`。本机已有 PyTorch，无需重装；新环境缺少 PyTorch 时可安装适配硬件的版本，本机 CUDA 轮子源为：

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

运行需要以下资源：

| 资源 | 本机位置或配置 |
|---|---|
| MediaPipe 姿态模型 | 默认 `video_motion/assets/pose_landmarker_full.task`，setup 自动下载；缺少时回退 `lite` |
| SMPL-X v1.1 权重 | `E:\TEST_RECORD\G1_Project\G1\models\SMPL\models_smplx_v1_1\models\smplx`，含 neutral/male/female；`auto` 后端优先使用 |
| G1 MJCF 和网格 | `E:\TEST_RECORD\G1_Project\G1\assets\mujoco_menagerie\unitree_g1` |
| 腿部 RL 权重 | `E:\TEST_RECORD\G1_Project\G1\assets\motion.pt` |

换机器时可通过 `G1_POSE_MODEL` 指定姿态模型，或用 `G1_POSE_QUALITY=lite/full/heavy` 选择精度；`python -m video_motion extract --quality ...` 也可逐次指定。SMPL-X 目录可用 `G1_SMPLX_MODEL_DIR` 指定，也可以直接指向 `E:\TEST_RECORD\G1_Project\G1\models\SMPL` 根目录；性别缺失时自动回退。`--pose-backend auto` 默认优先 SMPL-X，失败回退 MediaPipe；追求速度可用 `--pose-backend mediapipe`。机器人资源定位规则见 `g1_walkfree/resources.py` 和 `g1_walkfree/config.py`。视频处理主要使用 CPU；Full 模型比 Lite 更准但更慢，低配机器可用 `lite`。SMPL-X 拟合在 CUDA 可用时自动使用 GPU。Warp、MJX 和 Isaac 不属于此回放流程的必装依赖；AMASS 是可选训练数据，不参与视频回放。

聊天密钥读取环境变量 `DEEPSEEK_API_KEY` 或项目目录 `api_key.txt`。原 Start 中的密钥已迁入本地文件，启动代码不再硬编码密钥，`.gitignore` 排除了该文件。

## 4. 方向修正及坐标约定

统一使用 G1 躯干的右手坐标系：

| 方向/控制量 | 约定 |
|---|---|
| `+X` | 机器人正前方，与正向行走命令一致 |
| `+Y` | 机器人左侧 |
| `+Z` | 上方 |
| 正 `vx` | 沿机器人当前朝向前进 |
| 负 `shoulder_pitch` | 手臂向前抬起 |
| 正 `elbow` | 屈肘 |

人体局部坐标由肩和髋构建：左方向由右肩指向左肩，上方向由髋中心指向肩中心；**前方向 = 左方向 × 上方向**，随后重新正交化上方向。

旧代码误用“上 × 左”，构造了左手基底，将前后方向镜像；旧测试也沿用了同一个错误的样本假设。本次同时修正转换与测试。新的测试直接用 G1 模型计算双腕相对肩的坐标，确认人体前伸时腕部位于机器人 `+X`，且肩俯仰角为负；还检查相机旋转和整体平移不改变人体局部动作。

不应通过调换左右手或把所有关节角取反来修复前后问题。已有 npz 保存的是原始骨架，重新回放会自动使用新转换，无需因本次方向修正而重新提取。旧 report 是历史结果，应重新生成。

## 5. 视频示例与输出

`vedio` 按当前项目命名保留：

| 文件 | 用途 |
|---|---|
| `example.mp4` | TensorFlow tfjs-models 官方动态姿态测试视频，约4秒、96 KB |
| `source.txt` | 原始来源链接和仓库授权信息 |
| `example.npz` | 骨架与可见度、时间戳 |
| `example.report.json` | 最近一次 Start 回放报告 |
| `README.md` | 指向本文的导航 |

视频来源：<https://github.com/tensorflow/tfjs-models/blob/master/pose-detection/test_data/pose_1.mp4>，源仓库采用 Apache-2.0。

Start 输出动作到 `vedio/<视频文件名>.npz`，报告输出到同名 `.report.json`。重复使用同名视频会覆盖输出；若需要保存多个版本，使用下文分步命令指定不同输出名。

建议使用单人、肩肘腕和髋部清晰、无遮挡的短视频。多人场景只追踪一个人，不保证身份连续性。完全没有合格骨架时明确报错；低置信度帧保留上一姿态，恢复后限制目标变化速度。默认使用 MediaPipe Full 模型，并对世界关键点做置信度感知插值和 Savitzky-Golay 时序平滑。`--pose-backend auto` 会继续用 `E:\TEST_RECORD\G1_Project\G1\models\SMPL` 的 SMPL-X 权重做逐帧运动学拟合，输出 `smplx_joints/global_orient/body_pose/transl` 等字段；失败时回退 MediaPipe。提取文件仍保留旧的 `points/visibility` 格式，并新增 `pose_model/source_fps/sample_fps/stabilized/backend` 字段，旧 npz 仍可直接回放。MediaPipe 检测默认在独立工作进程运行，避免 Windows 下关闭检测器阻塞约 40 秒。

## 6. 分步处理与工具调用

仅提取动作：

```powershell
python -m video_motion extract "D:\视频\挥手.mp4" --output motions/wave.npz --fps 15 --max-seconds 10
```

重复回放，无需再次运行 MediaPipe：

```powershell
python -m video_motion play motions/wave.npz --viewer --speed 0.7
python -m video_motion play motions/wave.npz --gif motions/wave.gif
```

DeepSeek 通过 `RobotToolkit` 自动获得以下工具：

| 工具 | 功能 |
|---|---|
| `robot_learn_video` | 从本地视频提取骨架，保存动作，默认立即回放 |
| `robot_play_motion` | 播放已有 npz，可指定回放倍率 |
| 原有9个机器人工具 | 状态查询、行走、导航、转向、手臂、腰部、动作序列、停止和复位 |
| `robot_coordination_start/status` | 独立进程训练质心网络或执行保守小跳 |
| `robot_spatial_skill_list/get/run` | 空间绝对关键点技能库查询与运行 |
| `robot_train_video` 等训练工具 | 后台视频训练、状态查询、停止、渲染、GIF导出 |
| `robot_proactive_report` | 训练计时报告、按分钟间隔主动发送、立即发送 |
| `robot_docker_train/status/summary/export_video/stop` | 指定目录的 Docker 后台训练、总结与视频导出 |

聊天示例：

```text
学习本地视频 vedio/example.mp4 前8秒的上半身动作并回放，告诉我是否摔倒。
```

工具返回真实执行结果。命令行视频模式无需联网调用 DeepSeek；首次姿态模型下载需要网络。

## 7. 程序结构与执行流程

```text
Start.py                   统一启动、菜单、IDE路径处理
G1Robot.pyproj             Visual Studio 项目
robot_tools.py             机器人工具和视频工具分发
robot_bridge.py            线程安全机器人会话
deepseek_robot.py         DeepSeek 对话与 Function Calling 循环
video_motion/
  pose.py                  模型下载、Tasks视频推理、npz校验
  smplx_fit.py             E:\TEST_RECORD\G1_Project\G1\models\SMPL 权重发现、SMPL-X 逐帧拟合与关节映射
  mediapipe_worker.py      独立 MediaPipe 检测进程（规避 Windows close 阻塞）
  mediapipe_backend.py     逐帧检测与结果落盘
  retarget.py              右手坐标转换、肩肘IK、平滑回放、位置保持
  tools.py                 视频工具schema
  __main__.py              setup/extract/play/learn 命令
g1_walkfree/              模型与策略定位、控制器、原有界面
fullbody/                  全身参考重定向、PPO训练、评估、空间技能库
  global_track.py          接触约束全局根轨迹（WHAM-lite）
  refine.py                足底支撑/质心修正与接触复核
  training_viewer.py       训练实时渲染窗口（Tk + MuJoCo Renderer）
coordination/              关键节点-质心协调网络与独立跳跃
skills/                    节点技能库 registry.json + 空间技能库 spatial_registry.json
app_logging.py            日志（RotatingFileHandler，回退LocalAppData）
runtime_io.py             Windows安全写入（原子写+重试）
proactive.py              训练计时、主动训练报告与报告落盘
environment_check.py      只读环境诊断（不泄密钥）
Start_G1.bat              Windows双击启动脚本
vedio/                     示例与Start输出
requirements-video.txt     视频和仿真依赖
tests/test_video_motion.py  方向、IK、文件校验、仿真回归
examples/smoke_video_motion.py  静态人物短片的完整链路检查
```

流程为：视频解码 → MediaPipe Full 世界骨架（命令行和训练状态实时显示百分比）→ 置信度插值/时序平滑 → SMPL-X 运动学拟合（auto/smplx，逐帧进度）→ 人体躯干坐标 → G1 肩肘约束 IK → 角速度限制 → 仿真时间插值 → 位置与朝向保持 → 结果报告。后台训练同时记录开始时间、已用时间、纯训练用时、总耗时和预计剩余；训练启动后默认每10分钟主动发送报告，保存到任务目录的 `proactive_reports/`。全身链路还会估计支撑脚接触、全局根轨迹并做足底支撑修正，详见 [视频识别与全局轨迹升级](docs/VIDEO_TRACKING_UPGRADE.md)。

IK 使用独立 MjData，不修改运行中的仿真状态。回放锁定会话并逐步推进仿真，避免后台快跑跳过视频帧；初始姿态用0.6秒过渡。回放结束清零速度命令，空闲时暂停仿真；摔倒则中止回放。

## 8. 验证结果与报告解释

验证命令：

```powershell
python -m unittest tests.test_video_motion -v
python Start.py --mode video --headless --seconds 8
```

2026-10-17 方向修正后实测：

| 验证 | 结果 |
|---|---|
| 回归测试 | 4项通过 |
| 前伸方向 | 双腕沿G1躯干 +X，肩俯仰负值 |
| 官方动态示例 | 40帧，回放4.502仿真秒 |
| 完成/摔倒 | 完成，未摔倒 |
| 最大平面漂移 | 0.215m |
| 最大关节跟踪误差 | 0.0739rad |
| 低置信度帧 | 0 |

报告中的 `max_joint_error_rad` 比较实际关节角与重定向目标角，不是视频视觉相似度；`max_drift_m` 是相对回放起点的最大平面偏移。位置闭环只能减小漂移，不能保证完全原地或任意动作稳定。当前未自动操作 Visual Studio GUI，已通过与其相同的 Start 入口完成无窗口端到端验证。

## 9. 常见问题

- **手臂仍看起来反向**：确认运行当前目录的代码，执行方向回归测试；不要仅凭摄像机屏幕左右判断机器人前后。正面镜像视频会改变左右语义，可先使用官方示例排查。
- **没有检测到清晰人体**：换用肩肘腕及髋部无遮挡的单人全身视频。侧身、截断上半身、多人重叠会降低置信度。
- **机器人漂移或摔倒**：优先缩短视频、降低回放倍率和动作幅度；当前腿部策略没有接受视频动作条件训练。
- **找不到模块或资源**：确认 Visual Studio 使用已安装依赖的解释器，运行 `python -m video_motion setup`，按提示配置模型和权重路径。
- **退出仍有窗口**：视频回放结束会等待关闭三维窗口，这是为了保留观察结果；`--headless` 模式会直接退出。

全身参考重定向、训练环境与PPO已扩展，见 [FULLBODY_TRAINING.md](docs/FULLBODY_TRAINING.md)。当前采用就地根部参考，未恢复真实全局位移和手腕旋转，尚未训练出稳定完成示例的策略。
