# 视频提取 → SuperMimic 训练：协作与验收记录

- 日期：2026-11-04
- 分工：codex 负责 SuperMimic 侧（`learn_video.py`、`supermimic_train.sh`、`live_motion_viewer.py`、`train.py`）；Claude 负责 G1_Project 接入层（`G1_DeepSeek_Extensions`）的审查、加固与测试。
- 边界：Claude 本轮**未修改** `E:\TEST_RECORD\SuperMimic` 下任何文件，避免与 codex 并发写入冲突。

## 1. 端到端验收（Claude 独立复跑）

通过 `DeepSeekIntegration().dispatch("robot_supermimic_learn_video", ...)` 启动，未直接调用 SuperMimic 脚本：

| 项目 | 结果 |
| --- | --- |
| 成功任务 | `sm-video_20261104_215610_5c91cedb` |
| 输入 | `G1\vedio_self\test4.mp4`（2 秒，single，lite，mediapipe） |
| 阶段 | `extract → retarget → convert → train → done` |
| 产物 | `material/pose.npz`、`material/reference.npz`、`motion/test4.csv`(36 列/30 Hz)、`training/ckpt_*.pt`、`training/robot_contract.json`、`pipeline.json` |
| 状态 | job `completed`，returncode 0，`training_success=true`，`progress=1.0` |
| 训练指标 | `training_metrics.iteration` 可经 `job_status` 读取 |

失败路径也已验证：`max_seconds=1.0` 时仅 31 帧，姿态提取抛出
`ValueError: 未检测到清晰的肩、肘、腕及髋部...`；`pipeline.json.state=failed`，
`job.log` 含完整 traceback。即 **失败可诊断，成功无日志**（见 3.1）。

## 2. Claude 本轮修改（G1_Project）

文件：`G1_DeepSeek_Extensions/g1_deepseek_ext/adapters.py`

1. `learn_video` 现在在启动任务**之前**校验 `root_mode`、`min_visibility`、
   `smplx_gender`、`smplx_device`，非法值直接返回 `IntegrationError`，
   不再先建目录、起进程后才由 argparse 报错。
2. `smplx_model_dir` 走 `_resolve_video_input` 做存在性与目录白名单校验，
   并规范化后传给 `learn_video.py`。
3. `robot_supermimic_learn_video` 的 tool schema 补上
   `smplx_model_dir` / `smplx_gender` / `smplx_device`。此前适配器支持、
   但 DeepSeek 模型无法传参，等于 SMPL-X 后端不可用。
4. 返回值新增 `note`：`smoke` 或 `iterations < 1000` 时明确说明
   “只验证链路、不是可用策略”，并给出真实训练入口，避免模型把烟测说成训练完成。
5. `job_status` 现在把 `pipeline` 的 `stage`、`message`、`progress`、
   `training_metrics` 提升到结果顶层；`pipeline.state=failed` 时
   `success=false`。此前调用方必须自己钻进 `pipeline` 才能看到阶段。

文件：`G1_DeepSeek_Extensions/tests/test_integration.py`

文件：`G1_DeepSeek_Extensions/README.md`（补充 smoke/paper 语义与 SMPL-X 参数说明）

新增 3 个测试（连同 codex 的日志测试，共 15 个，全部通过）：

- schema 暴露 SMPL-X 参数；
- 非法参数在启动前被拒绝且不创建任务目录；
- `job_status` 正确暴露 pipeline 阶段/进度/指标。

验证命令：

```powershell
cd E:\TEST_RECORD\G1_Project\G1_DeepSeek_Extensions
python -m unittest discover -s tests -v   # Ran 15 tests ... OK
```

## 3. 建议 codex 处理（均在 SuperMimic 侧）

### 3.1 成功任务 `job.log` 为空

`learn_video.py` 成功路径不向 stdout 写任何内容，`JobManager` 的日志尾部和
G1 控制台因此看不到任何进展；只有崩溃时才有 traceback。建议在
`progress.update()` 同时向 stdout 打一行带时间戳的阶段信息并 flush，
例如 `[learn_video] retarget 0.45`。这会直接改善 DeepSeek 工具结果的可读性。

### 3.2 训练进度长期停在 0.65

整个训练期间 `pipeline.progress` 固定 0.65，`message` 固定“训练运行中”。
建议轮询 `train_log.csv` 的 `iteration` 与 `--iterations` 计算
`0.65 → 0.95` 的进度，并写入 `training_metrics`。

### 3.3 可配置路径被脚本硬编码

`scripts/supermimic_train.sh` 里：

```bash
REPRO=/mnt/e/TEST_RECORD/SuperMimic
VENV=/mnt/d/G1_Envs/SuperMimic
```

`IntegrationConfig.supermimic_venv` 和 `G1_DEEPSEEK_SUPERMIMIC_VENV` 实际
传不到训练脚本，`G1_DEEPSEEK_SUPERMIMIC_ROOT` 移动安装后也会断。建议改为
`${SUPERMIMIC_VENV:-/mnt/d/G1_Envs/SuperMimic}` 与
`${SUPERMIMIC_REPRO:-/mnt/e/TEST_RECORD/SuperMimic}`，并由
`learn_video.py` 显式 export。

### 3.4 其它小项

- `pipeline.json.viewer_pid` 在 viewer 退出后仍保留，容易误判“渲染仍在运行”；
  结束时写 `viewer_running=false` 是对的，但建议同时清空或标注 `viewer_exit_code`。
- 极短片段（`max_seconds < 2`）会在姿态提取阶段失败；若希望工具更友好，
  可在 `learn_video.py` 入口给出“建议 ≥2 秒”的明确错误，或由适配器前置提示。
- `--pair` 对单文件输入被静默忽略；可在 `learn_video.py` 里显式拒绝。

## 4. 实时日志读取（`robot_supermimic_logs`）

codex 先实现了 `robot_supermimic_logs`（训练迭代/百分比/ETA/指标历史/日志尾部）。
Claude 在其基础上补齐了以下能力，未新增重复工具：

1. **增量轮询**：新增 `cursor` 参数；首次调用返回 `next_cursor`，后续把
   `next_cursor` 作为 `cursor` 传回，只返回新增的 `training_log_delta`，
   避免每次轮询重复返回整段日志。日志按字节偏移读取，CRLF/ANSI 已归一化。
2. **两种目录布局**：流水线任务用 `output_root/training/{train_log.csv,args.json}`，
   独立 `robot_supermimic_train` 任务用 `output_root/{train_log.csv,args.json}`，
   训练 stdout 在后者是 `job.log`。工具现在会自动回退，并用
   `stdout_source` 标明来源。
3. **训练感知的总体进度**：流水线任务把训练区间映射到 `0.65→0.95`；
   独立训练任务直接用 `iteration/iterations`。新增 `summary` 人类可读摘要。
   （此前的 `progress` 在独立训练任务中为空，轮询看不到百分比。）

实测轮询 `sm-train_20261104_225408_a39ba761`（400 迭代）：
迭代从 0 增长到 16，`overall_progress` 0.04，`training_log_delta` 返回 2289 字节
新增日志，证明实时读取可用。

## 5. 复现命令

```powershell
# 单视频 → 提取 → 重定向 → CSV → 1 次迭代烟测（无渲染窗口）
cd E:\TEST_RECORD\G1_Project\G1_DeepSeek_Extensions
python -c "from g1_deepseek_ext import DeepSeekIntegration; print(DeepSeekIntegration().dispatch('robot_supermimic_learn_video', {'input': r'E:\TEST_RECORD\G1_Project\G1\vedio_self\test4.mp4', 'view_mode':'single', 'max_seconds':2, 'quality':'lite', 'pose_backend':'mediapipe', 'iterations':1, 'num_envs':1, 'mode':'smoke', 'viewer': False}))"

# 查询
python -c "from g1_deepseek_ext import DeepSeekIntegration; print(DeepSeekIntegration().dispatch('robot_supermimic_job_status', {'job_id':'<上一步 job_id>'}))"

# 实时日志/进度；再次调用时把上次 next_cursor 作为 cursor 传入
python -c "from g1_deepseek_ext import DeepSeekIntegration; print(DeepSeekIntegration().dispatch('robot_supermimic_logs', {'job_id':'<job_id>', 'lines':50}))"
```
