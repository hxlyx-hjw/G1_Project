# 双视角视频三维动作提取

新增链路：

```text
视频 A + 视频 B
  → 时间同步
  → MediaPipe 2D 图像关键点（两路）
  → DLT 三角化
  → 短时空洞时序自修正
  → 关节尺度归一化
  → 可选 SMPL-X 多视角拟合
  → 现有 G1 全身 retarget / PPO 训练
```

输出仍为 `times/points/visibility` 的 33 点格式，不改变现有
`fullbody.reference`、IK 和训练接口。

## 快速运行

两台相机应尽量固定、同步开机，并让同一个人完整出现在两路视频中。

```powershell
cd E:\TEST_RECORD\G1_Project\G1\G1_LLM_Robot\G1_LLM_Robot

python -m video_motion multiview `
  "D:\videos\view_a.mp4" `
  "D:\videos\view_b.mp4" `
  --output "motions\twoview\pose.npz" `
  --fps 30 --max-seconds 30 `
  --fov 60 --baseline 2.0 --azimuth-b 90 `
  --backend auto
```

## 第二类：文件夹 + list

DeepSeek 新增：

- `robot_multiview_pairs`：读取 list 并返回 a/b 配对
- `robot_multiview_train`：双视角提取、重定向和训练；`fallback=true` 时失败自动降级

文件夹示例：

```text
D:\two_view_action\
  list.txt
  a\001.mp4
  b\001.mp4
```

`list.txt` 支持：

```text
# 推荐：明确两列
a/001.mp4,b/001.mp4
a/002.mp4,b/002.mp4

# 也支持单列 a，自动寻找 b/001.mp4
a/001.mp4

# 也支持空格分隔
a/001.mp4 b/001.mp4
```

也可以使用文件名 `a.mp4`、`b.mp4`，或 `front`/`frontal`/`forward` 与
`side`/`lateral` 配对。一个文件夹有多对时，DeepSeek 可用 `pair` 选择
key 或索引。

命令行等价流程：

```powershell
python -m video_motion multiview `
  "D:\two_view_action\a\001.mp4" `
  "D:\two_view_action\b\001.mp4" `
  --output motions/twoview/pose.npz
```

DeepSeek 工具调用顺序：

```text
robot_multiview_pairs(folder="D:\two_view_action")
→ 询问 RL-PD 因子和手/脚范围
→ robot_multiview_train(folder="D:\two_view_action",
                        pair="001",
                        fallback=true,
                        rl_pd_influence=0.7,
                        hand_rl_pd_range=[0.2,0.6],
                        foot_rl_pd_range=[0.5,0.9])
→ robot_training_status(job_id=...)
```

第一类单视频工具 `robot_learn_video`、`robot_train_video` 保持不变；
第二类失败时才降级到 a 视角单目，不会把单目结果伪装成双视角。

然后走现有重定向和训练：

```powershell
python -m fullbody retarget `
  "motions\twoview\pose.npz" `
  --output "motions\twoview\reference.npz" `
  --root-mode contact_grounded

python -m fullbody.train_balanced `
  "motions\twoview\reference.npz" `
  --output "runs\twoview" `
  --iterations 500 --envs 4 --device cuda `
  --rl-pd-influence 0.7 `
  --hand-rl-pd-range 0.2 0.6 `
  --foot-rl-pd-range 0.5 0.9
```

## 相机标定 JSON

有标定参数时强烈建议使用标定模式，否则只能用正交相机和基线做近似。

```json
{
  "camera_a": {
    "K": [[900, 0, 960], [0, 900, 540], [0, 0, 1]],
    "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "t": [0, 0, 0],
    "dist": [0, 0, 0, 0, 0]
  },
  "camera_b": {
    "K": [[900, 0, 960], [0, 900, 540], [0, 0, 1]],
    "R": [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
    "t": [0, 0, 2.0],
    "dist": [0, 0, 0, 0, 0]
  }
}
```

`K/R/t` 使用 OpenCV 相机坐标约定（相机看向 `+Z`，像素 `x` 向右、
`y` 向下）；`P = K[R|t]`。也可以直接为每个相机提供 `P`。

```powershell
python -m video_motion multiview A.mp4 B.mp4 `
  --calibration cameras.json --output motions/twoview/pose.npz
```

## 同步

共同时间轴以两路视频中较短的**有效时长**为主：

- 先读取两路视频的实际时长；
- 只扫描较短视频所需的时长，避免长视频做无效提取；
- 再在有效重叠区间内重采样；
- `multiview_report.json` 中的 `timeline.policy` 固定为
  `shortest_common_overlap`，并记录哪一路决定结束时间。

默认认为两路视频同时开始。若存在固定时间差：

```powershell
--offset-a 0.0 --offset-b -0.2
```

两个时间轴会重采样到共同的重叠区间。当前不自动分析音频或闪光同步，
建议拍摄时使用同一拍手/闪光作为人工起点。

两路动作时序不一致时，默认还会自动搜索时间偏移：

```text
auto_sync = true
max_sync_offset_seconds = 12.0
sync_step_seconds = 0.5
```

自动同步不是看画面相似度，而是对候选偏移做抽帧三角化，按有效关节比例
和共同时长覆盖率评分。只有明显优于原偏移时才采用。可在
`multiview_report.json` 的 `sync_search` 中查看：

```json
{
  "applied": true,
  "offset_b": -9.0,
  "baseline_valid_fraction": 0.0028,
  "best_valid_fraction": 0.61,
  "step_seconds": 0.5
}
```

关闭自动同步：

```powershell
--no-auto-sync
```

扩大搜索范围或细化步长：

```powershell
--max-sync-offset-seconds 20 --sync-step-seconds 0.25
```

三角化后还会检查有效关节比例，默认：

```text
min_valid_joint_fraction = 0.05
```

低于 5% 时会明确报错并触发 `fallback=true` 的单目降级，而不是把低质量
三维点送入训练。确认机位和标定可用后，可显式调低阈值；不建议在生产
训练中设为 0。

## 动作误差与自修正

双视角三角化在遮挡、低置信或重投影误差过大时会产生短时缺失。默认开启
有上限的时序修复：

```text
self_correct = true
repair_gap_seconds = 0.25
```

只修复被可靠帧包围、且间隙不超过 `repair_gap_seconds` 的关节空洞；
前/后边缘的短空洞也只在同样上限内补齐。修复后的关节可见度最低标为
0.55，并在 `corrected_joints` 中标记。超过上限的遮挡保持缺失，由后续
重定向沿用上一姿态，避免凭空生成大幅动作。

关闭自修正：

```powershell
python -m video_motion multiview A.mp4 B.mp4 `
  --output motions/twoview/pose.npz `
  --no-self-correct
```

调整可修复间隙：

```powershell
python -m video_motion multiview A.mp4 B.mp4 `
  --output motions/twoview/pose.npz `
  --repair-gap-seconds 0.35
```

`multiview_report.json` 会包含：

```json
{
  "timeline": {
    "policy": "shortest_common_overlap",
    "common_duration_seconds": 8.4,
    "shorter_source": "b"
  },
  "self_correction": {
    "corrected_frames": 12,
    "corrected_fraction": 0.018
  }
}
```

## 无标定模式

不提供 `--calibration` 时：

- 用 `--fov` 估计两路相机内参；
- 用 `--baseline` 和 `--azimuth-b`（默认 90°）构造正交相机对；
- DLT 得到的绝对尺度不可信，最后按 `--target-leg-length`（默认 0.85 m）
  做人体尺度归一化。

无标定模式依赖“相机确实接近正交、主体居中、镜头畸变较小”的假设。

## 后端选择

- `--backend dlt`：只做 2D + DLT，速度快。
- `--backend smplx`：DLT 后强制做 SMPL-X 多视角三维拟合。
- `--backend auto`：优先 SMPL-X，失败回退 DLT。

SMPL-X 使用现有 `G1_SMPLX_MODEL_DIR` 或 `D:\SMPL` 权重。Docker 内需要
重建镜像后运行，并挂载两路视频和 `motions` 输出目录。
