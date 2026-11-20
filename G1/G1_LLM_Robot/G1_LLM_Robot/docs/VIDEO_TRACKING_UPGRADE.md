# 视频识别与全局轨迹升级

本页记录 `video_motion` 和 `fullbody` 参考生成链路的升级。目标是借鉴
`G1_视频模仿学习方案逻辑.md` 中的 SMPL-X / WHAM 思路，同时不破坏原有
188 维观测、29 维动作、平衡先验、关节辅助网络、腾空支撑凸包和手臂稳定
优先逻辑。

## 已实现

- **SMPL-X 运动学拟合**：新增 `video_motion/smplx_fit.py`，使用
  `E:\TEST_RECORD\G1_Project\G1\models\SMPL\models_smplx_v1_1\models\smplx` 下的 v1.1 权重，把 MediaPipe
  世界关键点作为观测，逐帧优化 SMPL-X 的 `global_orient/body_pose/transl`，
  输出 127 关节及完整参数。`pose.extract(backend='auto')` 默认优先 SMPL-X，
  失败时回退 MediaPipe；`backend='smplx'` 会明确报错而不是静默降级。
  该拟合使用 SMPL-X 运动学先验，但不是 4D-Humans/WHAM 的图像级回归，
  `betas` 目前固定为 0。
- **姿态模型升级**：默认从 MediaPipe PoseLandmarker Lite 切换为 Full，支持
  `lite/full/heavy`；模型缺失时自动下载，下载失败时回退已有 Lite。可用
  `G1_POSE_QUALITY`、`--pose-quality` 或 `video_motion extract --quality` 选择。
- **时序稳定**：对 33 个世界关键点做置信度感知插值与 Savitzky-Golay 平滑，
  降低单目深度抖动、单帧漏检和遮挡恢复时的目标跳变。原始 `points/visibility`
  仍按旧格式保存，新增 `pose_model/source_fps/sample_fps/stabilized` 元数据。
- **WHAM-lite 全局轨迹**：`fullbody/global_track.py` 利用支撑脚在世界中应保持
  不动的约束，从足端相对髋部的位置反推根节点水平位移、根高和相对首帧的转向；
  同时输出左右脚接触标签。它恢复的是接触约束下的近似轨迹，不是完整 WHAM 的
  学习式世界坐标回归。
- **参考重定向**：`fullbody/reference.py` 支持
  `root_mode=in_place/contact_grounded/auto`。默认全身任务使用
  `contact_grounded`：保留水平根轨迹，写入朝向四元数，并优先使用全局轨迹的
  接触标签。`in_place` 仍保留为兼容模式。
- **支撑修正**：`fullbody/refine.py` 使用稳健 IK、脚底水平约束、支撑脚/质心
  约束和根轨迹平滑；接触帧按机器人足底几何确定根高，离地帧在相邻接触帧之间
  插值，不再无条件把所有帧压回地面。
- **训练环境**：`fullbody/env.py` 优先读取参考文件中的真实接触标签，评估信息
  新增接触匹配、腾空比例、关键点 RMSE 和参考根位移；原有奖励与网络结构不变。
- **检测稳定性**：Windows 下 `PoseLandmarker.close()` 可能阻塞约 40 秒。
  现在默认在独立工作进程内完成检测，写盘后在对象析构前直接退出，实测
  19 帧从 43 秒降到约 1.2 秒。可用 `G1_MEDIAPIPE_INPROCESS=1` 回退进程内模式。
- **提取进度**：MediaPipe 子进程通过 `G1_PROGRESS current total` 行实时回传
  骨架检测进度；SMPL-X 拟合继续使用逐帧回调。CLI 显示百分比，后台训练把
  阶段/帧数/百分比写入 `status.json` 的 `extraction` 字段。
- **渲染稳定性**：修复 GIF 录制直接保存 `mujoco.Renderer` 复用缓冲导致的
  重复帧/花屏；训练监视窗口改用长周期质量自适应，减少 OpenGL 渲染器反复重建。

## 使用方式

```powershell
# 主菜单/Start 入口：默认 Full + 接触约束全局轨迹 + refine
python Start.py --mode fullbody-retarget --video "D:\视频\动作.mp4" --reference motions/my_reference.npz

# 只提取姿态，指定精度
python -m video_motion extract "D:\视频\动作.mp4" --output motions/wave.npz --quality full

# 显式使用 SMPL-X（默认 auto 已优先尝试）
python -m video_motion extract "D:\视频\动作.mp4" --output motions/wave.npz --backend smplx

# 只用 MediaPipe，速度优先
python -m video_motion extract "D:\视频\动作.mp4" --output motions/wave.npz --backend mediapipe

# 只生成全身参考；默认已经 refine
python -m fullbody retarget motions/wave.npz --output motions/fullbody/reference.npz

# 回退旧行为：就地参考、不做支撑修正
python -m fullbody retarget motions/wave.npz --output motions/fullbody/reference.npz --root-mode in_place --no-refine
```

环境变量：

- `G1_POSE_QUALITY=lite|full|heavy`：默认 `full`。
- `G1_POSE_MODEL=<path>`：显式指定 `.task` 模型文件。
- `G1_SMPLX_MODEL_DIR=<path>`：显式指定 SMPL-X 的 `smplx` 目录，也可以直接
  指向 `E:\TEST_RECORD\G1_Project\G1\models\SMPL` 根目录。默认自动搜索
  `E:\TEST_RECORD\G1_Project\G1\models\SMPL\models_smplx_v1_1\models\smplx`、`models/smplx` 和 `~/.smplx`。
- `G1_MEDIAPIPE_INPROCESS=1`：回退到进程内 MediaPipe 检测；仅用于调试，
  退出时可能重新出现长时间阻塞。

SMPL-X 命令行参数：

- `--smplx-gender neutral|male|female`：默认 `neutral`；若该性别权重缺失，
  自动回退到已下载的其它权重。
- `--smplx-device auto|cpu|cuda`：默认 `auto`，本机 CUDA 可用时使用 GPU。
- `--smplx-iterations N`：每帧优化迭代数，默认 60；降低到 30 可明显加速，
  但关节拟合误差会略增。

## 验证

- `python -m unittest discover -s tests -v`：包含 SMPL-X 配置与映射回归。
- 官方 `vedio/example.mp4` 端到端验证：Full 模型提取 60 帧，生成
  `contact_grounded` 参考并 refine，随后构建 `ImitationEnv` 连续步进 20 次无异常；
  评估信息包含 `reference_root_mode`、`reference_contact_left/right` 和
  `contact_match`。
- SMPL-X 端到端验证：19 帧、60 次迭代，平均关节误差约 0.055 m，
  `smplx_joints` 形状 `(19,127,3)`，`contact_grounded + refine` 后
  `ImitationEnv` 连续步进 20 次奖励有限。

## 限制

- 当前实现是 **MediaPipe 关键点 + SMPL-X 运动学拟合**，不是 4D-Humans/WHAM
  的图像级 SMPL-X 重建，也没有学习式世界坐标回归。`betas` 固定为 0，
  对相机运动、脚部遮挡和单目尺度误差仍敏感。
- 空中阶段没有可靠的绝对高度观测，根高主要靠接触帧和相邻帧插值；不能把该轨迹
  当作真实跳跃弹道。跳跃高度/滞空仍应由显式跳跃模板或额外高度先验提供。
- 参考动作的物理可执行性仍受 IK、关节限位和简化碰撞模型限制；训练通过不等于
  实机可部署。
