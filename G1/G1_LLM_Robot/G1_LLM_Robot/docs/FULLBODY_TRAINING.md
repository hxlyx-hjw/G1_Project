# 全身重定向与模仿训练

本模块把视频肩肘重定向扩展为全身参考动作、29维残差动作的物理训练环境及PyTorch PPO。可以生成参考、查看运动学动作、实际更新策略权重、恢复训练和进行物理评估。视频姿态默认优先使用 `E:\TEST_RECORD\G1_Project\G1\models\SMPL` 的 SMPL-X v1.1 权重做运动学拟合，失败时回退 MediaPipe。**训练链路已经跑通，官方视频上的短训练策略尚未稳定完成动作。**

## 从 Start.py 开始

Visual Studio 打开 `G1Robot.pyproj`，以 `Start.py` 启动。菜单新增：3全身重定向、4运动学预览、5训练、6物理评估。第一次先选3生成参考，再选4检查动作，之后选5训练、选6评估。训练默认100轮，每轮256个控制步；菜单6自动加载默认输出目录里的 `policy.pt`，没有权重时明确使用PD基线。

命令行完整流程：

```powershell
# 提取视频并全身重定向；默认用vedio/example.mp4
python Start.py --mode fullbody-retarget --seconds 8 --fps 10 --root-mode contact_grounded

# 运动学预览：直接设置参考姿态，没有动力学稳定性含义
python Start.py --mode fullbody-preview

# 真正的PPO模仿训练
python Start.py --mode fullbody-train --iterations 100 --steps 256

# 使用训练权重进行物理回放；打开三维窗口
python Start.py --mode fullbody-eval

# 无窗口评估
python Start.py --mode fullbody-eval --headless
```

自定义视频、参考及输出：

```powershell
python Start.py --mode fullbody-retarget --video "D:\视频\动作.mp4" --reference motions/my_reference.npz --seconds 10 --root-mode contact_grounded
python Start.py --mode fullbody-train --reference motions/my_reference.npz --output runs/my_policy --iterations 500
python Start.py --mode fullbody-train --reference motions/my_reference.npz --output runs/my_policy --resume runs/my_policy/policy.pt --iterations 100
python Start.py --mode fullbody-eval --reference motions/my_reference.npz --output runs/my_policy --headless
```

`--iterations` 是本次增加的训练轮数。恢复会加载策略、优化器和迭代计数，环境与随机采样重新初始化，不保证逐位复现中断位置。

## 环境

沿用现有 MuJoCo、NumPy、SciPy、PyTorch；新增直接依赖 `gymnasium`：

```powershell
python -m pip install -r requirements-fullbody.txt
python -m video_motion setup
```

视频提取另外需要 `smplx` 包和 SMPL-X 权重。`video_motion setup` 会显示实际
发现的模型目录、默认性别和 CUDA 可用性。默认搜索
`E:\TEST_RECORD\G1_Project\G1\models\SMPL\models_smplx_v1_1\models\smplx`；也可设置
`G1_SMPLX_MODEL_DIR=E:\TEST_RECORD\G1_Project\G1\models\SMPL`。只用 MediaPipe 时加 `--pose-backend mediapipe`。

不要求 Isaac、Linux 或 stable-baselines3。默认单环境CPU物理步进，网络也使用CPU；`--device cuda` 仅把网络移到CUDA，MuJoCo物理仍在CPU，单环境时不保证更快。当前没有实现Warp批量环境或多进程采样。

## 全身重定向范围

- 12个腿关节：通过髋—膝、膝—踝方向以及脚跟—脚尖朝向约束拟合。
- 3个腰关节：拟合髋部坐标系中的肩部朝向。
- 8个肩肘关节：拟合上臂、前臂方向。
- 6个手腕关节：参考动作保留默认值；训练动作仍覆盖全部29关节。
- 浮动根部：朝向由肩髋平面恢复；默认 `root_mode=contact_grounded` 用支撑脚接触约束恢复水平位移和转向，接触帧由脚底几何确定根高，离地帧在相邻接触帧之间插值。它不是完整 WHAM：单目相机运动、深度噪声和脚部遮挡仍可能被误判为根部移动。
- 兼容旧行为：`--root-mode in_place` 或 `--no-refine` 可关闭全局轨迹/支撑修正；旧参考文件和旧 checkpoint 仍可加载。

使用右手坐标系：`+X前、+Y左、+Z上`，`前=左×上`。这是与上半身方向修正一致的映射。

每帧采用带关节限位、稳健损失和时序正则的数值IK。肩髋膝踝低置信度帧保持上一参考；若没有任何完整全身帧则报错。`refine.py` 会加入脚底水平、支撑脚与质心约束，并保留接触帧的根高；但仍没有动力学可行性求解、足部落点规划或地面穿透全身修正，因此IK参考不保证适合物理执行。默认接触标签来自足端接触约束估计，refine 后再按脚底几何复核；它不是动捕接触标签。单目深度、遮挡、足部朝向估计和相机运动仍会带来误差。

## 模型和控制

`fullbody/model.py` 读取原始29自由度G1模型，在输出目录独立生成全力矩模型；不依赖原有12自由度 `motion.pt`，也不会覆盖旧上半身回放模型。保留足部球接触，禁用其余网格碰撞以简化初始训练。这意味着自碰撞和跌倒后的身体地面接触并不真实，策略不能直接部署到实机。

物理频率500Hz，策略频率50Hz。纯模仿环境使用 `参考角 + 0.25 × 动作` 作为PD目标；平衡先验环境则组合为 `默认站姿 + 0.25 × 平衡策略 + 腿部参考权重 × 视频偏移 + 残差动作`，手臂使用独立幅度权重。新增 `motion_gain` 动作包络会根据躯干直立度、根部速度和质心支撑偏差放大或收回动作，最终仍限制到关节范围和电机力矩范围。手腕虽然没有可靠参考旋转，仍允许策略为平衡输出残差。

188维观测包含投影重力、根部角速度、根部位置误差、根部线速度、29维实际关节角和速度、参考角和速度、关节角误差、上一动作以及参考相位正余弦。

训练从参考随机相位初始化，使用GAE、PPO裁剪目标、价值损失、熵项和梯度限幅。物理摔倒为终止；参考结束和可配置回合时限为截断（默认按参考时长加2秒、至少10秒），截断使用终态价值bootstrap。奖励包含关节姿态、速度、根部位置、手脚末端位置、足接触和躯干直立，以及动作幅度和变化惩罚。

### 每关节辅助网络、腾空凸包与手臂优先

- 每个关节各有一套小型残差MLP（共享观测上下文编码，29个独立输出头），输出叠加到原Actor均值；最后一层零初始化，因此加载旧权重或新训练起点与旧策略逐位一致。默认 `joint_assist_gain=0.18`，残差经tanh限幅。
- 腾空判定为足部球体与地面无接触。此时把左右足部球投影到水平面，按足半径、合理支撑半径 `0.12m` 和短时速度预测扩张后求凸包，计算质心相对凸包的有符号裕度：内部为正，外部为负。
- 腾空仿真正在降低模仿任务与动作惩罚：根部/末端/接触项缩放为 `0.55`，动作幅度与平滑惩罚缩放为 `0.45`；同时加入凸包内部奖励和越界软惩罚。地面接触阶段权重保持原样。
- 手臂稳定优先：一般动作增益 `gain` 之外再乘稳定优先系数 `floor+(1-floor)*margin^power`，默认 `floor=0.35`、`power=1.0`，失衡时先收手臂；奖励中手臂姿态与腕端跟踪占30%，并按同一稳定系数缩放。旧checkpoint显式走 `arm_priority=false`，臂幅与奖励口径保持原样。
- 新增命令行开关：`--joint-assist/--no-joint-assist`、`--joint-assist-gain`、`--airborne-reward/--no-airborne-reward`。对应checkpoint保存 `joint_assist`、`joint_assist_gain`、`joint_assist_residual`、`airborne_reward` 和完整 `motion` 字段。

这是针对同一个固定参考的相位条件模仿策略；尚未实现多动作数据集、动作选择、地形泛化、域随机化和参考未来窗口。

## 输出与查看

```text
motions/fullbody/reference.npz    全身参考，包含times/qpos/qvel/joint_names/contacts/ik_error/rejected/root_mode/root_scale
motions/fullbody/model/           重定向使用的独立模型
runs/fullbody/policy.pt           策略、优化器、迭代数及维度信息
runs/fullbody/metrics.jsonl       每轮训练记录
runs/fullbody/evaluation/evaluation.json  物理评估结果
runs/fullbody/preview/evaluation.json     运动学预览结果
```

`qpos` 是36维（根位置3+四元数4+29关节），`qvel` 是35维（根速度6+29关节）。`root_mode` 可能是 `in_place`、`contact_grounded` 或 refine 后的 `contact_grounded_support_refined`；`contacts` 为 `(T,2)` 左右脚接触标签。关节顺序读取模型执行器并写入文件，加载时严格校验。checkpoint需要与188维观测和29维动作一致；开发早期182维checkpoint不兼容。

`kinematic_only=true` 仅证明参考可以预览，不能证明策略稳定。评估报告中应同时检查 `policy_used`、`fallen` 和 `reference_completed`；`mean_reward`较高也可能在片段结束前摔倒，不能单独用于判断成功。

## 独立命令

```powershell
python -m fullbody retarget vedio/example.npz --output motions/fullbody/reference.npz
python -m fullbody retarget vedio/example.npz --output motions/fullbody/reference.npz --root-mode in_place --no-refine
python -m fullbody preview motions/fullbody/reference.npz --viewer
python -m fullbody train motions/fullbody/reference.npz --output runs/example --iterations 100 --steps 256
python -m fullbody evaluate motions/fullbody/reference.npz --checkpoint runs/example/policy.pt --viewer
# 不指定checkpoint：物理PD基线
python -m fullbody evaluate motions/fullbody/reference.npz
python -m unittest discover -s tests -v
```

DeepSeek已新增后台全身训练工具、状态查询、停止和常开渲染窗口，见 [CHAT_TRAINING.md](CHAT_TRAINING.md)。原robot_learn_video仍执行上半身回放，训练使用robot_train_video。

## 评估失稳诊断

用户自有视频的CUDA训练实验、可用缩幅权重和未完成的全幅动作问题见 [SELF_VIDEO_TRAINING.md](SELF_VIDEO_TRAINING.md)。

评估现在额外保存 `trajectory.json`，并返回实际仿真时长、参考时长、完成比例、终止原因、骨盆高度、直立程度、根部平移误差、速度及终止帧力矩饱和比例。`root_height` 表示骨盆低于0.35m；`pelvis_tilt` 表示骨盆朝上轴Z分量低于0.35。

用户600轮权重 `runs/my_policy/policy.pt` 的复现结果：127步/2.54秒，参考9.8566秒，完成25.77%；终止原因root_height，骨盆0.3406m，平面位置误差0.6023m，根部速度2.7736m/s，终止帧力矩饱和比例0。诊断在 `runs/my_policy/diagnostic/`。较高平均奖励和较小关节误差不代表支撑稳定，也不能据单个终止帧排除此前力矩饱和。原策略未修改。

## 当前验证与后续训练

官方示例生成40帧全身参考，0帧因全身关键点遮挡而拒绝，平均IK残差约0.0564（混合方向与正则项，无长度单位）。参考运动学预览可完成；训练梯度、权重保存/恢复、评估和原有视频功能有回归测试。

最终188维版本已通过6项回归。官方示例进行10轮、每轮256步的短训练后，权重位于 `runs/fullbody_v1/policy.pt`；物理评估第38步（约0.76秒）失稳，平均关节MSE约0.00941，未完成参考。该结果保存在 `runs/fullbody_v1/evaluation/evaluation.json`。物理评估摔倒时命令返回退出码2。

短训练仅用于验证实现，不是收敛权重。实际官方视频评估结果保存在输出目录的JSON中：当前策略仍会失稳。继续增加轮数可以尝试改善，但不保证单靠增加轮数收敛；优先检查参考的支撑脚、重心和足部滑移，改善接触约束与参考动作质量，再考虑并行采样、课程学习和预训练策略初始化。
