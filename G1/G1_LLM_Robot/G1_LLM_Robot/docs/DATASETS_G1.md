# G1 本地数据集下载、接入与训练说明

本文档对应当前机器上的数据集根目录：

```text
D:\G1_Datasets
```

所有原始数据与项目代码分离。项目只读取数据，不修改原始文件；训练结果仍写入
`E:\TEST_RECORD\G1_Project\G1\runs`。

## 1. 当前状态

| 数据集 | 本地内容 | 训练方式 | 状态 |
|---|---|---|---|
| AMASS | 447 个已重定向 G1 `reference.npz` | 参考动作 PPO | 可直接训练 |
| AIST++ | 10 个舞种 × 2 机位，20 个 MP4，含相机标定 | 双视角视频 | 可直接训练 |
| Motion-X | 301 个已解包 MP4，1/119 个 parquet 分片 | 单视频 | 可直接训练 |
| LAFAN1 | 77 个 BVH，约 4.6 小时 | 需 BVH→G1 转换 | 已下载 |
| 100STYLE | 100 种风格、1620 个 BVH，约 3.37GB | 需 BVH→G1 转换 | 已下载/解包 |
| BEDLAM | 空目录 + 许可说明 | 单视频/合成多视角 | 需注册许可 |
| Human3.6M | 空目录 + 许可说明 | 标定双视角 | 需注册许可 |
| MPI-INF-3DHP | 空目录 + 许可说明 | 标定多视角 | 需注册许可 |
| TotalCapture | 空目录 + 许可说明 | 标定多视角 | 需申请许可 |
| CMU Panoptic | 空目录 + 许可说明 | 多视角 | 需研究用途确认 |
| MoVi | 空目录 + 许可说明 | 4 视角视频 | 需申请许可 |
| CMU MoCap | 空目录 + 许可说明 | 原始动捕 | 需 ASF/AMC/BVH 转换 |

> 当前没有代替用户接受任何需要人工确认的许可协议。受限数据集需要你本人完成
> 注册并接受条款后，才能把文件放入对应目录。

## 2. 目录结构

```text
D:\G1_Datasets\
  00_INDEX\
    catalog.json                 # 自动生成的统一索引
  AMASS\
    README_LINK.md               # 指向 D:\AMASS 和已转换动作
    manifests\reference_list.txt
  AISTPP\
    annotations\
      motions\                   # 411 个动作标注
      cameras\                   # 9 机位标定与 mapping.txt
    videos\                      # 10 组 c01/c03 视频
    calibrations\                # 已转换成项目可直接读取的 K/R/t
    manifests\
      dual_view.txt
      single_view.txt
      pairs.json
    download_subset.py
  MotionX\
    parquet\                     # jxie/motion_x_video 分片
    videos\                      # 已解包的 301 个 MP4
    manifests\video_list.txt
  LAFAN1\
    ubisoft-laforge-animation-dataset\
    bvh\                         # 77 个 BVH
    manifests\bvh_list.txt
  100STYLE\
    100STYLE.zip
    data\                        # 解包后的 BVH
    manifests\bvh_list.txt
  BEDLAM\
  Human3.6M\
  MPI_INF_3DHP\
  TotalCapture\
  CMUPanoptic\
  MoVi\
  CMU_MoCap\
```

## 3. 统一数据入口

项目新增模块：

```text
fullbody/datasets.py
```

查看所有数据集：

```powershell
cd E:\TEST_RECORD\G1_Project\G1\G1_LLM_Robot\G1_LLM_Robot
python -m fullbody.datasets inventory
```

列出片段：

```powershell
python -m fullbody.datasets list --dataset AISTPP --limit 10
python -m fullbody.datasets list --dataset MotionX --limit 10
python -m fullbody.datasets list --dataset AMASS --limit 10
```

按关键词过滤：

```powershell
python -m fullbody.datasets list --dataset AISTPP --query gBR
python -m fullbody.datasets list --dataset MotionX --query Drum
```

刷新清单和统一索引：

```powershell
python -m fullbody.datasets build
```

每次增加、删除或解包数据后执行一次 `build`，会更新：

```text
D:\G1_Datasets\00_INDEX\catalog.json
D:\G1_Datasets\MotionX\manifests\video_list.txt
D:\G1_Datasets\LAFAN1\manifests\bvh_list.txt
D:\G1_Datasets\100STYLE\manifests\bvh_list.txt
D:\G1_Datasets\AMASS\manifests\reference_list.txt
```

## 4. 接入训练

### 4.1 AIST++ 双视角

AIST++ 使用 c01/c03 两个标定机位，夹角约 83°。相机参数已经转换为项目
`video_motion.multiview` 需要的 `K/R/t/dist` 格式。

真实双视角烟测结果：

```text
有效关节比例：95.4%
中位重投影误差：4.66 px
相机模式：calibrated
```

只检查选择结果：

```powershell
python -m fullbody.datasets train `
  --dataset AISTPP --clip 0 `
  --iterations 1 --envs 1 --device cpu `
  --max-seconds 5 --dry-run
```

正式训练：

```powershell
python -m fullbody.datasets train `
  --dataset AISTPP --clip 0 `
  --iterations 500 --envs 4 --device auto `
  --backend dlt --max-seconds 12 --repair-gap-seconds 0.7 `
  --rl-pd-influence 0.7 `
  --hand-rl-pd-range 0.2 0.6 `
  --foot-rl-pd-range 0.5 0.9
```

`--clip` 可以使用：

- `0`、`1` 等索引；
- 标注 key，例如 `gBR_sBM_cAll_d04_mBR0_ch01`；
- `pair_key`，例如 `gBR_sBM_c01_d04_mBR0_ch01`。

可用片段：

```text
gBR  breaking
gHO  house
gJB  jazz ballet
gJS  jazz street
gKR  krump
gLH  locking hiphop
gLO  locking
gMH  middle hiphop
gPO  popping
gWA  waacking
```

`gWA_sBM_cAll_d25_mWA0_ch01` 本地质量检查：

```text
双视角有效关节：94.9%
完整时间轴：360 帧 / 11.97 秒
中位重投影误差：6.09 px
左膝短空洞由 0.7 秒时序修复覆盖
```

### 4.2 Motion-X 单视频

```powershell
python -m fullbody.datasets train `
  --dataset MotionX `
  --clip Ancient_Drum_11_clip6 `
  --iterations 500 --envs 4 --device auto `
  --fps 30 --max-seconds 60 `
  --rl-pd-influence 0.7 `
  --hand-rl-pd-range 0.2 0.6 `
  --foot-rl-pd-range 0.5 0.9
```

### 4.3 AMASS 参考动作

```powershell
python -m fullbody.datasets list --dataset AMASS --limit 20

python -m fullbody.datasets train `
  --dataset AMASS `
  --clip A1-Stand_poses_120_jpos `
  --iterations 500 --envs 4 --device auto
```

AMASS 不经过视频提取，直接使用 G1 `reference.npz`，是当前最稳定的动捕训练路径。

### 4.4 训练任务查询

`train` 会返回：

```json
{
  "job_id": "20261027_200050_b02fbc",
  "state": "queued",
  "output_dir": "E:\\TEST_RECORD\\G1\\runs\\chat_training\\20261027_200050_b02fbc"
}
```

查看状态：

```powershell
python -m fullbody.datasets status <job_id>
```

停止训练并保留最新权重：

```powershell
python -m fullbody.datasets stop <job_id>
```

从已有任务的 `training/latest.pt` 继续训练到新的目标轮次：

```powershell
python -m fullbody.datasets resume <job_id> --iterations 30000 --device auto
```

续训会保留同一个任务目录、metrics、best、policy 和历史记录，只追加新的轮次。

在 G1 聊天界面中使用：

```text
robot_training_status(job_id="<job_id>")
robot_training_stop(job_id="<job_id>")
```

### 4.5 已验证的训练烟测

两条链路都已完成：

```text
AIST++  双视角 DLT + 重定向 + 1 轮 PPO + 独立物理评估
MotionX 单视频 MediaPipe/SMPL-X + 重定向 + 1 轮 PPO + 独立物理评估
```

测试任务：

```text
E:\TEST_RECORD\G1_Project\G1\runs\chat_training\20261027_200050_b02fbc
E:\TEST_RECORD\G1_Project\G1\runs\chat_training\20261027_200341_8a1d37
E:\TEST_RECORD\G1_Project\G1\runs\chat_training\20261027_204528_5f55a0
```

Motion-X 的单视频烟测使用 `Baduanjin_1_clip2`，在 CPU 上 5 秒片段约
2 分钟（主要耗时是 SMPL-X 拟合）。1 轮训练摔倒属于未收敛，不代表数据链路失败。

`gWA` 正式训练任务 `20261027_204528_5f55a0` 从 4,445 轮续训到
30,000 轮，使用 8 环境，保留原 metrics/best/policy 历史。

## 5. 数据来源与下载命令

### AIST++

- 主页：https://google.github.io/aistplusplus_dataset/
- 下载页：https://google.github.io/aistplusplus_dataset/download.html
- 视频下载脚本：https://raw.githubusercontent.com/google/aistplusplus_api/main/downloader.py

当前只下载了 10 组、每组 c01/c03 两机位，避免一次拉取完整 515GB AIST Dance DB。

扩展更多片段：

1. 修改 `D:\G1_Datasets\AISTPP\download_subset.py` 的 `choose_sequences()`；
2. 重新运行脚本，已存在视频会跳过；
3. 运行 `python -m fullbody.datasets build`。

### Motion-X

- 项目：https://github.com/IDEA-Research/Motion-X
- 视频分片：https://huggingface.co/datasets/jxie/motion_x_video
- Motion-X++：https://huggingface.co/datasets/YuhongZhang/Motion-Xplusplus

当前下载的是第 1 个分片，共 301 个视频。完整视频约 15GB、119 个分片。

单个分片下载：

```powershell
curl.exe --ssl-no-revoke -L --continue-at - `
  -o "D:\G1_Datasets\MotionX\parquet\train-00001-of-00119.parquet" `
  "https://huggingface.co/datasets/jxie/motion_x_video/resolve/main/data/train-00001-of-00119.parquet?download=true"
```

解包：

```powershell
python -m fullbody.datasets extract-motionx `
  --parquet "D:\G1_Datasets\MotionX\parquet\train-00001-of-00119.parquet"
```

也可以使用随包脚本继续下载剩余分片：

```powershell
python "D:\G1_Datasets\MotionX\download_shards.py" `
  --start 1 --end 118 --extract
```

完整 119 个分片约 15GB，下载耗时取决于网络。

### LAFAN1

- 仓库：https://github.com/ubisoft/ubisoft-laforge-animation-dataset
- 许可：CC BY-NC-ND 4.0
- 已下载：77 BVH、496,672 帧、约 4.6 小时

```powershell
git lfs install
git clone https://github.com/ubisoft/ubisoft-laforge-animation-dataset.git `
  D:\G1_Datasets\LAFAN1\ubisoft-laforge-animation-dataset
```

### 100STYLE

- 主页：https://www.ianxmason.com/100style/
- Zenodo：https://zenodo.org/record/8127870
- 许可：CC BY 4.0
- 内容：约 400 万帧、100 种 locomotion 风格
- 已解包：1620 个 BVH，约 3.37GB

下载：

```powershell
curl.exe --ssl-no-revoke -L --continue-at - `
  -o "D:\G1_Datasets\100STYLE\100STYLE.zip" `
  "https://zenodo.org/api/records/8127870/files/100STYLE.zip/content"
```

解包：

```powershell
Expand-Archive -LiteralPath "D:\G1_Datasets\100STYLE\100STYLE.zip" `
  -DestinationPath "D:\G1_Datasets\100STYLE\data" -Force
python -m fullbody.datasets build
```

### AMASS

- 官方：https://amass.is.tue.mpg.de/
- 当前机器：`D:\AMASS`
- 已转换 G1 动作：`E:\TEST_RECORD\G1_Project\G1\motions\amass`
- 当前 447 个参考动作

当前镜像只包含部分 AMASS。完整 AMASS 需要登录官方网站并接受许可。

## 6. 受限数据集

以下数据集没有代替用户下载：

| 数据集 | 官方入口 | 要求 |
|---|---|---|
| BEDLAM | https://bedlam.is.tue.mpg.de/ | 注册并接受许可；Hugging Face 为 gated |
| Human3.6M | http://vision.imar.ro/human3.6m/ | 官方注册 |
| MPI-INF-3DHP | https://vcai.mpi-inf.mpg.de/3dhp-dataset/ | 申请/许可 |
| TotalCapture | https://cvssp.org/data/totalcapture/ | 注册，研究用途 |
| CMU Panoptic | http://domedb.perception.cs.cmu.edu/ | 研究用途确认 |
| MoVi | http://www.biomotionlab.ca/movi/ | 非商业研究许可 |
| CMU MoCap | http://mocap.cs.cmu.edu/ | 原始 ASF/AMC 或 BVH |

下载后可放入对应目录，并按下节格式接入。

### BEDLAM

接受许可并登录 Hugging Face 后：

```powershell
huggingface-cli login
huggingface-cli download Intelligent-Systems/BEDLAM `
  --repo-type dataset `
  --local-dir D:\G1_Datasets\BEDLAM
```

### Human3.6M

保留官方目录结构，例如：

```text
D:\G1_Datasets\Human3.6M\
  S1\
  S5\
  ...
  Release-v1.2\
```

需要把官方相机标定转换成 `camera_a/camera_b` 的 K/R/t JSON。

### MPI-INF-3DHP / TotalCapture / CMU Panoptic / MoVi

这些数据带多机位视频和标定。接入时需要：

1. 选两路全身可见、夹角 60°~120° 的相机；
2. 读取官方内参/外参，转换为 K/R/t/dist；
3. 写 `list.txt`：`video_a.mp4,video_b.mp4`；
4. 用 `robot_multiview_train` 或 `video_motion.multiview` 训练。

## 7. BVH 转换说明

LAFAN1、100STYLE 和 CMU MoCap 是骨架动画，不是 MP4，也不是 G1
`reference.npz`。当前数据集入口会列出它们，但不会直接送入 PPO。

转换目标：

```text
BVH 全局关节位置
  → 映射到 MediaPipe 33 点
  → pose.npz (times/points/visibility)
  → fullbody.reference.convert
  → fullbody.refine.refine
  → G1 reference.npz
```

转换完成后放入：

```text
D:\G1_Datasets\<DATASET>\references\
```

并在 `fullbody/datasets.py` 中把对应数据集标记为可直接训练。

## 8. 双视角质量规则

默认要求：

```text
min_valid_joint_fraction = 0.05
max_reprojection = 40 px
self_correct = true
repair_gap_seconds = 0.7（fullbody.datasets 的 AIST++ 入口）
```

低于有效关节阈值时会拒绝双视角并降级到 a 视角单目，避免训练错误三维点。
使用标定相机时，实测 AIST++ 有效关节可达 95% 以上。

直接使用 `python -m video_motion multiview` 时默认修复间隙仍为 0.25 秒；
舞蹈动作建议显式传 `--repair-gap-seconds 0.7`。

没有标定的手机双视角通常无法直接得到可靠 DLT。应优先：

- 使用 AIST++、Human3.6M、MPI-INF-3DHP、TotalCapture 的官方标定；
- 或先做相机标定；
- 不要仅凭“正面/侧面”猜测 baseline 和 FOV。

## 9. 磁盘与扩展

当前 D 盘占用由各数据集目录独立统计。新增完整 Motion-X 约 15GB；
完整 AIST Dance DB 约 515GB。下载前检查：

```powershell
Get-PSDrive D | Select-Object Used,Free
```

本次检查时的实际占用：

```text
D:\G1_Datasets 合计约 6.2GB
其中 100STYLE 约 4.62GB（含 zip 和解包 BVH）
LAFAN1 约 814MB
AISTPP 约 439MB
MotionX 约 341MB
D 盘剩余约 140GB
```

建议优先扩展：

1. AIST++ 更多舞蹈片段（双视角质量最高）；
2. Motion-X 更多 parquet 分片（单视频规模最大）；
3. LAFAN1/100STYLE 的 BVH 转换（腿部和平衡最有价值）；
4. 受限的 Human3.6M/MPI-INF-3DHP/TotalCapture（双视角验收）。

## 10. 许可提醒

- AMASS、AIST++、Motion-X、BEDLAM、Human3.6M、MPI-INF-3DHP、TotalCapture、
  CMU Panoptic、MoVi 均主要为研究用途。
- LAFAN1 为 CC BY-NC-ND 4.0。
- 100STYLE 为 CC BY 4.0。
- 不要把受限数据重新分发或用于商业训练，除非获得明确授权。
- 数据集目录中的 `README_LINK.md` / `README_DOWNLOAD.md` 保留官方入口。

## 11. 快速自检

```powershell
cd E:\TEST_RECORD\G1_Project\G1\G1_LLM_Robot\G1_LLM_Robot

python -m fullbody.datasets inventory
python -m fullbody.datasets build

python -m fullbody.datasets train `
  --dataset AISTPP --clip 0 `
  --iterations 1 --envs 1 --device cpu `
  --max-seconds 5 --dry-run
```

如果 `inventory` 中 AISTPP 显示 `ready`、`dual_view_pairs >= 1`，说明双视角数据
和训练入口已接通。
