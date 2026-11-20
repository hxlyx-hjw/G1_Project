# AMASS G1 数据下载与接入

数据源使用 Hugging Face 镜像
`ember-lab-berkeley/AMASS_Retargeted_for_G1`，下载到 `D:\AMASS`，
转换后的训练参考保存到 `E:\TEST_RECORD\G1_Project\G1\motions\amass`。

该镜像标注为 CC-BY-4.0，内容是 AMASS 经 G1 重定向后的动作，不包含
原始 SMPL+H 参数。使用前仍需遵守 AMASS 原始的非商业科研许可。

## 下载

```powershell
cd E:\TEST_RECORD\G1_Project\G1\G1_LLM_Robot\G1_LLM_Robot
python -m fullbody amass-download
python -m fullbody amass-list
```

下载器使用 `hf-mirror.com` 直链、断点续传和并行下载，不会写入
账号或访问令牌。可通过环境变量覆盖路径：

```powershell
$env:G1_AMASS_ROOT = 'D:\AMASS\datasets\AMASS_Retargeted_for_G1'
$env:G1_AMASS_MOTIONS = 'E:\TEST_RECORD\G1_Project\G1\motions\amass'
```

## 转换并接入训练

先转换 5 个动作验证：

```powershell
python -m fullbody amass-export --limit 5
```

确认无误后转换全部动作：

```powershell
python -m fullbody amass-export
```

转换器处理了以下差异：

- 镜像文件实际使用 IsaacLab 的 `wxyz` 四元数；
- 消除四元数符号翻转；
- 数值归一化；
- 按每个动作的中位脚底高度做整体落地校准，保留腾空时的相对高度；
- 使用 MuJoCo 计算 `qvel`；
- 校验并裁剪到 29 个 G1 关节的实际范围；
- 写入 `times/qpos/qvel/joint_names/root_mode`，可直接被现有 `Reference` 加载。

转换后的目录和原数据集保持相同的层级：

```text
E:\TEST_RECORD\G1_Project\G1\motions\amass\ACCAD\Female1General_c3d\A1-Stand_poses_120_jpos.npz
E:\TEST_RECORD\G1_Project\G1\motions\amass\manifest.json
```

## 训练示例

```powershell
python -m fullbody train `
  "E:\TEST_RECORD\G1_Project\G1\motions\amass\ACCAD\Female1General_c3d\A1-Stand_poses_120_jpos.npz" `
  --output E:\TEST_RECORD\G1_Project\G1\runs\amass_stand --iterations 100
```

现有训练器仍是单参考训练；转换后的每个动作都可以独立作为
`reference.npz` 使用。多动作混合采样尚未接入。

## DeepSeek 自主训练

DeepSeek 可通过三个工具调用全部 447 个动作：

- `robot_amass_datasets`：列出数据集及动作数量。
- `robot_amass_list`：按 `dataset`、`query` 分页检索，返回稳定 `motion_id`。
- `robot_amass_train`：按 `motion_id` 启动训练；也支持 `dataset+name` 或 `path`。

推荐调用顺序：

```text
robot_amass_datasets
→ robot_amass_list(dataset="ACCAD", query="walk", limit=20)
→ 询问用户 rl_pd_influence 和 hand/foot_rl_pd_range
→ robot_amass_train(motion_id="...", hand_rl_pd_range=[0.2,0.6],
                    foot_rl_pd_range=[0.5,0.9], rl_pd_influence=0.7,
                    iterations=500, device="auto")
→ robot_training_status(job_id="...")
```

`robot_amass_train` 默认 `backend="local"`，已在本机完成 ACCAD 与
BMLhandball 两个数据集的实际启动验证。只有 `g1-llm-robot:cu128`
镜像已构建且 Docker 可用时，才设置 `backend="docker"`，之后使用
`robot_docker_status` 查询。训练、主动报告、停止、技能登记和视频导出
沿用原有工具。
