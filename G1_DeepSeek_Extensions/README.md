# G1 DeepSeek 研究工具扩展

本目录把独立的 `SuperMimic` 和 `RAX` 接入 G1 现有 DeepSeek Function Calling
循环。采用外部子类和运行时注册，不修改以下现有源码：

```text
G1/.../deepseek_robot.py
G1/.../robot_tools.py
G1/.../Start.py
SuperMimic/**
RAX/**
```

## 启动

双击 `start_chat.bat`，或执行：

```powershell
cd E:\TEST_RECORD\G1_Project\G1_DeepSeek_Extensions
python start_chat.py
```

不连接 DeepSeek，只检查两个环境：

```powershell
python start_chat.py --integration-status
```

只列出扩展工具声明：

```powershell
python start_chat.py --list-integration-tools
```

原 `Start_G1.bat` 和 `Start.py` 保持原样；需要研究工具时使用本目录的启动器。

## 工具

SuperMimic：

| 工具 | 作用 |
| --- | --- |
| `robot_supermimic_status` | 检查 WSL 环境、98k 策略和最近指标 |
| `robot_supermimic_learn_video` | 单/双视角提取→G1重定向→自动训练，并保持实时渲染窗口 |
| `robot_supermimic_evaluate` | 异步 CPU MuJoCo 评测并可选保存轨迹 |
| `robot_supermimic_train` | 异步训练或小型烟测 |
| `robot_supermimic_render` | 异步渲染参考虚影 MP4 |
| `robot_supermimic_job_status` | 查询任务状态、管线阶段/进度和训练指标 |
| `robot_supermimic_logs` | 实时读取训练百分比、指标历史、耗时、ETA和各日志尾部 |

`robot_supermimic_logs` 同时支持流水线任务（`output_root/training/`）和独立训练任务
（`output_root/`）两种目录布局。首次调用返回日志尾部与 `next_cursor`；之后把
`next_cursor` 作为 `cursor` 再调用，只返回新增训练日志，可用于持续轮询实时进度。

`robot_supermimic_learn_video` 默认 `mode=smoke`（少量迭代），只验证
“提取→重定向→30 Hz CSV→训练”链路，不会得到可用策略；真实训练需显式传
`mode=paper` 并提高 `iterations`，或对本次输出目录的 `motion/` 调用
`robot_supermimic_train`。SMPL-X 后端可另行传 `smplx_model_dir`、
`smplx_gender`、`smplx_device`。

RAX：

| 工具 | 作用 |
| --- | --- |
| `robot_rax_status` | 检查 G1 只读后端和最近缩放结果 |
| `robot_rax_run` | 异步执行探索、冻结、排练、提交和正式评测 |
| `robot_rax_report` | 读取正式报告 |
| `robot_rax_list` | 列出已有运行 |
| `robot_rax_job_status` | 查询任务状态和日志尾部 |

示例指令：

```text
检查 SuperMimic 和 RAX 的环境。
用 SuperMimic 学习 vedio_self/test7.mp4 的动作，提取后自动训练并保留渲染窗口。
用 SuperMimic 学习双视角文件夹 D:\Videos\wave，自动选择第一组 a/b 并训练。
查看 SuperMimic 任务的实时训练日志和训练进度。
用 SuperMimic 评估 data/eval_subset，并保存轨迹。
用 SuperMimic 对 walk1_subject1_10s 渲染视频。
运行一次预算 128 的 RAX，完成后给我正式报告。
```

## 隔离和路径

- SuperMimic 通过 `Ubuntu-22.04` WSL 和
  `/mnt/d/G1_Envs/SuperMimic` 独立环境启动。
- RAX 使用当前启动器的 Python，并通过 `--g1-root` 只读调用现有 G1。
- SuperMimic 输入和输出只能位于 `E:\TEST_RECORD\SuperMimic` 内。
- RAX 运行只能写入 `E:\TEST_RECORD\RAX\rax_min\runs`。
- 后台任务记录和日志保存在本目录 `runtime/jobs`。
- 长任务会返回 `job_id`；必须查询完成后再读取结果，不能把启动当作完成。

## 路径覆盖

移动项目时可在启动前设置：

```powershell
$env:G1_DEEPSEEK_G1_ROOT = 'E:\TEST_RECORD\G1_Project\G1\G1_LLM_Robot\G1_LLM_Robot'
$env:G1_DEEPSEEK_SUPERMIMIC_ROOT = 'E:\TEST_RECORD\SuperMimic'
$env:G1_DEEPSEEK_RAX_ROOT = 'E:\TEST_RECORD\RAX'
$env:G1_DEEPSEEK_WSL_DISTRO = 'Ubuntu-22.04'
$env:G1_DEEPSEEK_SUPERMIMIC_VENV = '/mnt/d/G1_Envs/SuperMimic'
```

## 测试

```powershell
cd E:\TEST_RECORD\G1_Project\G1_DeepSeek_Extensions
python -m unittest discover -s tests -v
```
