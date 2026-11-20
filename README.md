# G1 Project

本目录集中存放名称含 `G1` 的项目、程序包、导出结果和说明文档。

## 目录

```text
G1_Project/
  G1/                              # G1 数据根、应用、模型、训练和视频
    G1_Desktop/                    # PyQt5 一体化控制台（对话/机器人/训练/环境）
  g1_exports/                      # 策略/参考视频导出工具与历史导出
  G1-DAgent/                       # 双 Agent 调试/总结分析程序
  G1_人形机器人项目说明.md
  G1_项目文件夹树状图.md
```

图形控制台：双击 `G1\启动G1界面.bat`，或 `cd G1 && python -m G1_Desktop.app`。

服务器部署资产已统一移动到：

```text
E:\TEST_RECORD\Server_Deploy
```

当前推荐上传 `Server_Deploy\AutoDL\releases\AutoDL_Dance_PPO_COMPLETE_20261102.tar.gz`。
服务器训练包、Docker 配置和 SHA256 文件只保留在 `Server_Deploy`；`G1_Project`
仅保留本地 G1 应用、数据、训练输出和调试工具。详细索引见
`Server_Deploy\README.md`。

## G1 数据根

```text
E:\TEST_RECORD\G1_Project\G1
```

核心应用：

```text
E:\TEST_RECORD\G1_Project\G1\G1_LLM_Robot\G1_LLM_Robot
```

数据目录 `runs/`、`outputs/`、`motions/`、`skills/`、`logs/`、`vedio/` 和
`vedio_self/` 仍由 Junction 挂载，目标已统一更新到本目录。

```text
G1_LLM_Robot/G1_LLM_Robot/runs -> G1/runs
G1_LLM_Robot/G1_LLM_Robot/motions -> G1/motions
G1_LLM_Robot/G1_LLM_Robot/outputs -> G1/outputs
G1_LLM_Robot/G1_LLM_Robot/skills -> G1/skills
G1_LLM_Robot/G1_LLM_Robot/logs -> G1/logs
G1_LLM_Robot/G1_LLM_Robot/vedio -> G1/vedio
```

## 外部兼容链接

以下链接已更新到新位置：

```text
D:\SMPL                                         -> G1/models/SMPL
D:\WHAM                                         -> G1/models/WHAM
D:\VcXsrv                                       -> G1/tools/VcXsrv
C:\Users\lenovo\mujoco_menagerie\unitree_g1     -> G1/assets/mujoco_menagerie/unitree_g1
C:\Users\lenovo\Desktop\EmbodiedAI\G1_LLM_Robot -> G1/G1_LLM_Robot
```

最后一项用于兼容仍使用旧桌面路径的脚本。

## 路径迁移

已更新的关键入口：

- G1 `data_paths.py`
- G1 `environment_check.py`
- G1 `g1_walkfree/config.py`
- G1 SMPL-X 模型发现
- G1 CPU 包对应副本
- G1-DAgent `config.py`
- `g1_exports` 两个渲染脚本
- 视频列表 `vedio_self/list.txt`
- RAX 的 G1 只读后端与哈希清单

历史技能库 JSON 保留原始记录；`data_paths.rebase_paths()` 在读取时自动把旧
`E:\TEST_RECORD\G1` 前缀重定向到新根，因此不破坏旧技能。

## 验证

```text
G1 主项目测试          83 passed
G1 CPU 包测试          77 passed, 5 skipped
G1-DAgent 测试          6 passed
G1 桌面控制台测试       2 passed
g1_exports CLI 测试    4 passed
RAX harness 测试        6 passed
G1 环境诊断            全部通过
RAX held-out            7/8 at 128 episodes
```

历史 `runs/`、`outputs/` 和报告中的旧路径字符串属于运行记录，不影响当前程序；
当前代码、配置、技能读取和活动文档已使用新路径。

## GitHub 源码分发

公开源码仓库：

```text
https://github.com/hxlyx-hjw/G1_Project
```

完整归档 Release：

```text
https://github.com/hxlyx-hjw/G1_Project/releases/tag/v1.0.0-20261121
```

直接下载：

```text
https://github.com/hxlyx-hjw/G1_Project/releases/download/v1.0.0-20261121/G1_Project_Data_20261121.tar.gz
```

GitHub 仓库保留以下全部可维护内容：

- G1 DeepSeek Function Calling 控制 Agent
- G1 Desktop PyQt 控制台
- G1-DAgent 调试 Agent
- G1 DeepSeek Extensions
- PPO、重定向、MuJoCo 与视频提取源码
- 单元测试、配置模板、文档和小型机器人资产

以下内容不进入普通 Git 历史：

- `api_key.txt`、`tavily_api_key.txt` 和实际 `.env`
- WHAM、SMPL-X 等许可模型权重
- Python 虚拟环境
- 训练输出、历史日志、视频和缓存

完整项目（除密钥、虚拟环境和缓存）另以公开 Release 归档提供。

模型权重、SMPL-X 许可文件和本地密钥的恢复方式见
[MODEL_AND_DATA.md](MODEL_AND_DATA.md)。
