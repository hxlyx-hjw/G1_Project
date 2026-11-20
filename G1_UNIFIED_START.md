# G1 统一启动入口

## 图形控制台（推荐）

双击 G1 数据根目录中的 `启动G1界面.bat`，或执行：

```powershell
cd E:\TEST_RECORD\G1_Project\G1
python -m G1_Desktop.app
```

一个窗口集中对话、机器人状态/控制、训练任务、SuperMimic/RAX 环境与 56 个工具。
默认同时打开 MuJoCo 3D 窗口；加 `--no-viewer` 可关闭。详细说明见
`G1\G1_Desktop\README.md`。

## 命令行入口

双击 G1 数据根目录中的 `启动G1.bat`。

默认行为：

- 直接进入 DeepSeek Function Calling 对话；
- 自动打开 MuJoCo G1 仿真窗口；
- 注册原有机器人、视频、AMRAX、Docker 和训练工具；
- 注册 SuperMimic 与 RAX 扩展工具。

等效命令：

```powershell
cd E:\TEST_RECORD\G1_Project\G1
python Start_G1_Unified.py
```

无界面模式：

```powershell
python Start_G1_Unified.py --headless
```

只检查环境：

```powershell
python Start_G1_Unified.py --integration-status
```

查看全部 56 个工具：

```powershell
python Start_G1_Unified.py --list-tools
```

启动时会自动定位：

```text
G1 源码：      E:\TEST_RECORD\G1_Project\G1\G1_LLM_Robot\G1_LLM_Robot
扩展工具：     E:\TEST_RECORD\G1_Project\G1_DeepSeek_Extensions
SuperMimic：   E:\TEST_RECORD\SuperMimic
RAX：          E:\TEST_RECORD\RAX
```
