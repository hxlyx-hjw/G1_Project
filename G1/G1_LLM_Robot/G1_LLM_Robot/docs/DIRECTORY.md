# 项目目录导航

| 位置 | 内容 |
|---|---|
| `Start.py` | 统一启动入口，Visual Studio 启动文件 |
| `G1Robot.pyproj` | Visual Studio Python项目 |
| `README.md` | 总体使用说明 |
| `docs/` | 全身训练、视频功能及目录说明 |
| `fullbody/` | 全身参考、物理环境、PPO训练、评估及空间技能库 |
| `video_motion/` | 视频骨架提取与上半身回放 |
| `g1_walkfree/` | 原有行走控制器与界面 |
| `coordination/` | 关键节点-质心协调网络与独立跳跃 |
| `skills/` | 节点技能库 registry.json 与空间技能库 spatial_registry.json |
| 根目录其余Python模块 | DeepSeek对话、工具分发、机器人会话及通用辅助模块，保留位置以兼容导入 |
| `app_logging.py` / `runtime_io.py` / `environment_check.py` | 日志、Windows安全写入、只读环境诊断 |
| `Start_G1.bat` | Windows双击启动脚本 |
| `tests/` | 不联网的自动回归测试 |
| `examples/` | 手动演示、机器人接口检查、联网端到端验证 |
| `vedio/` | 视频示例与Start视频提取结果，沿用既有目录名 |
| `motions/` | 全身参考与其他动作数据 |
| `fullbody/datasets.py` | 本地数据集索引、片段选择和训练入口 |
| `D:\G1_Datasets\` | AIST++/Motion-X/LAFAN1/100STYLE 等外部数据集 |
| `runs/` | 训练权重、指标与评估诊断，现有结果保留原路径 |
| `outputs/` | 演示GIF等展示文件 |
| `requirements-*.txt` | 环境依赖 |
| `api_key.txt` | 本地密钥配置，维持原位并已从Git忽略 |
| `.vs/`、`bin/`、`__pycache__/` | IDE或Python生成目录，保留并从Git忽略 |
| `logs/` | 运行日志（app.log，不可写时回退LocalAppData） |

所有命令在项目根目录执行：

```powershell
python Start.py
python -m unittest discover -s tests -v
python -m examples.smoke_video_motion
python -m examples.demo_llm_actions
python -m examples.example_module_usage --no-view
```

`examples/e2e_test.py` 会联网调用DeepSeek，需密钥并产生API调用；未纳入自动离线测试。`examples/test_bridge.py` 是原有脚本式接口检查，未纳入unittest发现目录。

完整训练说明见 [FULLBODY_TRAINING.md](FULLBODY_TRAINING.md)，总览见 [README](../README.md)。本次仅归类辅助文件，未移动或删除用户视频、参考动作、训练权重和本地配置。

外部数据集的下载状态、许可和训练命令见
[DATASETS_G1.md](DATASETS_G1.md)。
