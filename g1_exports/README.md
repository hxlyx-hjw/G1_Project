# g1_exports

G1 策略回放与参考动作视频导出工具。

| 文件 | 作用 |
| --- | --- |
| `render_reference_video.py` | 把 retarget 后的 `reference.npz` 渲染成 MP4 |
| `render_policy_video.py` | 回放训练好的策略并渲染 MP4 |
| `render_common.py` | 视频写入器（优先 `avc1`，自动回退 `mp4v`） |

两个脚本都依赖本机的训练包：

```text
E:\TEST_RECORD\Server_Deploy\CPU_Server\G1-CPU-Training-Package
```

迁移后可用 `--package` 或环境变量 `G1_CPU_PACKAGE` 指定新位置。

## 用法

```powershell
cd E:\TEST_RECORD\G1_Project\g1_exports

# 查看参数（不会加载 MuJoCo/torch）
python render_reference_video.py --help
python render_policy_video.py --help

# 参考动作
python render_reference_video.py --run <导出目录>

# 策略回放
python render_policy_video.py --run <导出目录> --checkpoint <权重.pt>

# 只渲染前 5 秒到临时文件
python render_reference_video.py --run <导出目录> --max-seconds 5 --output preview.mp4
```

默认输出位于 `--run` 目录内：MP4、封面 PNG 和 `.report.json`。
`--max-seconds` 可用于快速验证。

## 测试

```powershell
cd E:\TEST_RECORD\G1_Project\g1_exports
python -m pytest tests -q
```

测试只验证命令行契约和错误提示，不需要训练包或 GPU。
