# G1-DAgent

独立的双 Agent 训练观察与调参工具。它**只读**调用现有 G1 项目的数据和状态，
不会修改 `E:\TEST_RECORD\G1_Project\G1\G1_LLM_Robot\G1_LLM_Robot`：

1. **状态分析 Agent**：读取 `runs/chat_training` 和 `runs/docker_jobs` 的
   `status.json`、`metrics.jsonl`、`evaluation.json`，总结事实、风险和方向。
2. **参数调试 Agent**：依据结果生成有边界的参数建议、重试命令和验证步骤。
   只生成建议，不自动修改或启动训练。

两个 Agent 共享一个编排器，输出 Markdown/JSON 报告到 `reports/`。

## 安装

```powershell
cd E:\TEST_RECORD\G1_Project\G1-DAgent
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

然后按 [API_SETUP.md](API_SETUP.md) 填入 DeepSeek API Key。没有 Key 时仍可运行，
只是不做 LLM 文本分析。

## 命令

```powershell
# 列出最近的训练任务
python main.py list

# 查看最新任务状态
python main.py status

# 指定任务
python main.py status --job 20261024_182324_aa0701

# 只生成参数调试建议
python main.py debug

# 两个 Agent 协作并写 Markdown/JSON 报告
python main.py report
python main.py report --job 20261024_182324_aa0701 --no-llm

# 定时生成最新任务报告，每 60 秒一次
python main.py watch --interval 60
```

`--no-llm` 放在子命令前也有效，例如：

```powershell
python main.py --no-llm report
```

## 参数调试边界

建议不会越出已有训练约束：

- `motion_gain`：`1.0~2.0`
- `leg_weight`：`0~2`
- `arm_weight`：`0.25~1.5`
- `rl_pd_influence`：`0~1`
- `joint_assist_gain`：`0~2`
- 手部/脚部收敛范围：各 `[0,1]`

建议逻辑优先处理摔倒、参考完成度低和漂移；参数变化以 `0.02~0.10` 小步为主。
每次只改变一个变量，并保留旧 checkpoint 做物理 A/B 对照。

## 数据来源

默认读取：

```text
E:\TEST_RECORD\G1_Project\G1\runs\chat_training
E:\TEST_RECORD\G1_Project\G1\runs\docker_jobs
E:\TEST_RECORD\G1_Project\G1\skills\registry.json
```

可用环境变量覆盖：

```powershell
$env:G1_DATA_ROOT = 'E:\TEST_RECORD\G1_Project\G1'
$env:G1_DAGENT_REPORTS = 'E:\TEST_RECORD\G1_Project\G1-DAgent\reports'
```

## 测试

```powershell
cd E:\TEST_RECORD\G1_Project\G1-DAgent
python -m unittest discover -s tests -v
```
