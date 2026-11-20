# API 配置

G1-DAgent 复用 DeepSeek 的 OpenAI 兼容接口。没有 API Key 时：

- `list`、`status`、`debug` 和 `report --no-llm` 可正常使用；
- 只有两个 Agent 的自然语言分析会关闭。

## 推荐方式：当前项目 `.env`

1. 复制模板：

```powershell
Copy-Item E:\TEST_RECORD\G1_Project\G1-DAgent\.env.example E:\TEST_RECORD\G1_Project\G1-DAgent\.env
```

2. 编辑 `E:\TEST_RECORD\G1_Project\G1-DAgent\.env`，填入自己的 Key：

```dotenv
DEEPSEEK_API_KEY=sk-请替换为你的Key
G1_MODEL_BASE_URL=https://api.deepseek.com
G1_MODEL_NAME=deepseek-flash
```

不要把 `.env` 提交到 Git、粘贴到聊天记录或写进训练报告。

## 临时环境变量

```powershell
$env:DEEPSEEK_API_KEY = 'sk-...'
$env:G1_MODEL_BASE_URL = 'https://api.deepseek.com'
$env:G1_MODEL_NAME = 'deepseek-flash'
python main.py report
```

## 本地兼容服务

如果使用 Ollama 或其他 OpenAI 兼容服务：

```powershell
$env:DEEPSEEK_API_KEY = 'local'
$env:G1_MODEL_BASE_URL = 'http://localhost:11434/v1'
$env:G1_MODEL_NAME = '你的模型名'
python main.py report
```

## 验证

```powershell
cd E:\TEST_RECORD\G1_Project\G1-DAgent
python main.py report --no-llm
python main.py report
```

第二次命令应能生成 `reports/<job_id>_<timestamp>.md`，其中包含
“状态 Agent LLM 分析”和“参数 Agent LLM 审查”。如果 API 不可用，
报告仍会保存本地确定性分析，并记录 `llm_error`。
