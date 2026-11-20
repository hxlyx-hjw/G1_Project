# Tavily API 配置与代码检查结果

## 推荐：配置文件（适合Visual Studio）

已创建项目根目录空文件 `tavily_api_key.txt`，与 `Start.py` 同级：

```text
E:\TEST_RECORD\G1_Project\G1\tavily_api_key.txt
```

将你在 <https://app.tavily.com/> 获取的Tavily Key写入该文件，只写密钥本身，不加引号或变量名。不要填到DeepSeek的 `api_key.txt`，不要把DeepSeek Key作为Tavily Key。

保存后重新运行 `Start.py`，选择AI聊天。程序自动注册 `search_web`，动作视频搜索和后台学习共用同一凭证。无需修改Python源代码，也无需额外设置 `--search`。配置文件已加入 `.gitignore`。

## 可选：环境变量

当前PowerShell窗口：

```powershell
$env:TAVILY_API_KEY = '你的Tavily密钥'
python Start.py --mode chat
```

持久保存到当前Windows用户：

```powershell
[Environment]::SetEnvironmentVariable('TAVILY_API_KEY', '你的Tavily密钥', 'User')
```

保存用户环境变量后，重新打开终端或Visual Studio，使新进程继承配置。

读取优先级：构造器/CLI显式参数 → `TAVILY_API_KEY` → 项目 `tavily_api_key.txt`。环境变量存在时会覆盖文件内容。

Python接口：

```python
bot = RobotChatBot(tavily_api_key='你的Tavily密钥')
```

CLI `deepseek_robot.py --tavily-key ...` 同样支持，但会将密钥放在命令行中；通常更推荐本地文件或环境变量。

## 验证

在聊天里说：

```text
调用search_web搜索G1机器人视频动作模仿的相关资料。
```

返回授权失败时检查是否填写Tavily而非DeepSeek密钥、账户额度与有效性；网络错误需检查搜索服务连接。代码检查不能代替真实凭证请求，本次未使用用户密钥调用Tavily。

## 本次修复

- 修复Tavily配置错误回退到DeepSeek `api_key.txt` 的问题，彻底分离两个密钥来源。
- 统一聊天、动作候选搜索、后台任务的密钥解析，支持项目级专用文件和UTF-8 BOM。
- 修复搜索返回的content为null时格式化异常。
- 训练恢复时校验算法、观测维度、动作维度、关节顺序；训练参数非法时提前报错。
- 技能库JSON损坏时明确报错并拒绝覆盖，不再当空库处理。
- 修复模型XML读取未关闭文件的问题。
- 新增配置优先级、自动工具注册和技能库损坏保护测试。

测试命令：`python -m unittest discover -s tests -q`。
