# 405错误修复说明

`ERROR 405 Method Not Allowed` 的原因是 DeepSeek 客户端曾被配置为：

```text
http://api.deepseek.com
```

DeepSeek API必须使用HTTPS。当前已统一修复为：

```text
https://api.deepseek.com
```

并加入启动校验：如果传入的 `base_url` 不是HTTPS，程序会在启动时明确提示，而不会发出错误请求。

启动：

```powershell
python Start.py --mode chat
```

或：

```powershell
python deepseek_robot.py --base-url https://api.deepseek.com
```

若使用Visual Studio，请确认启动参数中没有覆盖为 `http://api.deepseek.com`。Tavily搜索使用自身HTTPS接口，与DeepSeek的405问题无关。
