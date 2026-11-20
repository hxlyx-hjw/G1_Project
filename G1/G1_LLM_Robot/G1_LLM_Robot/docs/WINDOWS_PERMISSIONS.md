# WinError 5 处理说明

项目已加入Windows安全写入和日志机制：

- 训练权重先写入唯一临时文件，再用带重试的原子替换更新目标文件。
- 训练状态和配置 JSON 也使用唯一临时文件与带重试的原子替换，不再共用
  `status.tmp`；状态写入失败只记录警告，不会中断训练。
- 技能库 `skills/registry.json` 同样使用安全写入。
- 读取权重遇到短暂占用会重试。
- 工具异常写入 `logs/app.log`，控制台仍返回简短错误。
- 查看器加载新权重时不直接覆盖正在读取的文件。
- `Start_G1.bat` 会固定项目目录后启动，避免Visual Studio/双击时工作目录错误。

WinError 5 常见来源是文件被查看器、杀毒程序、同步软件占用，或目标目录没有写权限。程序默认不强制管理员运行，因为管理员权限不能解除另一个进程的文件占用。请先关闭旧的G1窗口、训练进程和GIF导出进程，再重试。

如果确定是目录权限问题：右键项目文件夹 → 属性 → 安全 → 给当前Windows用户授予“修改/写入”权限；不要直接给所有用户完全控制。也可以右键 `Start_G1.bat` → “以管理员身份运行”，但应只作为权限排查手段。

可选管理员启动脚本：

```powershell
Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-Command',('Set-Location -LiteralPath "'+(Get-Location).Path+'"; python Start.py')
```

程序日志位置优先是项目 `logs/app.log`，项目目录不可写时回退到 `%LOCALAPPDATA%\G1Robot\logs\app.log`。

当前仍需要避免同时打开同一个输出文件的外部编辑器；训练和查看器本身已采用临时文件、原子替换和读取重试。
