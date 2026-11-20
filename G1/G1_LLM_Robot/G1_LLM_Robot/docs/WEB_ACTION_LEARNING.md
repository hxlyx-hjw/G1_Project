# Harness + 网络视频自主动作学习

最新搜索已统一到原Harness Tavily，DDGS/Bing回退已移除；权重自检和可选模型复核说明见 [SELF_CHECK_AND_SEARCH.md](SELF_CHECK_AND_SEARCH.md)。下文旧公共回退描述以此更新为准。

现在可以把 Harness 的联网搜索、机器人训练和技能库串成一条后台通路：

```text
动作描述
  → Tavily / 公共视频搜索
  → 候选URL
  → 下载或yt-dlp解析
  → 视频解码检查
  → MediaPipe全身可见性检查
  → 全身重定向与足部支撑修正
  → CUDA PPO训练
  → 独立MuJoCo物理评估
  → 通过后登记skills/registry.json
  → DeepSeek后续直接调用技能仿真
```

## 聊天用法

启动：

```powershell
python Start.py --mode chat
```

直接让模型自主搜索和训练：

```text
学习“站立挥手”这个动作：搜索网上的视频，下载能检测到单人全身的候选，使用GPU训练100轮，打开渲染窗口。训练通过物理评估后加入技能库。
```

也可以先让 Harness 搜索，再把候选URL传给训练工具：

```text
先搜索“单人全身站立挥手动作视频”，列出候选。
使用第一个候选训练这个动作，腿部初始权重0.15，课程目标0.5。
```

新增工具：

- `robot_search_action_videos(action, limit?)`：使用已配置的Tavily；没有Tavily时尝试DDGS和Bing RSS回退，返回候选URL及来源。
- `robot_learn_action(action, source_urls?, ...)`：自主搜索或使用给定候选，后台下载、筛选和训练，立即返回job_id。
- 原有 `robot_training_status`、`robot_training_stop`、`robot_training_viewer` 继续用于任务控制。
- 训练通过评估后原有 `robot_skill_list`、`robot_skill_get`、`robot_skill_run` 可以调用技能。

训练任务可以指定 `source_urls`，这是更可靠的方式：Harness先搜索并由模型挑选候选，再把URL传给学习工具。URL必须为http/https；最多10个候选，最多尝试配置的 `max_candidates` 个。

## 候选视频筛选

每个候选都会记录到任务目录的 `search.json` 和 `candidate_attempts.json`。下载预算为单视频200MB、180秒；长于10分钟的视频被跳过。候选必须能解码，并且必须存在至少0.5秒的连续肩、髋、膝、踝关键点可见区间。下载失败、不能解码、身体被遮挡或没有连续全身动作时自动尝试下一个候选。

通过筛选的真实来源写入 `source.json`，包括来源URL、下载方式、上传者/许可证（如果下载器提供）、本地文件、视频时长、搜索候选描述和筛选区间；训练技能记录也保留这份来源信息。

搜索标题和描述只用于候选排序，代码不会把标题当作动作标签。`semantic_match_verified=false` 会一直保留在技能记录中：物理评估通过说明机器人完成了当前参考片段，不证明视频内容确实是用户描述的动作。动作语义核实需要视觉动作分类或人工确认，当前未实现。

## GPU和后台行为

`robot_learn_action` 默认 `device=auto`，CUDA可用时使用GPU；当前GPU用于PPO网络更新，MuJoCo物理和MediaPipe仍主要运行在CPU。`device=cuda`但环境不可用会失败并返回状态，不会静默切到CPU。

训练、搜索和下载均在后台进程完成，聊天可继续工作。渲染窗口显示任务阶段和最近保存策略的物理回放。用 `robot_training_status` 查看真实状态；只有 `completed` 且 `evaluation.fallen=false`、`evaluation.reference_completed=true` 时才会入技能库。

默认腿部权重0.15是稳定缩幅；设置为1.0只是全幅参考尝试，不能保证成功。课程目标只有在多次整段评估达标后逐级扩大。当前根部采用就地参考，不能恢复网上视频中的真实全局位移。

## 技能库

通过评估的技能存入：

```text
skills/registry.json
```

技能包含：

- `checkpoint`：策略权重。
- `reference`：全身参考动作。
- `source`：搜索URL、候选和筛选区间。
- `evaluation`：摔倒、完成、漂移、关节误差。
- `leg_weight`：实际腿部参考权重。
- `semantic_match_verified=false`：动作语义尚未由分类器确认。

调用示例：

```text
列出技能库中的技能。
运行 skill_6291e8dccb，并打开仿真窗口。
```

## 真实验证

已通过本地HTTP模拟Harness视频候选完成完整链路：

- 视频候选下载成功。
- CUDA设备：NVIDIA GeForce RTX 5070 Laptop GPU。
- 1轮短训练成功。
- 独立评估：2.0秒参考片段完成，未摔倒。
- 技能自动登记：`skill_6291e8dccb`。
- 当时19项离线回归测试通过，后续增加搜索过滤与后台任务测试。

公网搜索实测：DDGS返回网络错误，Bing RSS可访问但曾返回无关词典结果，已新增视频页过滤。因此尚未验证任意动作的公网搜索→合适视频获取；已验证的是给定HTTP候选后的完整训练入库通路。建议配置Tavily或由Harness提供明确视频候选。

这次验证使用官方姿态示例作为传输候选，目的为验证通路；技能记录明确标记动作语义未核实。真正使用时应优先选择固定机位、单人、全身无遮挡且动作持续完整的视频。

依赖：`yt-dlp` 已安装并加入 [requirements-fullbody.txt](../requirements-fullbody.txt)。Tavily仍通过 `TAVILY_API_KEY` 配置；密钥不会写入任务配置或技能库。
