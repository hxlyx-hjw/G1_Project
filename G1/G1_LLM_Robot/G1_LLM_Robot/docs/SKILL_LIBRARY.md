# 训练技能库

训练任务通过独立物理评估后，自动登记到 `skills/registry.json`。技能记录包括：技能ID、名称、来源任务、视频动作参考、策略权重、算法和评估报告。

只有同时满足 `reference_completed=true` 且 `fallen=false` 的训练结果才会登记。未完成或摔倒策略保留在训练目录中，但不会被 `robot_skill_run` 直接调用。

聊天示例：

```text
列出所有可用训练技能。
查询 skill_xxxxxxxxxx 的详细信息。
运行技能 skill_xxxxxxxxxx，并打开仿真窗口。
```

DeepSeek 工具：

- `robot_skill_list`：列出已登记技能。
- `robot_skill_get`：查询技能详情。
- `robot_skill_run`：直接用保存的 `.pt` 和 `.npz` 在 MuJoCo 中运行，并返回真实评估结果。
- `robot_train_video`：训练完成并通过评估后自动入库；可用 `skill_name` 指定名称。

技能仿真运行结果写入 `runs/skill_runs/<skill_id>_<随机ID>/`，不会覆盖技能库、训练权重或参考动作。运行默认打开三维窗口；设置 `viewer=false` 可无窗口运行。

手动检查：

```powershell
python -c "from fullbody.skills import list_skills; print(list_skills())"
python -m unittest discover -s tests -v
```

技能库只登记物理评估通过的策略，不能替代人工检查动作幅度、漂移和接触质量。当前根部仍采用就地参考，腿部权重小于1时属于缩幅动作。
