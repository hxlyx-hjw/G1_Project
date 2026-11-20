# 原Harness搜索统一与训练权重自检

## 搜索

已删除动作学习的DDGS/Bing独立回退。聊天 `search_web` 和 `robot_search_action_videos`、后台 `robot_learn_action` 统一调用 `harness_search.py` 中的原有Tavily advanced搜索，使用同一个 `TAVILY_API_KEY` 或构造器传入的 `tavily_api_key`。配置存在时Start聊天自动启用原搜索工具。后台密钥只通过环境传递，不写任务JSON。

本次环境检查未发现TAVILY_API_KEY，因而未进行成功的公网Tavily实测。没有密钥会直接报配置缺失，不再返回无关公共搜索结果。不要把DeepSeek API Key当作Tavily Key。可继续使用原Harness已有配置，或提供明确候选URL绕过搜索。

## 自检与自调节

新增 `fullbody/self_check.py`，`robot_train_video` 和 `robot_learn_action` 支持：

- `auto_adjust=true`：默认开启，每10轮训练自检。
- `arm_weight=1.0`：肩肘腕参考幅度，范围0.25~1.5；不是机器人质量或PD刚度。
- `leg_weight=0.15`：原有腿部参考幅度，范围0~2，仍受关节限位和平衡包络约束。
- `motion_gain=1.35`：平衡感知动作放大系数，范围1~2；稳定时放大、失衡趋势出现时自动收回。
- `model_review=false`：可选。开启后，异常指标持续到第30轮倍数的检查时，最多调用现有DeepSeek配置一次提出权重建议。

自检流程：

1. 从参考起点执行完整物理回放，分别统计腿部MSE、手臂MSE、是否摔倒、最大漂移和速度。
2. 失稳时分别尝试腿幅下降0.05、手幅下降0.1；稳定但原参考跟踪误差大时尝试小步增加。
3. 对每个候选实际完整回放，以完成率、漂移和原始参考误差比较，改善不足则不采用。
4. 模型只能建议权重，单次变化限制在±0.1，并夹到允许范围；同样必须通过物理对照。模型超时、JSON非法或未配置时记录原因，继续规则自检。
5. 记录 `training/self_checks.jsonl`、metrics及状态；调整后的两组权重写入checkpoint。独立评估、渲染、GIF和技能库均恢复实际权重。

PPO损失保持原有定义，本次调整的是参考动作幅度，不是奖励系数。手臂自检覆盖肩肘腕，不能从MediaPipe骨架恢复缺失的手腕旋转。只降低幅度获得稳定不代表学会原动作。

聊天示例：

```text
训练 ../vedio_self/test1.mp4，使用GPU，开启auto_adjust。必要时允许model_review调用DeepSeek提出手腿权重建议，保持渲染，并逐次解释自检结果。
```

现有API Key、base_url、model由聊天实例通过子进程环境传递；配置文件中不存密钥。模型复核会增加API调用。默认关闭，避免每次物理评估都调用模型。

## 验证与限制

- 10轮CUDA短训练触发实际自检，执行1个候选完整物理回放；候选未改善，保留腿权重0.15、手臂权重1.0。这符合“不为了调整而调整”的规则。
- 自动测试验证原Harness客户端参数一致、改善候选才能采用、模型建议不能绕过物理验证。
- 本次未调用付费模型做真实建议，模型接口通过模拟响应约束测试；不能宣称模型建议能保证收敛。
- 自检只比较同一参考初态，不是因果证明或多种子鲁棒性认证；无法代替接触轨迹、重心规划、参考质量改进。

文件：`harness_search.py`、`fullbody/self_check.py`、`train_balanced.py`；聊天参数见 `fullbody/tools.py`。
