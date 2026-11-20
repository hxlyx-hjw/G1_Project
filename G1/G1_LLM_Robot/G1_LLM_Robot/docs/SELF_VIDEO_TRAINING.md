# vedio_self/test1.mp4 训练与评估结果

## 输入和可学习区间

原视频位于项目上层 `vedio_self/test1.mp4`，时长11.35秒。前后为近景讲解、录屏控件或镜头切换，腿部不可见。自动选择最长连续全身有效区间：**原视频4.1816–10.8522秒**，持续6.6706秒，68帧。未将不可见腿部伪造为训练目标。

完整骨架：`motions/self_test1/pose_full.npz`；截取骨架：`pose.npz`，原始参考：`reference_raw.npz`，支撑修正参考：`reference.npz`，均位于该动作目录。

## 实际完成的训练

CUDA环境已验证，RTX 5070 Laptop，torch 2.11.0+cu128。没有新增安装依赖，现有环境已满足。物理仍使用CPU MuJoCo；GPU负责4个环境合并的网络推理和PPO更新，环境依次步进。

新增脚部落点、最小间距、足底朝向、支撑重心约束及平滑。纯PD仍会倒地，因此采用预训练G1腿部平衡策略 + 29维视频残差PPO。

### 稳定版（动作幅度受限）

40轮、20480控制样本；保留第30轮权重：

```text
runs/self_test1/training/policy.pt
```

独立物理回放完成334步/6.68秒，未摔倒，最大平面偏移0.0683米，整体关节RMSE0.1242rad、腿部0.1857rad、腰臂0.0444rad。

**重要限制：视频腿部偏移仅按0.15比例叠加，根部采用就地参考，没有复现原视频全幅横向步伐。** 未训练平衡先验也能完成此缩幅参考；PPO对误差改善很小，不应将稳定性全部归功于本次训练。

回放：`outputs/self_video/trained_replay.gif`。

```powershell
python -m fullbody evaluate motions/self_test1/reference.npz --checkpoint runs/self_test1/training/policy.pt --output runs/self_test1/view --viewer
python -m examples.render_self
```

### 完整腿部参考权重尝试（未成功）

CUDA训练总计103轮，最新权重在 `runs/self_test1/full_leg/latest.pt`，选取策略在同目录 `policy.pt`。训练内出现过未倒地但漂移约3.35米；独立评估选取权重约4.6秒倾倒，未完成。训练内与独立结果存在差异，不能认定成功。

```powershell
python -m fullbody evaluate motions/self_test1/reference.npz --checkpoint runs/self_test1/full_leg/policy.pt --output runs/self_test1/full_leg/view --viewer
```

不建议将失败版替换稳定版，也不保证继续增加轮数能解决。当前还缺少视频全局轨迹恢复、接触时序与落点约束的动力学优化、参考动作可行性检查以及更严格的漂移/动作相似度成功标准。训练器内部的completed只表示参考时间走完且未触发跌倒阈值，不代表动作质量合格。

## 复现与继续实验

从项目根目录执行：

```powershell
python -m video_motion extract ../vedio_self/test1.mp4 --output motions/self_test1/pose_full.npz --max-seconds 30 --fps 20
python -m fullbody.prepare_self
python -m fullbody retarget motions/self_test1/pose.npz --output motions/self_test1/reference_raw.npz
python -m fullbody.refine
python -m fullbody.train_balanced motions/self_test1/reference.npz --output runs/self_test1/new_training --iterations 40 --envs 4 --horizon 128 --device cuda --leg-weight 0.15
```

脚本 `prepare_self` 与 `refine` 的模块入口预设本视频路径；函数接口可用于其他输入。训练器支持 `--resume <latest.pt>`；`--iterations`为新增轮数。完整腿部参考权重使用 `--leg-weight 1.0`，目前没有合格结果。

日志位于 `outputs/self_video/`；独立评估和逐帧诊断位于 `runs/self_test1/final_evaluation/` 及 `runs/self_test1/full_leg/evaluation/`。

原有6项回归通过。当前交付是训练实现、稳定缩幅策略和完整腿部失败实验；**尚未完成原视频横向步伐的全幅动作学习目标**。
