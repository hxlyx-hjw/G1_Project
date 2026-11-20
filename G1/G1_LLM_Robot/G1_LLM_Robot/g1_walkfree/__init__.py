"""宇树 G1 · 自由可控行走 + 手臂控制（模块化版本）。

这个包把「参考三个文件」得到的结论固化下来：
    stand_g1.py / stand_g1_pd.py  -> 站稳靠模型的 position 执行器 + 投影重力 PD
    stand_g1_rl.py                -> 行走靠 unitree_rl_gym 的 12 自由度预训练策略

本包的设计是**融合**：把 menagerie 的 29 自由度模型改造成
「腿部 = 力矩电机（吃 RL 策略输出） + 腰/手臂 = 位置伺服（自由控制）」，
于是行走和手臂可以同时自由操控，平衡由策略本身保证。

模块划分
--------
config      所有常量与默认参数（策略超参、姿态、预设、上限）—— 集中一处方便调
resources   运行时资源定位（menagerie / 策略权重）与模型改造（生成临时 MJCF）
controller  WalkController：策略推理 + PD 力矩 + 手臂位置目标，唯一的"业务核心"
runners     headless / 3D 窗口 两种运行方式
ui          图形控制台（放大版界面）：主题、控件、主面板、键盘映射
"""

__version__ = "2.0"
__all__ = ["config", "resources", "controller", "runners"]
