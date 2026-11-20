"""图形控制台子包。

把界面拆成四块，各管一件事：

* :mod:`~g1_walkfree.ui.theme`   尺寸/字体/配色 —— **"把界面放大"只改这一个文件**
* :mod:`~g1_walkfree.ui.widgets` 可复用大号控件（大滑杆、状态卡、滚动容器）
* :mod:`~g1_walkfree.ui.panel`   主面板：拼装布局 + 定时器驱动仿真
* :mod:`~g1_walkfree.ui.keymap`  键盘映射
"""

from .panel import WalkFreePanel, launch

__all__ = ["WalkFreePanel", "launch"]
