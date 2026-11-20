"""键盘映射：把按键接到面板的回调上。

设计要点
--------
* 按键**只改滑块的值**（再触发回调），不直接改控制器 —— 保证
  界面显示和实际指令永远一致，不会出现"控制器变了但滑杆没动"的错位。
* 大小写都绑定（``<KeyPress-w>`` 和 ``<KeyPress-W>``），Shift 按着也生效。
* 方向键：``←/→`` 切换左右臂，``↑/↓`` 微调"最后碰过的那只手臂关节"。
"""

from __future__ import annotations

import tkinter as tk

from .. import config as C


def bind_keys(root: tk.Tk, panel) -> None:
    """把快捷键绑到窗口上。

    参数
        root   Tk 根窗口
        panel  :class:`~g1_walkfree.ui.panel.WalkFreePanel` 实例
    """

    # ---- 速度：W/S 前后，A/D 左右，Q/E 转向 ----
    def speed(dvx: float = 0.0, dvy: float = 0.0, dwz: float = 0.0):
        panel.sl_vx.nudge(dvx)
        panel.sl_vy.nudge(dvy)
        panel.sl_wz.nudge(dwz)

    for key, args in {
        "w": dict(dvx=+C.KEY_STEP_VX), "s": dict(dvx=-C.KEY_STEP_VX),
        "a": dict(dvy=+C.KEY_STEP_VY), "d": dict(dvy=-C.KEY_STEP_VY),
        "q": dict(dwz=+C.KEY_STEP_WZ), "e": dict(dwz=-C.KEY_STEP_WZ),
    }.items():
        for variant in (key, key.upper()):
            root.bind("<KeyPress-%s>" % variant,
                      lambda _e, a=args: speed(**a))

    # ---- 急停 / 复位 / 退出 ----
    root.bind("<space>", lambda _e: panel.emergency_stop())
    for k in ("r", "R"):
        root.bind("<KeyPress-%s>" % k, lambda _e: panel.reset())
    for k in ("x", "X"):
        root.bind("<KeyPress-%s>" % k, lambda _e: panel.on_close())

    # ---- 手臂预设：数字键 1~5 ----
    for i, name in enumerate(list(C.ARM_PRESETS.keys())[:9]):
        root.bind("<KeyPress-%d>" % (i + 1),
                  lambda _e, n=name: panel.apply_preset(n))

    # ---- 左右臂切换 ----
    root.bind("<Left>", lambda _e: panel.switch_side("left"))
    root.bind("<Right>", lambda _e: panel.switch_side("right"))

    # ---- 微调当前手臂关节 ----
    root.bind("<Up>", lambda _e: panel.nudge_current_arm(+C.KEY_STEP_ARM))
    root.bind("<Down>", lambda _e: panel.nudge_current_arm(-C.KEY_STEP_ARM))

    # 让窗口一开始就拿到焦点，方向键/字母键才收得到
    root.focus_force()
