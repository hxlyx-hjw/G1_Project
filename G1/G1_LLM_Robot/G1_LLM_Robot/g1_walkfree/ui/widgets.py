"""可复用的放大版控件。

* :class:`LabeledSlider`  一行 = 标签 + 大滑杆 + 数值显示
* :class:`QuickButtonRow` 一排快捷指令按钮（自适应换行）
* :class:`StatusCard`     键值对状态卡，数字用等宽字体避免跳动
* :class:`ScrollableFrame` 可滚动容器（窗口不够高时不会挤掉控件）

这些控件都不认识"G1"、"行走"，只负责显示和回调，
方便单独测试、也方便以后复用到别的机器人上。
"""

from __future__ import annotations

import tkinter as tk
from typing import Callable

from . import theme as T


# ======================================================================
class LabeledSlider(tk.Frame):
    """标签 + 大滑杆 + 数值。

    用法::

        sl = LabeledSlider(parent, fonts, "前进 vx", -1.2, 1.2, 0.0,
                           fmt="%+.2f m/s", on_change=lambda v: ...)
        sl.set(0.5)
        val = sl.get()
    """

    def __init__(self, parent, fonts: T.Fonts,
                 label: str,
                 from_: float, to: float, value: float,
                 fmt: str = "%+.2f",
                 on_change: Callable[[float], None] | None = None,
                 label_width: int = T.Sizes.SLIDER_LABEL_W):
        super().__init__(parent, bg=T.Colors.BG)
        self._fmt = fmt
        self._on_change = on_change
        self._fonts = fonts

        # 左侧标签
        tk.Label(self, text=label, font=fonts.body, bg=T.Colors.BG,
                 fg=T.Colors.TEXT, width=label_width, anchor="w"
                 ).grid(row=0, column=0, sticky="w", padx=(0, T.PAD_INNER))

        # 中间滑杆
        self.var = tk.DoubleVar(value=value)
        self.scale = T.make_slider(self, fonts, from_, to, value,
                                   command=self._on_scale)
        self.scale.configure(variable=self.var)
        self.scale.grid(row=0, column=1, sticky="w")

        # 右侧数值
        self.value_label = tk.Label(self, text=fmt % value, font=fonts.value,
                                    bg=T.Colors.BG, fg=T.Colors.KNOB,
                                    width=T.Sizes.SLIDER_VALUE_W, anchor="e")
        self.value_label.grid(row=0, column=2, sticky="e", padx=(T.PAD_INNER, 0))

    # ---- 内部：滑杆被拖动 ----
    def _on_scale(self, raw: str) -> None:
        v = float(raw)
        self.value_label.config(text=self._fmt % v)
        if self._on_change:
            self._on_change(v)

    # ---- 外部接口 ----
    def get(self) -> float:
        return float(self.var.get())

    def set(self, value: float) -> None:
        """外部改值（会同步刷新数字，但**不会**触发 on_change，避免递归）。"""
        self.var.set(value)
        self.value_label.config(text=self._fmt % float(value))

    def nudge(self, delta: float) -> None:
        """在当前值上增减，并**会**触发 on_change（键盘微调用）。"""
        lo, hi = self.scale.cget("from"), self.scale.cget("to")
        v = min(max(float(self.var.get()) + delta, float(lo)), float(hi))
        self.set(v)
        if self._on_change:
            self._on_change(v)


# ======================================================================
class QuickButtonRow(tk.Frame):
    """一排快捷按钮，超过每行上限自动换行。"""

    def __init__(self, parent, fonts: T.Fonts,
                 items: list[tuple[str, Callable[[], None]]],
                 per_row: int = 4):
        super().__init__(parent, bg=T.Colors.BG)
        for i, (text, cmd) in enumerate(items):
            r, c = divmod(i, per_row)
            tk.Button(self, text=text, command=cmd,
                      font=fonts.hint, relief="groove", bd=1,
                      bg=T.Colors.CARD_BG, activebackground=T.Colors.TRACK,
                      padx=T.Sizes.BUTTON_PAD_X, pady=6,
                      width=T.Sizes.BUTTON_W_SMALL
                      ).grid(row=r, column=c, padx=4, pady=4, sticky="ew")
        for c in range(per_row):
            self.grid_columnconfigure(c, weight=1)


# ======================================================================
class StatusCard(tk.Frame):
    """键值对状态卡。

    ``update({...})`` 传入字典：键是行名，值是已格式化好的字符串。
    """

    def __init__(self, parent, fonts: T.Fonts, keys: list[str]):
        super().__init__(parent, bg=T.Colors.CARD_BG,
                         highlightthickness=1, highlightbackground=T.Colors.TRACK)
        self._values: dict[str, tk.Label] = {}
        for i, k in enumerate(keys):
            tk.Label(self, text=k, font=fonts.status_key, bg=T.Colors.CARD_BG,
                     fg=T.Colors.MUTED, anchor="w"
                     ).grid(row=i, column=0, sticky="w",
                            padx=(T.PAD_INNER, 6), pady=4)
            v = tk.Label(self, text="—", font=fonts.status, bg=T.Colors.CARD_BG,
                         fg=T.Colors.TEXT, anchor="w")
            v.grid(row=i, column=1, sticky="w", padx=(0, T.PAD_INNER), pady=4)
            self._values[k] = v
        self.grid_columnconfigure(1, weight=1)

    def update_values(self, data: dict[str, str], warn: bool = False) -> None:
        """刷新所有值；``warn=True`` 时整体变红（用于"已摔倒"）。"""
        for k, v in data.items():
            lbl = self._values.get(k)
            if lbl is not None:
                lbl.config(text=v, fg=T.Colors.WARN if warn else T.Colors.TEXT)


# ======================================================================
class ScrollableFrame(tk.Frame):
    """竖向可滚动容器。

    把内容放进 ``.body``；窗口高度不够时可以滚动查看，不会挤掉控件。
    """

    def __init__(self, parent, bg: str = T.Colors.BG):
        super().__init__(parent, bg=bg)

        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self, orient="vertical",
                                      command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.body = tk.Frame(self.canvas, bg=bg)
        self._win = self.canvas.create_window((0, 0), window=self.body, anchor="nw")

        # 内容尺寸变化时更新滚动区域
        self.body.bind("<Configure>", self._on_body_configure)
        # 容器尺寸变化时让内容宽度跟随
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        # 鼠标滚轮（Windows/macOS 用 <MouseWheel>，Linux 用 Button-4/5）
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)

    def _on_body_configure(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event) -> None:
        self.canvas.itemconfigure(self._win, width=event.width)

    def _on_wheel(self, event) -> None:
        # 只在鼠标位于本区域时滚动，避免抢别的滚动条
        widget = self.winfo_containing(event.x_root, event.y_root)
        node = widget
        while node is not None:
            if node is self:
                step = -1 * (event.delta // 120) if event.delta else 0
                self.canvas.yview_scroll(step, "units")
                return
            node = getattr(node, "master", None)
