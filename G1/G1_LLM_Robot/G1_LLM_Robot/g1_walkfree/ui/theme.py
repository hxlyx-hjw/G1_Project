"""界面主题：**所有尺寸、字体、间距、配色集中在这里**。

想"把界面再放大一点"或"换成深色"，只需要改本文件，其余 UI 代码不用动。

为什么不用 ttk 自带的 Scale？
------------------------------
ttk 的滑杆很难加粗（trough 厚度基本固定），而我们要的是**大号、好点**的滑杆。
所以滑杆用经典 ``tk.Scale``：它直接支持 ``length`` / ``width`` /
``sliderlength`` / ``font``，做大非常方便。其余控件继续用 ttk 以保持原生观感。
"""

from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

# ======================================================================
# 一、整体尺寸
# ======================================================================
#: 主窗口**期望**大小（宽 x 高）。实际会用 :func:`compute_layout` 按屏幕裁剪。
WINDOW_SIZE = (1560, 950)

#: 最小尺寸：拖小到这个尺寸以下就不让再缩了，避免控件被挤成一团
WINDOW_MIN_SIZE = (1080, 760)

#: 窗口四周留白
PAD_OUTER = 12

#: 同一区域内控件之间的间距
PAD_INNER = 8

#: 分栏之间的间距
PAD_GUTTER = 14

#: 屏幕边缘至少留出这么多像素（给任务栏/窗口边框）
SCREEN_MARGIN_X = 60
SCREEN_MARGIN_Y = 90


# ======================================================================
# 二、字体
# ======================================================================
#: 按优先级挑选中英文都好看的字体族；找不到就退回 Tk 默认
_FONT_FAMILY_CANDIDATES = [
    "Microsoft YaHei UI",   # Windows 10/11 自带，中文最舒适
    "Microsoft YaHei",
    "PingFang SC",          # macOS
    "Noto Sans CJK SC",     # Linux
    "Segoe UI",
    "Helvetica",
]

#: 等宽字体（状态数值对齐全靠它）
_MONO_CANDIDATES = ["Cascadia Mono", "Consolas", "DejaVu Sans Mono",
                    "Menlo", "Courier New"]


def _pick_family(candidates, root):
    """返回候选列表里系统真正存在的第一个字体族。"""
    try:
        available = {f.lower() for f in tkfont.families(root)}
    except Exception:
        available = set()
    for name in candidates:
        if name.lower() in available:
            return name
    return candidates[-1]


class Fonts:
    """字体尺寸表。数值已按"看得清"调过，比默认大 30%~60%。"""

    def __init__(self, root):
        fam = _pick_family(_FONT_FAMILY_CANDIDATES, root)
        mono = _pick_family(_MONO_CANDIDATES, root)

        self.family = fam
        self.mono_family = mono

        #: 分区标题（LabelFrame 的标题）
        self.section = (fam, 13, "bold")
        #: 正文标签
        self.body = (fam, 12)
        #: 小号说明文字
        self.hint = (fam, 11)
        #: 按钮文字
        self.button = (fam, 12, "bold")
        #: 主标题
        self.title = (fam, 17, "bold")
        #: 状态数值（等宽，方便数字跳动时不抖动）
        self.status = (mono, 13)
        #: 状态标题
        self.status_key = (mono, 12)
        #: 大号数值（滑杆右侧）
        self.value = (mono, 13, "bold")


# ======================================================================
# 三、控件尺寸
# ======================================================================
class Sizes:
    """各种控件的像素尺寸/行高。"""

    # ---- 滑杆 ----
    SLIDER_LENGTH = 400        # 滑轨长度（会被 compute_layout 按屏幕宽度微调）
    SLIDER_WIDTH = 26          # 滑轨厚度（越大越好点）
    SLIDER_KNOB = 34           # 滑块长度
    SLIDER_BORDER = 2          # 滑轨描边
    SLIDER_LABEL_W = 18        # 左侧标签宽度（字符数）
    SLIDER_VALUE_W = 8         # 右侧数值宽度（字符数）

    # ---- 按钮 ----
    BUTTON_PAD_X = 14          # 文字左右内边距
    BUTTON_PAD_Y = 8           # 文字上下内边距
    BUTTON_W_SMALL = 10        # 快捷按钮宽度（字符）
    BUTTON_W_MED = 14          # 一般按钮宽度

    # ---- 状态卡 ----
    STATUS_LINE_H = 24         # 每行行高
    STATUS_ROWS = 8            # 行数

    # ---- 窗口内部 ----
    LEFT_COL_W = 520           # 左栏（行走 + 状态）最小宽度
    RIGHT_COL_W = 520          # 右栏（手臂）最小宽度


#: 一个 :class:`LabeledSlider` 除滑杆本身以外占掉的像素（标签 + 数值 + 内边距）。
#: 实测值（Microsoft YaHei UI 12 / 等宽 13 放大后）：标签约 190px、数值约 90px、
#: 再加内边距与边框约 30px。用经验值估算，避免去测量字体度量。
SLIDER_CHROME_PX = 330


def compute_layout(root: tk.Misc) -> tuple[tuple[int, int], int]:
    """按当前屏幕尺寸算出「窗口多大」和「滑杆多长」。

    两栏布局的总宽度 = 2 * (滑杆长 + 非滑杆部分) + 栏间距 + 页面留白。
    屏幕装不下时优先缩短滑杆（最小 220px），而不是让控件被裁掉。

    返回
        ((窗口宽, 窗口高), 滑杆长度)
    """
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    avail_w = sw - SCREEN_MARGIN_X
    avail_h = sh - SCREEN_MARGIN_Y

    length = Sizes.SLIDER_LENGTH
    while length > 220:
        total = (2 * (SLIDER_CHROME_PX + length) + PAD_GUTTER
                 + 2 * PAD_OUTER + 16)
        if total <= avail_w:
            break
        length -= 20

    total_w = 2 * (SLIDER_CHROME_PX + length) + PAD_GUTTER + 2 * PAD_OUTER + 16
    win_w = int(min(WINDOW_SIZE[0], avail_w))
    win_w = max(win_w, min(total_w, avail_w))
    win_w = max(win_w, WINDOW_MIN_SIZE[0])
    win_h = int(max(min(WINDOW_SIZE[1], avail_h), WINDOW_MIN_SIZE[1]))
    return (win_w, win_h), int(length)


# ======================================================================
# 四、配色
# ======================================================================
class Colors:
    """少量用到的颜色；ttk 会沿用系统主题，这里只给经典控件和强调用。"""

    BG = "#f4f5f7"             # 窗口底色
    CARD_BG = "#ffffff"        # 卡片底
    TRACK = "#d7dae0"          # 滑杆轨道
    KNOB = "#3d6df0"           # 滑杆滑块
    TEXT = "#1f2329"
    MUTED = "#7a808a"
    WARN = "#d9480f"           # 摔倒提示
    OK = "#2f9e44"             # 正常提示


# ======================================================================
# 五、ttk 样式安装
# ======================================================================
def install(root: tk.Tk) -> Fonts:
    """创建字体表并把字号应用到 ttk 控件上，返回字体表。

    调用一次即可（通常在创建主窗口后立即调用）。
    """
    fonts = Fonts(root)
    style = ttk.Style(root)

    # Windows 上用 "vista" 观感最好；找不到就退回默认
    try:
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass

    # ---- 逐类放大字号 ----
    style.configure(".", font=fonts.body)
    style.configure("TLabel", font=fonts.body)
    style.configure("TButton", font=fonts.button,
                    padding=(Sizes.BUTTON_PAD_X, Sizes.BUTTON_PAD_Y))
    style.configure("TCheckbutton", font=fonts.body, padding=(6, 4))
    style.configure("TRadiobutton", font=fonts.body, padding=(8, 4))
    style.configure("TLabelframe", padding=PAD_INNER, borderwidth=1, relief="solid")
    style.configure("TLabelframe.Label", font=fonts.section)
    style.configure("TNotebook.Tab", font=fonts.section, padding=(18, 10))
    style.configure("Hint.TLabel", font=fonts.hint, foreground=Colors.MUTED)
    style.configure("Body.TLabel", font=fonts.body, foreground=Colors.TEXT)
    style.configure("Title.TLabel", font=fonts.title, foreground=Colors.TEXT)
    style.configure("Section.TLabel", font=fonts.section, foreground=Colors.TEXT)
    style.configure("Status.TLabel", font=fonts.status, foreground=Colors.TEXT)
    style.configure("StatusKey.TLabel", font=fonts.status_key, foreground=Colors.MUTED)
    style.configure("Warn.TLabel", font=fonts.status, foreground=Colors.WARN)
    style.configure("Ok.TLabel", font=fonts.status, foreground=Colors.OK)

    root.configure(bg=Colors.BG)
    return fonts


def make_slider(parent: tk.Widget, fonts: Fonts,
                from_: float, to: float, value: float,
                command=None) -> tk.Scale:
    """创建一个统一风格的大号水平滑杆。

    用经典 ``tk.Scale`` 而不是 ``ttk.Scale``：后者很难把滑轨加粗、把滑块做大，
    而 ``tk.Scale`` 直接支持 ``length`` / ``width`` /
    ``sliderlength`` / ``font``，做大非常方便。

    参数
        parent   父容器
        fonts    字体表（用于滑杆上的刻度/数值文字）
        from_    最小值
        to       最大值
        value    初始值
        command 值变化时的回调，签名 ``command(str_value)``
    """
    s = tk.Scale(
        parent,
        from_=from_, to=to,
        orient="horizontal",
        resolution=0.01,                 # 保留两位小数
        length=Sizes.SLIDER_LENGTH,
        width=Sizes.SLIDER_WIDTH,        # 滑轨厚度
        sliderlength=Sizes.SLIDER_KNOB,  # 滑块长度
        bd=Sizes.SLIDER_BORDER,
        relief="flat",
        troughcolor=Colors.TRACK,
        bg=Colors.KNOB,
        activebackground=Colors.KNOB,
        highlightthickness=0,
        font=fonts.hint,
        command=command,
    )
    s.set(value)
    return s
