"""可复用控件：卡片、状态胶囊、聊天气泡、工具调用卡、思考指示器。"""

from __future__ import annotations

import html
import json
import re
import time

from PyQt5.QtCore import QPoint, QRect, QSize, Qt, QTimer
from PyQt5.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPlainTextEdit,
    QProgressBar,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import theme as T


# ----------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------
def repolish(widget: QWidget) -> None:
    """修改动态属性后，强制 Qt 重新套用样式。"""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def hline() -> QFrame:
    line = QFrame()
    line.setProperty("role", "hline")
    line.setFixedHeight(1)
    return line


def rich_text(text: str) -> str:
    """把模型输出里常见的 Markdown 轻量渲染成 HTML。"""
    text = html.escape(str(text or ""))
    text = re.sub(
        r"`([^`\n]+)`",
        lambda m: '<span style="font-family:%s;background:#0f151d;'
        'padding:1px 4px;border-radius:4px;">%s</span>' % (T.FONT_MONO, m.group(1)),
        text,
    )
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)

    out: list[str] = []
    for line in text.split("\n"):
        heading = re.match(r"^(#{1,3})\s+(.*)$", line)
        bullet = re.match(r"^\s*[-*]\s+(.*)$", line)
        numbered = re.match(r"^\s*(\d+[.)])\s+(.*)$", line)
        if heading:
            level = len(heading.group(1))
            size = {1: 15, 2: 14, 3: 13}.get(level, 13)
            out.append(
                '<div style="font-size:%dpx;font-weight:700;margin:8px 0 3px 0;">%s</div>'
                % (size, heading.group(2))
            )
        elif bullet:
            out.append('<div style="margin:2px 0 2px 12px;">• %s</div>' % bullet.group(1))
        elif numbered:
            out.append(
                '<div style="margin:2px 0 2px 12px;">%s %s</div>'
                % (numbered.group(1), numbered.group(2))
            )
        elif line.strip():
            out.append(line + "<br>")
        else:
            out.append("<br>")
    return "".join(out)


def _short(text: str, limit: int) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " …"


def elide_path(path: str, width: int = 52) -> str:
    """路径太长时保留开头盘符和结尾目录，中间用省略号。"""
    text = str(path or "")
    if len(text) <= width:
        return text
    head = text[:10].rstrip("\\/")
    tail = text[-(width - len(head) - 1):]
    return head + "…" + tail


class FlowLayout(QLayout):
    """自动换行的流式布局（用于快捷指令按钮）。"""

    def __init__(self, parent: QWidget | None = None, hspacing: int = 8,
                 vspacing: int = 8):
        super().__init__(parent)
        self._items: list = []
        self._hspacing = hspacing
        self._vspacing = vspacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item) -> None:  # noqa: N802
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientations(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(
            margins.left() + margins.right(), margins.top() + margins.bottom()
        )

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        margins = self.contentsMargins()
        effective = rect.adjusted(
            margins.left(), margins.top(), -margins.right(), -margins.bottom()
        )
        x, y = effective.x(), effective.y()
        line_height = 0
        for item in self._items:
            hint = item.sizeHint()
            if x + hint.width() > effective.right() + 1 and line_height > 0:
                x = effective.x()
                y += line_height + self._vspacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self._hspacing
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + margins.bottom()


# ----------------------------------------------------------------------
# 基础容器
# ----------------------------------------------------------------------
class Card(QFrame):
    """带内边距的圆角卡片，可带标题。"""

    def __init__(self, title: str = "", parent: QWidget | None = None, padding: int = 16):
        super().__init__(parent)
        self.setProperty("card", True)
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(padding, padding, padding, padding)
        self.body.setSpacing(10)
        if title:
            header = QLabel(title)
            header.setProperty("role", "section")
            self.body.addWidget(header)


class Chip(QLabel):
    """状态胶囊；state ∈ idle / ok / warn / danger / info。"""

    def __init__(self, text: str = "", state: str = "idle", parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setProperty("chip", True)
        self.setAlignment(Qt.AlignCenter)
        self.set_state(state, text)

    def set_state(self, state: str, text: str | None = None) -> None:
        if text is not None:
            self.setText(text)
        self.setProperty("state", state)
        repolish(self)


class MetricCard(QFrame):
    """指标卡：标签 + 大号数值 + 可选进度条。"""

    def __init__(self, label: str, value: str = "—", unit: str = "",
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("card", True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(4)

        self.label = QLabel(label)
        self.label.setProperty("role", "metricLabel")
        lay.addWidget(self.label)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.value = QLabel(value)
        self.value.setProperty("role", "metricValue")
        row.addWidget(self.value)
        self.unit = QLabel(unit)
        self.unit.setProperty("role", "metricLabel")
        self.unit.setAlignment(Qt.AlignBottom)
        row.addWidget(self.unit)
        row.addStretch(1)
        lay.addLayout(row)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        self.bar.hide()
        lay.addWidget(self.bar)

    def set_value(self, value: str, unit: str | None = None,
                  percent: float | None = None, state: str = "") -> None:
        self.value.setText(value)
        if unit is not None:
            self.unit.setText(unit)
        if percent is None:
            self.bar.hide()
        else:
            self.bar.show()
            self.bar.setValue(max(0, min(100, int(round(percent)))))
            if state:
                self.bar.setProperty("state", state)
                repolish(self.bar)


# ----------------------------------------------------------------------
# 聊天
# ----------------------------------------------------------------------
MAX_BUBBLE = 780


class MessageBubble(QWidget):
    """一条聊天消息。role ∈ user / assistant / system / proactive。"""

    def __init__(self, role: str, text: str = "", reasoning: str = "",
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.role = role
        self._reasoning = reasoning

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        bubble = QFrame()
        bubble.setProperty("bubble", True)
        bubble.setProperty("role", role)
        bubble.setMaximumWidth(MAX_BUBBLE)
        bubble.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum)
        inner = QVBoxLayout(bubble)
        inner.setContentsMargins(14, 11, 14, 11)
        inner.setSpacing(6)

        head = QLabel()
        head.setProperty("role", "hint")
        who = {
            "user": "你",
            "assistant": "G1 助手",
            "system": "系统",
            "proactive": "训练监护 · 主动报告",
        }.get(role, "G1")
        head.setText(who + " · " + time.strftime("%H:%M"))
        inner.addWidget(head)

        if reasoning:
            toggle = QToolButton()
            toggle.setText("▸ 思考过程")
            toggle.setCheckable(True)
            toggle.setChecked(False)
            toggle.setCursor(Qt.PointingHandCursor)
            toggle.setStyleSheet("QToolButton { color: %s; }" % T.ACCENT)
            toggle.toggled.connect(
                lambda checked: toggle.setText(
                    ("▾ " if checked else "▸ ") + "思考过程"
                )
            )
            reasoning_label = QLabel(rich_text(reasoning))
            reasoning_label.setWordWrap(True)
            reasoning_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            reasoning_label.setStyleSheet("color: %s; font-size: 9pt;" % T.TEXT_DIM)
            reasoning_label.hide()
            toggle.toggled.connect(reasoning_label.setVisible)
            inner.addWidget(toggle)
            inner.addWidget(reasoning_label)

        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setTextFormat(Qt.RichText)
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        width = self._content_width(text)
        self.label.setFixedWidth(width)
        self.label.setText(rich_text(text))
        inner.addWidget(self.label)

        if reasoning:
            reasoning_label.setFixedWidth(width)

        if role == "user":
            outer.addStretch(1)
            outer.addWidget(bubble)
        else:
            avatar = QLabel({"assistant": "G1", "proactive": "!"}.get(role, "i"))
            avatar.setProperty("role", "avatar")
            avatar.setProperty("who", {"assistant": "g1", "user": "user"}.get(role, "sys"))
            avatar.setAlignment(Qt.AlignCenter)
            avatar.setFixedSize(28, 28)
            outer.addWidget(avatar, 0, Qt.AlignTop)
            outer.addWidget(bubble)
            outer.addStretch(1)

        self._fade_in(bubble)

    def _content_width(self, text: str) -> int:
        """按最长一行估算气泡宽度，短消息不拉满、长消息在 744px 处换行。"""
        metrics = self.label.fontMetrics()
        longest = 0
        for line in str(text or "").splitlines() or [""]:
            longest = max(longest, metrics.horizontalAdvance(line))
        return max(220, min(MAX_BUBBLE - 36, longest + 16))

    @staticmethod
    def _fade_in(widget: QWidget) -> None:
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
        from PyQt5.QtCore import QEasingCurve, QPropertyAnimation

        anim = QPropertyAnimation(effect, b"opacity", widget)
        anim.setDuration(180)
        anim.setStartValue(0.25)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.finished.connect(lambda: widget.setGraphicsEffect(None))
        widget._fade_anim = anim  # type: ignore[attr-defined]
        anim.start()

    def append_text(self, text: str) -> None:
        self.label.setText(self.label.text() + rich_text(text))


class ThinkingBubble(QWidget):
    """等待模型回复时的呼吸提示。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        avatar = QLabel("G1")
        avatar.setProperty("role", "avatar")
        avatar.setProperty("who", "g1")
        avatar.setAlignment(Qt.AlignCenter)
        avatar.setFixedSize(28, 28)

        bubble = QFrame()
        bubble.setProperty("bubble", True)
        bubble.setProperty("role", "assistant")
        lay = QHBoxLayout(bubble)
        lay.setContentsMargins(14, 9, 14, 9)
        self.label = QLabel("正在思考")
        self.label.setStyleSheet("color: %s;" % T.TEXT_MUTED)
        lay.addWidget(self.label)

        outer.addWidget(avatar, 0, Qt.AlignTop)
        outer.addWidget(bubble)
        outer.addStretch(1)

        self._dots = 0
        self._timer = QTimer(self)
        self._timer.setInterval(420)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _tick(self) -> None:
        self._dots = (self._dots + 1) % 4
        self.label.setText("正在思考" + "·" * self._dots)

    def stop(self) -> None:
        self._timer.stop()


class ToolCallCard(QFrame):
    """一次工具调用的可视化卡片：名称、参数、耗时、结果。"""

    def __init__(self, name: str, args_text: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.name = name
        self._started = time.perf_counter()
        self._expanded = True
        self.setProperty("tool", True)
        self.setProperty("state", "running")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(8)
        self.dot = QLabel("●")
        self.dot.setStyleSheet("color: %s;" % T.WARNING)
        header.addWidget(self.dot)

        title = QLabel(name)
        title.setProperty("toolName", True)
        header.addWidget(title)

        self.status = QLabel("运行中")
        self.status.setProperty("role", "hint")
        header.addWidget(self.status)
        header.addStretch(1)

        self.elapsed = QLabel("")
        self.elapsed.setProperty("role", "mono")
        header.addWidget(self.elapsed)

        self.toggle = QToolButton()
        self.toggle.setText("收起")
        self.toggle.clicked.connect(self._toggle)
        header.addWidget(self.toggle)
        lay.addLayout(header)

        self.args = QLabel()
        self.args.setProperty("role", "mono")
        self.args.setWordWrap(True)
        self.args.setText("参数  " + _pretty_args(args_text))
        lay.addWidget(self.args)

        self.summary = QLabel("等待结果…")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.summary.setStyleSheet("color: %s;" % T.TEXT_MUTED)
        lay.addWidget(self.summary)

        self.raw = QPlainTextEdit()
        self.raw.setProperty("role", "log")
        self.raw.setReadOnly(True)
        self.raw.setMaximumHeight(220)
        self.raw.hide()
        lay.addWidget(self.raw)

        self.raw_toggle = QToolButton()
        self.raw_toggle.setText("▸ 原始结果")
        self.raw_toggle.setCheckable(True)
        self.raw_toggle.setCursor(Qt.PointingHandCursor)
        self.raw_toggle.toggled.connect(self.raw.setVisible)
        self.raw_toggle.toggled.connect(
            lambda checked: self.raw_toggle.setText(
                ("▾ " if checked else "▸ ") + "原始结果"
            )
        )
        self.raw_toggle.hide()
        lay.addWidget(self.raw_toggle, 0, Qt.AlignLeft)

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        for widget in (self.args, self.summary, self.raw_toggle):
            widget.setVisible(self._expanded)
        if not self._expanded:
            self.raw.hide()
            self.raw_toggle.setChecked(False)
        self.toggle.setText("收起" if self._expanded else "展开")

    def _set_state(self, state: str) -> None:
        self.setProperty("state", state)
        repolish(self)
        color = {"running": T.WARNING, "ok": T.SUCCESS, "error": T.DANGER}.get(state, T.TEXT_MUTED)
        self.dot.setStyleSheet("color: %s;" % color)

    def finish(self, result: str, elapsed: float | None = None) -> None:
        if elapsed is None:
            elapsed = time.perf_counter() - self._started
        failed = _looks_failed(result)
        self._set_state("error" if failed else "ok")
        self.status.setText("失败" if failed else "完成")
        self.elapsed.setText("%.1fs" % elapsed)
        summary, raw = _summarize_result(self.name, result)
        self.summary.setText(rich_text(summary))
        self.raw.setPlainText(raw or result)
        self.raw_toggle.show()

    def fail(self, message: str) -> None:
        self._set_state("error")
        self.status.setText("异常")
        self.elapsed.setText("%.1fs" % (time.perf_counter() - self._started))
        self.summary.setText(rich_text(message))
        self.raw.setPlainText(message)
        self.raw_toggle.show()


# ----------------------------------------------------------------------
# 结果整理
# ----------------------------------------------------------------------
def _pretty_args(args_text: str) -> str:
    text = (args_text or "").strip()
    if not text or text == "{}":
        return "（无）"
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return _short(text, 240)
    if not isinstance(value, dict) or not value:
        return _short(json.dumps(value, ensure_ascii=False), 240)
    return "  ".join(
        "%s=%s" % (k, json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v)
        for k, v in value.items()
    )


def _looks_failed(result: str) -> bool:
    try:
        data = json.loads(result)
    except (ValueError, TypeError):
        return False
    if not isinstance(data, dict):
        return False
    if data.get("success") is False:
        return True
    if data.get("error"):
        return True
    if data.get("state") in ("failed", "cancelled"):
        return True
    verify = data.get("verification") or data.get("verify")
    if isinstance(verify, dict) and verify.get("outcome") in ("failed", "error"):
        return True
    return False


def _summarize_result(name: str, result: str) -> tuple[str, str]:
    """返回（适合卡片显示的摘要, 原始文本）。"""
    raw = str(result or "")
    try:
        from tool_output import format_result

        formatted = format_result(name, raw)
        summary = formatted.split("原始结构化输出：")[0].strip()
        lines = summary.splitlines()
        if lines and lines[0].startswith("【工具输出解析】"):
            lines = lines[1:]
        summary = "\n".join(lines).strip() or raw
        return _short(summary, 1400), raw
    except Exception:
        return _short(raw, 900), raw
