"""对话页：聊天记录、工具调用卡片、快捷指令。"""

from __future__ import annotations

from collections import defaultdict, deque

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..widgets import FlowLayout, MessageBubble, ThinkingBubble, ToolCallCard, repolish

QUICK_PROMPTS = [
    "查询机器人当前状态",
    "往前走一米",
    "原地左转 90 度",
    "双手上举",
    "停下",
    "检查训练任务状态",
    "检查 SuperMimic 和 RAX 环境",
]


class ChatInput(QPlainTextEdit):
    send = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setPlaceholderText(
            "输入指令，Enter 发送，Shift+Enter 换行。例如：往前走一米，然后举起双手。"
        )
        self.setProperty("role", "chatInput")
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setMinimumHeight(58)
        self.setMaximumHeight(150)
        self.textChanged.connect(self._auto_height)
        self._auto_height()

    def keyPressEvent(self, event):  # noqa: N802 (Qt 命名)
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if not (event.modifiers() & Qt.ShiftModifier):
                self.send.emit()
                return
        super().keyPressEvent(event)

    def _auto_height(self) -> None:
        height = int(self.document().size().height()) + 24
        self.setFixedHeight(max(58, min(150, height)))


class ChatPage(QWidget):
    send_requested = pyqtSignal(str)
    reset_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._thinking: ThinkingBubble | None = None
        self._pending: dict[str, deque] = defaultdict(deque)
        self._busy = False

        root = QVBoxLayout(self)
        root.setContentsMargins(26, 22, 26, 18)
        root.setSpacing(14)

        # ---- 页头 ----
        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        title = QLabel("对话")
        title.setProperty("role", "pageTitle")
        subtitle = QLabel("DeepSeek Function Calling · 工具调用与结果实时可见")
        subtitle.setProperty("role", "subtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles)
        header.addStretch(1)

        self.state_chip = QLabel("未连接")
        self.state_chip.setProperty("chip", True)
        self.state_chip.setProperty("state", "idle")
        header.addWidget(self.state_chip)

        clear_button = QPushButton("清空对话")
        clear_button.setProperty("variant", "ghost")
        clear_button.clicked.connect(self.reset_requested.emit)
        header.addWidget(clear_button)
        root.addLayout(header)

        # ---- 聊天记录 ----
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        self.body_layout = QVBoxLayout(body)
        self.body_layout.setContentsMargins(2, 4, 12, 4)
        self.body_layout.setSpacing(12)
        self.body_layout.addStretch(1)
        self.scroll.setWidget(body)
        root.addWidget(self.scroll, 1)

        self._stick_bottom = True
        bar = self.scroll.verticalScrollBar()
        bar.rangeChanged.connect(self._on_range_changed)
        bar.valueChanged.connect(self._on_scroll_changed)

        # ---- 快捷指令 ----
        chips = FlowLayout(hspacing=8, vspacing=8)
        for prompt in QUICK_PROMPTS:
            chip = QPushButton(prompt)
            chip.setProperty("chip", True)
            chip.setCursor(Qt.PointingHandCursor)
            chip.clicked.connect(lambda _=False, text=prompt: self._send_quick(text))
            chips.addWidget(chip)
        root.addLayout(chips)

        # ---- 输入区 ----
        composer = QFrame()
        composer.setProperty("card", "flat")
        composer_layout = QHBoxLayout(composer)
        composer_layout.setContentsMargins(10, 10, 10, 10)
        composer_layout.setSpacing(10)

        self.input = ChatInput()
        self.input.send.connect(self._send)
        composer_layout.addWidget(self.input, 1)

        self.send_button = QPushButton("发送")
        self.send_button.setProperty("variant", "primary")
        self.send_button.setMinimumSize(96, 58)
        self.send_button.clicked.connect(self._send)
        composer_layout.addWidget(self.send_button)
        root.addWidget(composer)

        hint = QLabel("所有机器人动作都会在 MuJoCo 仿真中真实执行；危险动作可用右上角「急停」立即中断。")
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        self.add_system(
            "欢迎使用 G1 具身智能控制台。直接用自然语言下达指令，"
            "模型会调用工具操作机器人，并把每一步调用显示在这里。"
        )

    # ------------------------------------------------------------------
    # 输入
    # ------------------------------------------------------------------
    def _send(self) -> None:
        text = self.input.toPlainText().strip()
        if not text or not self._busy:
            return
        self.input.clear()
        self.send_requested.emit(text)

    def _send_quick(self, text: str) -> None:
        if not self._busy:
            return
        self.send_requested.emit(text)

    def set_ready(self, ready: bool) -> None:
        self._busy = ready
        self.input.setEnabled(ready)
        self.send_button.setEnabled(ready)
        self.state_chip.setText("已连接" if ready else "未连接")
        self.state_chip.setProperty("state", "ok" if ready else "idle")
        repolish(self.state_chip)
        if ready:
            self.input.setFocus()

    # ------------------------------------------------------------------
    # 消息
    # ------------------------------------------------------------------
    def _append(self, widget: QWidget) -> None:
        self.body_layout.insertWidget(self.body_layout.count() - 1, widget)
        QTimer.singleShot(0, self._scroll_bottom)

    def _on_range_changed(self, _minimum: int, maximum: int) -> None:
        if self._stick_bottom:
            self.scroll.verticalScrollBar().setValue(maximum)

    def _on_scroll_changed(self, value: int) -> None:
        bar = self.scroll.verticalScrollBar()
        self._stick_bottom = value >= bar.maximum() - 40

    def _scroll_bottom(self) -> None:
        self._stick_bottom = True
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def add_user(self, text: str) -> None:
        self._append(MessageBubble("user", text))

    def add_assistant(self, text: str, reasoning: str = "") -> None:
        self._end_thinking()
        self._append(MessageBubble("assistant", text, reasoning=reasoning))

    def add_system(self, text: str) -> None:
        self._append(MessageBubble("system", text))

    def add_proactive(self, text: str) -> None:
        self._end_thinking()
        body = text
        prefix = "[DeepSeek 主动消息]\n"
        if body.startswith(prefix):
            body = body[len(prefix):]
        self._append(MessageBubble("proactive", body))

    def add_error(self, text: str) -> None:
        self._end_thinking()
        self._append(MessageBubble("system", "出错了：%s" % text))

    def add_notice(self, text: str, level: str = "info") -> None:
        self._append(MessageBubble("system", text))

    # ------------------------------------------------------------------
    # 思考指示
    # ------------------------------------------------------------------
    def begin_thinking(self) -> None:
        if self._thinking is None:
            self._thinking = ThinkingBubble()
            self._append(self._thinking)

    def _end_thinking(self) -> None:
        if self._thinking is None:
            return
        widget = self._thinking
        self._thinking = None
        widget.stop()
        self.body_layout.removeWidget(widget)
        widget.hide()
        widget.deleteLater()

    # ------------------------------------------------------------------
    # 工具调用
    # ------------------------------------------------------------------
    def tool_started(self, name: str, args_text: str) -> None:
        card = ToolCallCard(name, args_text)
        self._pending[name].append(card)
        self._append(card)

    def tool_finished(self, name: str, result: str, elapsed: float) -> None:
        queue = self._pending.get(name)
        if queue:
            queue.popleft().finish(result, elapsed)

    # ------------------------------------------------------------------
    def clear(self) -> None:
        self._end_thinking()
        while self.body_layout.count() > 1:
            item = self.body_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._pending.clear()
        self.add_system("对话已清空，机器人状态保持不变。")
