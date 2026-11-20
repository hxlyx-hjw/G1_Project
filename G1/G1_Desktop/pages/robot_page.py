"""机器人页：实时状态、行走/转向、手臂姿态、动作日志。"""

from __future__ import annotations

import time

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..widgets import Card, Chip, MetricCard

WALK_SPEEDS = {
    "forward": (0.45, 0.0, 0.0),
    "back": (-0.32, 0.0, 0.0),
    "left": (0.0, 0.30, 0.0),
    "right": (0.0, -0.30, 0.0),
    "turn_left": (0.0, 0.0, 0.6),
    "turn_right": (0.0, 0.0, -0.6),
}


class RobotPage(QWidget):
    tool_requested = pyqtSignal(str, dict)
    viewer_toggle = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._ready = False
        self._presets_done = False

        root = QVBoxLayout(self)
        root.setContentsMargins(26, 22, 26, 20)
        root.setSpacing(14)

        # ---- 页头 ----
        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        title = QLabel("机器人")
        title.setProperty("role", "pageTitle")
        subtitle = QLabel("MuJoCo 实时状态与直接控制 · 与对话指令共用同一仿真会话")
        subtitle.setProperty("role", "subtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles)
        header.addStretch(1)

        self.state_chip = Chip("未连接", "idle")
        header.addWidget(self.state_chip)

        self.busy_chip = Chip("空闲", "ok")
        header.addWidget(self.busy_chip)

        self.viewer_button = QPushButton("打开 3D 窗口")
        self.viewer_button.setProperty("variant", "ghost")
        self.viewer_button.clicked.connect(self.viewer_toggle.emit)
        header.addWidget(self.viewer_button)

        reset_button = QPushButton("复位站立")
        reset_button.setProperty("variant", "ghost")
        reset_button.clicked.connect(lambda: self._call("robot_reset", {}))
        header.addWidget(reset_button)
        root.addLayout(header)

        # ---- 指标卡 ----
        metrics = QGridLayout()
        metrics.setSpacing(10)
        self.pos = MetricCard("位置 x / y", "—", "m")
        self.height = MetricCard("高度 z", "—", "m")
        self.yaw = MetricCard("朝向", "—", "°")
        self.speed = MetricCard("线速度", "—", "m/s")
        self.stability = MetricCard("综合可靠性", "—", "%")
        self.sim_time = MetricCard("仿真时间", "—", "s")
        for idx, card in enumerate(
            (self.pos, self.height, self.yaw, self.speed, self.stability, self.sim_time)
        ):
            metrics.addWidget(card, 0, idx)
        root.addLayout(metrics)

        # ---- 主体 ----
        body = QHBoxLayout()
        body.setSpacing(14)

        # 行走控制
        walk_card = Card("行走控制")
        walk_card.setMinimumWidth(380)

        settings = QHBoxLayout()
        settings.setSpacing(8)
        label = QLabel("单步时长")
        label.setProperty("role", "hint")
        settings.addWidget(label)
        self.duration = QDoubleSpinBox()
        self.duration.setRange(0.2, 5.0)
        self.duration.setSingleStep(0.2)
        self.duration.setDecimals(1)
        self.duration.setValue(1.0)
        self.duration.setFixedWidth(78)
        settings.addWidget(self.duration)

        label = QLabel("速度")
        label.setProperty("role", "hint")
        settings.addWidget(label)
        self.speed_scale = QSlider(Qt.Horizontal)
        self.speed_scale.setRange(25, 150)
        self.speed_scale.setValue(100)
        self.speed_scale.setFixedWidth(90)
        self.speed_scale.valueChanged.connect(self._update_speed_label)
        settings.addWidget(self.speed_scale)
        self.speed_label = QLabel("100%")
        self.speed_label.setProperty("role", "mono")
        self.speed_label.setFixedWidth(54)
        settings.addWidget(self.speed_label)
        settings.addStretch(1)
        walk_card.body.addLayout(settings)

        grid = QGridLayout()
        grid.setSpacing(8)
        buttons = [
            ("", 0, 1, None),
            ("前进  ▲", 0, 0, "forward"),
            ("", 0, 2, None),
            ("左移  ◀", 1, 0, "left"),
            ("停止  ■", 1, 1, "stop"),
            ("右移  ▶", 1, 2, "right"),
            ("左转  ↺", 2, 0, "turn_left"),
            ("后退  ▼", 2, 1, "back"),
            ("右转  ↻", 2, 2, "turn_right"),
        ]
        for text, row, col, action in buttons:
            if action is None:
                continue
            button = QPushButton(text)
            button.setProperty("action", "stop" if action == "stop" else "walk")
            button.setMinimumHeight(52)
            button.setCursor(Qt.PointingHandCursor)
            if action == "stop":
                button.clicked.connect(lambda: self.tool_requested.emit("robot_stop", {}))
            else:
                button.clicked.connect(lambda _=False, a=action: self._walk(a))
            grid.addWidget(button, row, col)
        walk_card.body.addLayout(grid)

        hint = QLabel("按钮通过 robot_walk / robot_turn 工具执行；速度上限由后端安全限幅。")
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        walk_card.body.addWidget(hint)
        body.addWidget(walk_card, 3)

        right = QVBoxLayout()
        right.setSpacing(14)

        arm_card = Card("手臂姿态")
        self.preset_layout = QGridLayout()
        self.preset_layout.setSpacing(8)
        arm_card.body.addLayout(self.preset_layout)
        self.preset_hint = QLabel("连接后载入可用预设。")
        self.preset_hint.setProperty("role", "hint")
        arm_card.body.addWidget(self.preset_hint)
        right.addWidget(arm_card)

        log_card = Card("动作日志")
        self.log = QPlainTextEdit()
        self.log.setProperty("role", "log")
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.log.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.log.setPlaceholderText("机器人的动作与工具结果会记录在这里。")
        log_card.body.addWidget(self.log, 1)
        right.addWidget(log_card, 1)
        body.addLayout(right, 4)

        root.addLayout(body, 1)

    # ------------------------------------------------------------------
    # 动作
    # ------------------------------------------------------------------
    def _update_speed_label(self, value: int) -> None:
        self.speed_label.setText("%d%%" % value)

    def _walk(self, action: str) -> None:
        vx, vy, vyaw = WALK_SPEEDS[action]
        scale = self.speed_scale.value() / 100.0
        args = {
            "vx": round(vx * scale, 3),
            "vy": round(vy * scale, 3),
            "vyaw": round(vyaw * scale, 3),
            "seconds": float(self.duration.value()),
        }
        self.tool_requested.emit("robot_walk", args)

    def _call(self, name: str, args: dict) -> None:
        self.tool_requested.emit(name, args)

    # ------------------------------------------------------------------
    # 后端事件
    # ------------------------------------------------------------------
    def on_ready(self, info: dict) -> None:
        self._ready = bool(info.get("robot"))
        self.set_viewer_running(bool(info.get("viewer")))
        self.state_chip.set_state("ok", "仿真就绪" if self._ready else "未启用机器人")
        if not self._ready:
            self.preset_hint.setText("本次启动未启用机器人（--no-robot）。")
            return
        presets = info.get("arm_presets") or []
        if presets and not self._presets_done:
            self._presets_done = True
            for idx, preset in enumerate(presets):
                button = QPushButton(preset)
                button.setCursor(Qt.PointingHandCursor)
                button.clicked.connect(
                    lambda _=False, p=preset: self._call("robot_set_arm", {"preset": p})
                )
                self.preset_layout.addWidget(button, idx // 2, idx % 2)
            self.preset_hint.setText("预设来自 g1_walkfree/config.py 的 ARM_PRESETS。")

    def set_viewer_running(self, running: bool) -> None:
        self.viewer_button.setText("关闭 3D 窗口" if running else "打开 3D 窗口")

    def on_status(self, status: dict | None) -> None:
        if not status:
            return
        self.pos.set_value(
            "%+.2f, %+.2f" % (status.get("x", 0.0), status.get("y", 0.0))
        )
        self.height.set_value("%.2f" % status.get("z", 0.0))
        self.yaw.set_value("%+.1f" % status.get("yaw", 0.0))
        self.speed.set_value("%.2f" % status.get("speed", 0.0))

        reliability = status.get("reliability") or {}
        quality = float(reliability.get("quality", 0.0)) * 100.0
        bar_state = "success" if quality >= 70 else ("warning" if quality >= 40 else "danger")
        self.stability.set_value("%.0f" % quality, percent=quality, state=bar_state)
        self.sim_time.set_value("%.1f" % status.get("time", 0.0))

        if status.get("fallen"):
            self.state_chip.set_state("danger", "已摔倒")
        elif float(status.get("speed", 0.0)) > 0.05:
            self.state_chip.set_state("info", "移动中")
        else:
            self.state_chip.set_state("ok", "站立")

    def set_busy(self, busy: bool, name: str = "") -> None:
        if busy:
            self.busy_chip.set_state("warn", "执行中" + (" · " + name if name else ""))
        else:
            self.busy_chip.set_state("ok", "空闲")

    def append_log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        for line in str(text).splitlines():
            self.log.appendPlainText("[%s] %s" % (stamp, line))
        bar = self.log.verticalScrollBar()
        bar.setValue(bar.maximum())
