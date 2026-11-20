"""主窗口：把对话、机器人、训练、环境四个页面组装成一个控制台。"""

from __future__ import annotations

import threading
from typing import Any

from PyQt5.QtCore import QMetaObject, Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from . import theme as T
from .backend import BackendService
from .pages.chat_page import ChatPage
from .pages.env_page import EnvPage
from .pages.robot_page import RobotPage
from .pages.training_page import TrainingPage
from .widgets import Chip

ACTION_TOOLS = {
    "robot_walk",
    "robot_goto",
    "robot_turn",
    "robot_sequence",
    "robot_set_arm",
    "robot_set_waist",
    "robot_stop",
    "robot_reset",
    "robot_play_motion",
}

TRAINING_TOOLS = {
    "robot_learn_action",
    "robot_learn_video",
    "robot_amass_train",
    "robot_multiview_train",
    "robot_train_video",
    "robot_docker_train",
    "robot_training_stop",
    "robot_training_status",
    "robot_proactive_report",
}


def _app_icon() -> QIcon:
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor(T.ACCENT_SOFT))
    painter.setPen(QColor(T.ACCENT))
    painter.drawRoundedRect(2, 2, 60, 60, 16, 16)
    painter.setPen(QColor(T.ACCENT))
    font = painter.font()
    font.setBold(True)
    font.setPointSize(20)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignCenter, "G1")
    painter.end()
    return QIcon(pixmap)


class MainWindow(QMainWindow):
    def __init__(self, options: dict[str, Any]):
        super().__init__()
        self.options = options
        self.service = BackendService(options)
        self._closing = False

        self.setWindowTitle("G1 具身智能控制台")
        self.setWindowIcon(_app_icon())
        self.resize(1480, 940)
        self.setMinimumSize(1120, 720)

        self._build_ui()
        self._connect_signals()

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(600)
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start()

        self._training_timer = QTimer(self)
        self._training_timer.setInterval(1500)
        self._training_timer.timeout.connect(self._refresh_training)
        self._training_timer.start()

    # ------------------------------------------------------------------
    # 布局
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("Root")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_sidebar())

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self.banner = self._build_banner()
        content_layout.addWidget(self.banner)

        self.stack = QStackedWidget()
        self.chat_page = ChatPage()
        self.robot_page = RobotPage()
        self.training_page = TrainingPage()
        self.env_page = EnvPage()
        self.training_page.log_provider = self.service.job_log_tail
        for page in (
            self.chat_page,
            self._scrollable(self.robot_page),
            self.training_page,
            self._scrollable(self.env_page),
        ):
            self.stack.addWidget(page)
        self.stack.currentChanged.connect(self._sync_nav)
        content_layout.addWidget(self.stack, 1)
        content_layout.addWidget(self._build_statusbar())
        body.addWidget(content, 1)

        layout.addLayout(body, 1)
        self.setCentralWidget(root)

    @staticmethod
    def _scrollable(page: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(page)
        return scroll

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setProperty("role", "header")
        header.setFixedHeight(74)
        lay = QHBoxLayout(header)
        lay.setContentsMargins(20, 10, 18, 10)
        lay.setSpacing(12)

        mark = QLabel("G1")
        mark.setFixedSize(40, 40)
        mark.setAlignment(Qt.AlignCenter)
        mark.setStyleSheet(
            "background: %s; color: %s; border: 1px solid %s; border-radius: 12px;"
            "font-size: 15pt; font-weight: 800;" % (T.ACCENT_SOFT, T.ACCENT, T.ACCENT_DARK)
        )
        lay.addWidget(mark)

        titles = QVBoxLayout()
        titles.setSpacing(1)
        title = QLabel("G1 具身智能控制台")
        title.setProperty("role", "title")
        subtitle = QLabel("DeepSeek · MuJoCo 仿真 · SuperMimic / RAX 研究工具")
        subtitle.setProperty("role", "subtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        lay.addLayout(titles)
        lay.addStretch(1)

        self.backend_chip = Chip("后端 初始化", "warn")
        self.sim_chip = Chip("仿真 —", "idle")
        self.viewer_chip = Chip("3D —", "idle")
        self.proactive_chip = Chip("报告 —", "idle")
        for chip in (self.backend_chip, self.sim_chip, self.viewer_chip, self.proactive_chip):
            lay.addWidget(chip)

        self.estop_button = QPushButton("急停")
        self.estop_button.setProperty("variant", "danger")
        self.estop_button.setMinimumWidth(84)
        self.estop_button.setToolTip("立即清零速度指令，不等待当前动作结束（Ctrl+Shift+S）")
        self.estop_button.setShortcut("Ctrl+Shift+S")
        self.estop_button.clicked.connect(self._emergency_stop)
        lay.addWidget(self.estop_button)
        return header

    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setProperty("role", "sidebar")
        sidebar.setFixedWidth(190)
        lay = QVBoxLayout(sidebar)
        lay.setContentsMargins(12, 16, 12, 14)
        lay.setSpacing(6)

        nav_label = QLabel("工作区")
        nav_label.setProperty("role", "section")
        lay.addWidget(nav_label)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        entries = [
            ("对话", "与模型对话并观察工具调用"),
            ("机器人", "实时状态与直接动作控制"),
            ("训练", "后台训练任务与日志"),
            ("环境", "研究环境与工具目录"),
        ]
        for index, (text, tip) in enumerate(entries):
            button = QPushButton(text)
            button.setProperty("nav", True)
            button.setCheckable(True)
            button.setToolTip(tip)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda _=False, i=index: self.stack.setCurrentIndex(i))
            self.nav_group.addButton(button, index)
            lay.addWidget(button)
        self.nav_group.button(0).setChecked(True)

        lay.addStretch(1)

        tips = QLabel(
            "提示\n"
            "Enter 发送 / Shift+Enter 换行\n"
            "急停 Ctrl+Shift+S\n"
            "3D 窗口独立"
        )
        tips.setProperty("role", "hint")
        tips.setStyleSheet("font-size: 8pt;")
        tips.setWordWrap(True)
        tips.setMinimumWidth(158)
        lay.addWidget(tips)
        return sidebar

    def _build_banner(self) -> QFrame:
        banner = QFrame()
        banner.setStyleSheet(
            "QFrame { background: %s; border-bottom: 1px solid #5c2a31; }"
            % T.DANGER_SOFT
        )
        lay = QHBoxLayout(banner)
        lay.setContentsMargins(20, 8, 20, 8)
        lay.setSpacing(12)
        self.banner_label = QLabel("")
        self.banner_label.setStyleSheet("color: %s;" % T.DANGER)
        self.banner_label.setWordWrap(True)
        lay.addWidget(self.banner_label, 1)
        retry = QPushButton("重试连接")
        retry.setProperty("variant", "ghost")
        retry.clicked.connect(self._retry)
        lay.addWidget(retry)
        banner.hide()
        return banner

    def _sync_nav(self, index: int) -> None:
        button = self.nav_group.button(index)
        if button is not None:
            button.setChecked(True)

    def _build_statusbar(self) -> QWidget:
        bar = QFrame()
        bar.setProperty("role", "statusbar")
        bar.setFixedHeight(34)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(20, 0, 20, 0)
        lay.setSpacing(10)
        self.status_label = QLabel("就绪")
        self.status_label.setProperty("role", "hint")
        lay.addWidget(self.status_label, 1)
        self.sim_label = QLabel("t = 0.0 s")
        self.sim_label.setProperty("role", "mono")
        lay.addWidget(self.sim_label)
        self.tool_label = QLabel("工具 —")
        self.tool_label.setProperty("role", "mono")
        lay.addWidget(self.tool_label)
        version = QLabel("G1 Desktop 1.0")
        version.setProperty("role", "hint")
        lay.addWidget(version)
        return bar

    # ------------------------------------------------------------------
    # 信号
    # ------------------------------------------------------------------
    def _connect_signals(self) -> None:
        s = self.service.signals
        s.ready.connect(self._on_ready)
        s.failed.connect(self._on_failed)
        s.log.connect(self._on_log)
        s.notice.connect(self._on_notice)
        s.chat_started.connect(self._on_chat_started)
        s.chat_finished.connect(self._on_chat_finished)
        s.chat_error.connect(self._on_chat_error)
        s.tool_started.connect(self._on_tool_started)
        s.tool_finished.connect(self._on_tool_finished)
        s.proactive.connect(self._on_proactive)
        s.env_result.connect(self.env_page.on_env_result)
        s.training_changed.connect(self._refresh_training)

        self.chat_page.send_requested.connect(self.service.ask)
        self.chat_page.reset_requested.connect(self._reset_chat)
        self.robot_page.tool_requested.connect(self.service.run_robot_tool)
        self.robot_page.viewer_toggle.connect(self._toggle_viewer)
        self.training_page.stop_requested.connect(
            lambda job: self.service.training_action("stop", job)
        )
        self.training_page.viewer_requested.connect(
            lambda job: self.service.training_action("viewer", job)
        )
        self.training_page.report_requested.connect(
            lambda job: self.service.training_action("report", job)
        )
        self.training_page.gif_requested.connect(
            lambda job: self.service.training_action("gif", job)
        )
        self.training_page.proactive_changed.connect(self.service.configure_proactive)
        self.env_page.check_requested.connect(self.service.check_environment)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start_backend(self) -> None:
        self.service.start()

    def _retry(self) -> None:
        self.banner.hide()
        self.backend_chip.set_state("warn", "后端 初始化")
        self.chat_page.set_ready(False)
        self.service.retry()

    @pyqtSlot(dict)
    def _on_ready(self, info: dict) -> None:
        self.banner.hide()
        self.backend_chip.set_state("ok", "后端 就绪")
        self.chat_page.set_ready(True)
        self.robot_page.on_ready(info)
        self.training_page.on_ready(info.get("proactive"))
        self.env_page.on_ready(info)
        self.tool_label.setText("工具 %d" % len(info.get("tools") or []))

        warning = self.options.get("startup_warning", "")
        if warning:
            self.chat_page.add_system("注意：" + warning)

        robot = "机器人已启用" if info.get("robot") else "未启用机器人"
        self.chat_page.add_system(
            "已连接 %s（%s），%s。共 %d 个工具可用。"
            % (
                info.get("model", "—"),
                info.get("base_url", "—"),
                robot,
                len(info.get("tools") or []),
            )
        )
        if info.get("initial_status"):
            self.robot_page.append_log(info["initial_status"])
        self._set_status("后端初始化完成")

    @pyqtSlot(str)
    def _on_failed(self, message: str) -> None:
        self.backend_chip.set_state("danger", "后端 失败")
        self.chat_page.set_ready(False)
        self.banner_label.setText("后端初始化失败：%s" % message)
        self.banner.show()
        self.chat_page.add_error(message)
        self._set_status("后端初始化失败")

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    # ------------------------------------------------------------------
    # 聊天事件
    # ------------------------------------------------------------------
    @pyqtSlot(str)
    def _on_chat_started(self, text: str) -> None:
        self.chat_page.add_user(text)
        self.chat_page.begin_thinking()
        self._set_status("模型处理中…")

    @pyqtSlot(str, str)
    def _on_chat_finished(self, answer: str, reasoning: str) -> None:
        self.chat_page.add_assistant(answer, reasoning)
        self._set_status("回复完成")

    @pyqtSlot(str)
    def _on_chat_error(self, message: str) -> None:
        self.chat_page.add_error(message)
        self._set_status("对话失败")

    @pyqtSlot(str)
    def _on_proactive(self, text: str) -> None:
        self.chat_page.add_proactive(text)
        self._set_status("收到主动训练报告")

    @pyqtSlot(str)
    def _on_log(self, message: str) -> None:
        line = message.strip().splitlines()
        if line:
            self._set_status(line[-1][:180])

    @pyqtSlot(str, str)
    def _on_notice(self, level: str, text: str) -> None:
        self._set_status(text[:180])
        if level in ("warn", "danger"):
            self.chat_page.add_notice(text, level)

    # ------------------------------------------------------------------
    # 工具事件
    # ------------------------------------------------------------------
    @pyqtSlot(str, str)
    def _on_tool_started(self, name: str, args_text: str) -> None:
        self.chat_page.tool_started(name, args_text)
        if name in ACTION_TOOLS:
            self.robot_page.set_busy(True, name)
        self._set_status("调用工具 %s" % name)

    @pyqtSlot(str, str, float)
    def _on_tool_finished(self, name: str, result: str, elapsed: float) -> None:
        self.chat_page.tool_finished(name, result, elapsed)
        if name in ACTION_TOOLS:
            self.robot_page.set_busy(False)
            summary = result.strip().splitlines()[0] if result.strip() else ""
            self.robot_page.append_log(
                "%s → %s（%.2fs）" % (name, summary[:240], elapsed)
            )
        if name in TRAINING_TOOLS:
            QTimer.singleShot(800, self._refresh_training)
        self._set_status("%s 完成（%.2fs）" % (name, elapsed))

    # ------------------------------------------------------------------
    # 轮询
    # ------------------------------------------------------------------
    def _refresh_status(self) -> None:
        if not self.service.is_ready:
            return
        status = self.service.robot_status()
        if status is not None:
            self.robot_page.on_status(status)
            self.sim_label.setText("t = %.1f s" % status.get("time", 0.0))
            if status.get("fallen"):
                self.sim_chip.set_state("danger", "仿真 摔倒")
            else:
                self.sim_chip.set_state("ok", "仿真 运行")
        else:
            self.sim_chip.set_state("idle", "仿真 关闭")

        viewer = self.service.viewer_running
        self.viewer_chip.set_state("info" if viewer else "idle",
                                   "3D 开" if viewer else "3D 关")
        self.robot_page.set_viewer_running(viewer)

        proactive = self.service.proactive_state()
        if proactive:
            enabled = bool(proactive.get("enabled"))
            self.proactive_chip.set_state(
                "info" if enabled else "idle",
                "报告 %s" % ("开" if enabled else "关"),
            )
        else:
            self.proactive_chip.set_state("idle", "报告 —")

    def _refresh_training(self) -> None:
        if not self.service.is_ready:
            return
        jobs = self.service.training_jobs()
        self.training_page.on_jobs(jobs)

    # ------------------------------------------------------------------
    # 动作
    # ------------------------------------------------------------------
    def _emergency_stop(self) -> None:
        self.service.emergency_stop()
        self.robot_page.set_busy(False)
        self.robot_page.append_log("急停：速度指令已清零")

    def _toggle_viewer(self) -> None:
        if self.service.viewer_running:
            self.service.close_viewer()
        else:
            self.service.open_viewer()
        self._set_status("正在切换 3D 窗口…")

    def _reset_chat(self) -> None:
        self.service.reset_chat()
        self.chat_page.clear()
        self._set_status("对话上下文已清空")

    # ------------------------------------------------------------------
    # 退出
    # ------------------------------------------------------------------
    def closeEvent(self, event):  # noqa: N802 (Qt 命名)
        if self._closing:
            event.accept()
            return
        self._closing = True
        event.ignore()
        self._status_timer.stop()
        self._training_timer.stop()
        self.hide()
        self._set_status("正在关闭仿真与后台任务…")
        threading.Thread(target=self._shutdown_and_quit, daemon=True).start()

    def _shutdown_and_quit(self) -> None:
        try:
            self.service.shutdown()
        finally:
            QMetaObject.invokeMethod(self, "quit_app", Qt.QueuedConnection)

    @pyqtSlot()
    def quit_app(self) -> None:
        QApplication.instance().quit()
