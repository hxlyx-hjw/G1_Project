"""训练页：后台任务列表、进度、日志尾部与主动报告设置。"""

from __future__ import annotations

from typing import Callable

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..widgets import Card, Chip, elide_path

STATE_TEXT = {
    "queued": "已排队",
    "waiting": "等待参考",
    "preparing": "准备数据",
    "extracting": "提取骨架",
    "retargeting": "全身重定向",
    "searching": "搜索视频",
    "downloading": "下载视频",
    "screening": "动作筛选",
    "training": "训练中",
    "evaluating": "独立评估",
    "rendering": "渲染中",
    "completed": "已完成",
    "failed": "失败",
    "cancelled": "已停止",
    "handed_off": "已交接",
    "ready_for_handoff": "等待交接",
    "coordinator_preparing": "准备协调网络",
    "closed": "窗口已关闭",
}

def state_text(state: str) -> str:
    return STATE_TEXT.get(state, state or "未知")


def _job_prefix(job_id: str) -> str:
    """列表里只显示 月-日 时分秒，完整 ID 放在提示与详情里。"""
    parts = (job_id or "").split("_")
    if len(parts) >= 2 and len(parts[0]) == 8:
        return "%s-%s %s" % (parts[0][4:6], parts[0][6:8], parts[1])
    return job_id


def _chip_state(state: str) -> str:
    if state == "completed":
        return "ok"
    if state == "failed":
        return "danger"
    if state in ("cancelled",):
        return "idle"
    if state in ("queued", "waiting", "ready_for_handoff"):
        return "warn"
    return "info"


class TrainingPage(QWidget):
    stop_requested = pyqtSignal(str)
    viewer_requested = pyqtSignal(str)
    report_requested = pyqtSignal(str)
    gif_requested = pyqtSignal(str)
    proactive_changed = pyqtSignal(bool, float)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._ids: list[str] = []
        self._rows: dict[str, tuple[QLabel, Chip]] = {}
        self._current: str = ""
        self.log_provider: Callable[[str], str] | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(26, 22, 26, 20)
        root.setSpacing(14)

        # ---- 页头 ----
        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        title = QLabel("训练")
        title.setProperty("role", "pageTitle")
        subtitle = QLabel("后台模仿学习任务 · 状态、进度、日志与主动报告")
        subtitle.setProperty("role", "subtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles)
        header.addStretch(1)
        self.count_chip = Chip("0 个任务", "idle")
        header.addWidget(self.count_chip)
        root.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(14)

        # ---- 任务列表 ----
        left = Card("任务列表")
        left.setFixedWidth(340)
        self.jobs = QListWidget()
        self.jobs.setWordWrap(False)
        self.jobs.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.jobs.currentItemChanged.connect(self._on_selected)
        left.body.addWidget(self.jobs, 1)
        body.addWidget(left)

        # ---- 详情 ----
        right = QVBoxLayout()
        right.setSpacing(14)

        detail = Card("任务详情")
        detail_header = QHBoxLayout()
        self.state_chip = Chip("未选择任务", "idle")
        detail_header.addWidget(self.state_chip)
        self.job_id_label = QLabel("—")
        self.job_id_label.setProperty("role", "mono")
        self.job_id_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        detail_header.addWidget(self.job_id_label, 1)
        detail.body.addLayout(detail_header)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.stop_button = QPushButton("停止任务")
        self.stop_button.setProperty("variant", "danger")
        self.stop_button.clicked.connect(self._emit_stop)
        self.viewer_button = QPushButton("渲染窗口")
        self.viewer_button.setProperty("variant", "ghost")
        self.viewer_button.clicked.connect(self._emit_viewer)
        self.report_button = QPushButton("立即报告")
        self.report_button.setProperty("variant", "ghost")
        self.report_button.clicked.connect(self._emit_report)
        self.gif_button = QPushButton("导出 GIF")
        self.gif_button.setProperty("variant", "ghost")
        self.gif_button.clicked.connect(self._emit_gif)
        for button in (self.report_button, self.viewer_button, self.gif_button, self.stop_button):
            actions.addWidget(button)
        detail.body.addLayout(actions)

        grid = QGridLayout()
        grid.setHorizontalSpacing(26)
        grid.setVerticalSpacing(10)
        self.fields: dict[str, QLabel] = {}
        for idx, (key, label) in enumerate(
            (
                ("state", "阶段"),
                ("elapsed_text", "已用时间"),
                ("training_text", "训练用时"),
                ("eta_text", "预计剩余"),
                ("device", "设备"),
                ("message", "说明"),
            )
        ):
            name = QLabel(label)
            name.setProperty("role", "metricLabel")
            value = QLabel("—")
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            grid.addWidget(name, idx // 3, (idx % 3) * 2)
            grid.addWidget(value, idx // 3, (idx % 3) * 2 + 1)
            self.fields[key] = value
        detail.body.addLayout(grid)

        progress_row = QHBoxLayout()
        progress_row.setSpacing(10)
        self.iteration_label = QLabel("轮次 —")
        self.iteration_label.setProperty("role", "hint")
        progress_row.addWidget(self.iteration_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        progress_row.addWidget(self.progress, 1)
        detail.body.addLayout(progress_row)

        self.output_label = QLabel("输出目录：—")
        self.output_label.setProperty("role", "mono")
        self.output_label.setWordWrap(False)
        self.output_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        detail.body.addWidget(self.output_label)
        right.addWidget(detail)

        log_card = Card("训练日志")
        self.log = QPlainTextEdit()
        self.log.setProperty("role", "log")
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setPlaceholderText("选择任务后显示 worker.log 的尾部内容。")
        log_card.body.addWidget(self.log, 1)
        right.addWidget(log_card, 1)

        body.addLayout(right, 1)
        root.addLayout(body, 1)

        # ---- 主动报告 ----
        proactive = Card("主动训练报告")
        row = QHBoxLayout()
        row.setSpacing(12)
        self.proactive_enabled = QCheckBox("开启主动报告")
        self.proactive_enabled.setChecked(True)
        row.addWidget(self.proactive_enabled)
        row.addWidget(QLabel("间隔"))
        self.interval = QSpinBox()
        self.interval.setRange(1, 1440)
        self.interval.setValue(10)
        self.interval.setSuffix(" 分钟")
        self.interval.setFixedWidth(132)
        row.addWidget(self.interval)
        apply_button = QPushButton("应用设置")
        apply_button.setProperty("variant", "ghost")
        apply_button.clicked.connect(self._emit_proactive)
        row.addWidget(apply_button)
        self.proactive_hint = QLabel("训练启动后，后台会按间隔读取真实状态并生成报告。")
        self.proactive_hint.setProperty("role", "hint")
        row.addWidget(self.proactive_hint, 1)
        proactive.body.addLayout(row)
        root.addWidget(proactive)

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(2000)
        self._refresh_timer.timeout.connect(self._refresh_log)
        self._refresh_timer.start()

        self._set_buttons(False)

    # ------------------------------------------------------------------
    # 任务列表
    # ------------------------------------------------------------------
    def on_jobs(self, jobs: list[dict]) -> None:
        ids = [str(job.get("job_id")) for job in jobs]
        self.count_chip.set_state("info" if ids else "idle", "%d 个任务" % len(ids))

        if ids != self._ids:
            selected = self._current
            self._ids = ids
            self.jobs.blockSignals(True)
            self.jobs.clear()
            self._rows.clear()
            for job in jobs:
                job_id = str(job.get("job_id"))
                state = str(job.get("state", ""))

                item = QListWidgetItem()
                item.setData(Qt.UserRole, job_id)
                item.setToolTip(job_id)

                holder = QWidget()
                row = QHBoxLayout(holder)
                row.setContentsMargins(8, 5, 8, 5)
                row.setSpacing(8)
                name = QLabel(_job_prefix(job_id))
                name.setStyleSheet("font-size: 9pt;")
                chip = Chip(state_text(state), _chip_state(state))
                row.addWidget(name, 1)
                row.addWidget(chip)
                item.setSizeHint(holder.sizeHint())
                self.jobs.addItem(item)
                self.jobs.setItemWidget(item, holder)
                self._rows[job_id] = (name, chip)
                if job_id == selected:
                    self.jobs.setCurrentItem(item)
            if self.jobs.currentItem() is None and self.jobs.count():
                self.jobs.setCurrentRow(0)
            self.jobs.blockSignals(False)
            self._on_selected(self.jobs.currentItem(), None)
        else:
            for job in jobs:
                job_id = str(job.get("job_id"))
                state = str(job.get("state", ""))
                widgets = self._rows.get(job_id)
                if widgets is not None:
                    widgets[1].set_state(_chip_state(state), state_text(state))

        selected = self._selected_id()
        for job in jobs:
            if str(job.get("job_id")) == selected:
                self._render(job)
                break

    def _selected_id(self) -> str:
        item = self.jobs.currentItem()
        return str(item.data(Qt.UserRole)) if item else ""

    def _on_selected(self, current, _previous) -> None:
        job_id = str(current.data(Qt.UserRole)) if current else ""
        if job_id == self._current:
            return
        self._current = job_id
        self._set_buttons(bool(job_id))
        self.log.clear()
        self._refresh_log()

    def _set_buttons(self, enabled: bool) -> None:
        for button in (self.stop_button, self.viewer_button, self.report_button, self.gif_button):
            button.setEnabled(enabled)

    # ------------------------------------------------------------------
    def _render(self, job: dict) -> None:
        state = str(job.get("state", ""))
        color_state = {
            "completed": "ok",
            "failed": "danger",
            "cancelled": "idle",
            "training": "info",
            "queued": "warn",
        }.get(state, "info")
        self.state_chip.set_state(color_state, state_text(state))
        self.job_id_label.setText(str(job.get("job_id", "—")))

        for key in ("state", "elapsed_text", "training_text", "eta_text", "device"):
            label = self.fields.get(key)
            if label is not None:
                label.setText(str(job.get(key, "—")))
        self.fields["state"].setText(state_text(state))
        self.fields["message"].setText(str(job.get("message", "—")))

        iteration = job.get("iteration")
        total = job.get("total_iterations")
        if isinstance(iteration, (int, float)) and isinstance(total, (int, float)) and total > 0:
            percent = max(0, min(100, int(round(100.0 * iteration / total))))
            self.progress.setValue(percent)
            self.iteration_label.setText("轮次 %d / %d（%d%%）" % (iteration, total, percent))
        else:
            self.progress.setValue(0)
            self.iteration_label.setText(
                "轮次 %s" % ("—" if iteration is None else iteration)
            )

        output_dir = str(job.get("output_dir", "—"))
        self.output_label.setText("输出目录：  " + elide_path(output_dir, 48))
        self.output_label.setToolTip(output_dir)

        running = bool(job.get("worker_running"))
        self.stop_button.setEnabled(running and state not in ("completed", "failed", "cancelled"))

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------
    def _refresh_log(self) -> None:
        job_id = self._current
        if not job_id or self.log_provider is None:
            return
        text = self.log_provider(job_id)
        if text and text != self.log.toPlainText():
            bar = self.log.verticalScrollBar()
            at_bottom = bar.value() >= bar.maximum() - 4
            self.log.setPlainText(text)
            if at_bottom:
                bar.setValue(bar.maximum())

    # ------------------------------------------------------------------
    # 主动报告
    # ------------------------------------------------------------------
    def on_ready(self, proactive: dict | None) -> None:
        if not proactive:
            self.proactive_enabled.setChecked(False)
            self.proactive_enabled.setEnabled(False)
            self.proactive_hint.setText("当前会话未启用主动报告。")
            return
        self.proactive_enabled.setChecked(bool(proactive.get("enabled", True)))
        minutes = float(proactive.get("interval_minutes", 10.0))
        self.interval.setValue(max(1, int(round(minutes))))

    def _emit_proactive(self) -> None:
        self.proactive_changed.emit(
            self.proactive_enabled.isChecked(), float(self.interval.value())
        )

    def _emit_stop(self) -> None:
        if self._current:
            self.stop_requested.emit(self._current)

    def _emit_viewer(self) -> None:
        if self._current:
            self.viewer_requested.emit(self._current)

    def _emit_report(self) -> None:
        if self._current:
            self.report_requested.emit(self._current)

    def _emit_gif(self) -> None:
        if self._current:
            self.gif_requested.emit(self._current)
