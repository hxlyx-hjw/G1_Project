"""环境页：研究工具状态、运行配置与工具目录。"""

from __future__ import annotations

import html
import json

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..widgets import Card, Chip, elide_path

PATH_FIELDS = [
    ("g1_root", "G1 源码"),
    ("supermimic", "SuperMimic"),
    ("rax", "RAX"),
    ("wsl", "WSL 发行版"),
    ("venv", "SuperMimic 环境"),
    ("jobs", "任务记录"),
]


class EnvPage(QWidget):
    check_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._tools: list[dict] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(26, 22, 26, 20)
        root.setSpacing(14)

        # ---- 页头 ----
        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        title = QLabel("环境")
        title.setProperty("role", "pageTitle")
        subtitle = QLabel("研究工具连通性、运行配置与全部工具声明")
        subtitle.setProperty("role", "subtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles)
        header.addStretch(1)
        self.check_button = QPushButton("检查研究环境")
        self.check_button.setProperty("variant", "primary")
        self.check_button.setEnabled(False)
        self.check_button.clicked.connect(self._check)
        header.addWidget(self.check_button)
        root.addLayout(header)

        top = QHBoxLayout()
        top.setSpacing(14)

        # ---- 研究环境 ----
        env_card = Card("研究环境")
        chips = QHBoxLayout()
        chips.setSpacing(8)
        self.supermimic_chip = Chip("SuperMimic · 未检查", "idle")
        self.rax_chip = Chip("RAX · 未检查", "idle")
        chips.addWidget(self.supermimic_chip)
        chips.addWidget(self.rax_chip)
        chips.addStretch(1)
        env_card.body.addLayout(chips)

        self.env_output = QPlainTextEdit()
        self.env_output.setProperty("role", "log")
        self.env_output.setReadOnly(True)
        self.env_output.setPlaceholderText("点击「检查研究环境」读取 SuperMimic / RAX 状态。")
        self.env_output.setMinimumHeight(170)
        env_card.body.addWidget(self.env_output, 1)
        top.addWidget(env_card, 3)

        # ---- 运行配置 ----
        config_card = Card("运行配置")
        self.config_grid = QGridLayout()
        self.config_grid.setHorizontalSpacing(16)
        self.config_grid.setVerticalSpacing(5)
        self.config_values: dict[str, QLabel] = {}
        rows = [
            ("model", "模型"),
            ("base_url", "接口地址"),
            ("temperature", "温度"),
            ("reasoning_effort", "推理强度"),
        ] + PATH_FIELDS
        for idx, (key, label) in enumerate(rows):
            name = QLabel(label)
            name.setProperty("role", "metricLabel")
            value = QLabel("—")
            value.setProperty("role", "mono")
            # 路径由 wrap_path() 在反斜杠处手动分行，关闭 Qt 自动换行，
            # 否则它会在盘符冒号处错误断行。
            value.setWordWrap(False)
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.config_grid.addWidget(name, idx, 0)
            self.config_grid.addWidget(value, idx, 1)
            self.config_values[key] = value
        self.config_grid.setColumnStretch(1, 1)
        config_card.body.addLayout(self.config_grid)
        config_card.body.addStretch(1)
        top.addWidget(config_card, 2)
        root.addLayout(top)

        # ---- 工具目录 ----
        tools_card = Card("工具目录")
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        self.search = QLineEdit()
        self.search.setPlaceholderText("按名称或描述筛选…")
        self.search.textChanged.connect(self._filter)
        left_layout.addWidget(self.search)
        self.tool_list = QListWidget()
        self.tool_list.setWordWrap(True)
        self.tool_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tool_list.setMinimumHeight(150)
        self.tool_list.currentItemChanged.connect(self._show_tool)
        left_layout.addWidget(self.tool_list, 1)
        self.tool_count = QLabel("—")
        self.tool_count.setProperty("role", "hint")
        left_layout.addWidget(self.tool_count)
        splitter.addWidget(left)

        self.tool_detail = QTextBrowser()
        self.tool_detail.setOpenExternalLinks(False)
        self.tool_detail.setMinimumHeight(150)
        self.tool_detail.setPlaceholderText("选择一个工具查看说明和参数。")
        splitter.addWidget(self.tool_detail)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        tools_card.body.addWidget(splitter, 1)
        root.addWidget(tools_card, 1)

    # ------------------------------------------------------------------
    def _check(self) -> None:
        self.check_button.setEnabled(False)
        self.check_button.setText("检查中…")
        self.supermimic_chip.set_state("warn", "SuperMimic · 检查中")
        self.rax_chip.set_state("warn", "RAX · 检查中")
        self.check_requested.emit()

    def on_env_result(self, target: str, text: str) -> None:
        label = "SuperMimic" if target == "supermimic" else "RAX"
        chip = self.supermimic_chip if target == "supermimic" else self.rax_chip
        try:
            data = json.loads(text)
            ok = isinstance(data, dict) and data.get("success") is not False and not data.get(
                "error"
            )
        except (ValueError, TypeError):
            ok = False
        chip.set_state("ok" if ok else "danger", "%s · %s" % (label, "可用" if ok else "异常"))

        name = (
            "robot_supermimic_status" if target == "supermimic" else "robot_rax_status"
        )
        body = text
        try:
            from tool_output import format_result

            body = format_result(name, text)
        except Exception:
            try:
                body = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
            except (ValueError, TypeError):
                body = str(text)
        self.env_output.appendPlainText("=" * 18 + " " + label + " " + "=" * 18)
        self.env_output.appendPlainText(body + "\n")

        if self.supermimic_chip.property("state") != "warn" and self.rax_chip.property(
            "state"
        ) != "warn":
            self.check_button.setEnabled(True)
            self.check_button.setText("重新检查")

    def on_ready(self, info: dict) -> None:
        values = {
            "model": str(info.get("model", "—")),
            "base_url": str(info.get("base_url", "—")),
            "temperature": str(info.get("temperature", "—")),
            "reasoning_effort": str(info.get("reasoning_effort", "—")),
        }
        for key, value in (info.get("paths") or {}).items():
            values[key] = str(value)
        for key, label in self.config_values.items():
            value = str(values.get(key, "—"))
            if key in ("g1_root", "supermimic", "rax", "venv", "jobs"):
                label.setText(elide_path(value, 34))
                label.setToolTip(value)
            else:
                label.setText(value)
                label.setToolTip("")

        self._tools = list(info.get("tools") or [])
        self._populate_tools(self._tools)
        self.check_button.setEnabled(True)
        self.check_button.setText("检查研究环境")

    # ------------------------------------------------------------------
    def _populate_tools(self, tools: list[dict]) -> None:
        self.tool_list.blockSignals(True)
        self.tool_list.clear()
        for tool in tools:
            name = tool.get("name", "")
            source = tool.get("source", "")
            item = QListWidgetItem("%s · %s" % (name, source))
            item.setData(Qt.UserRole, tool)
            item.setToolTip(tool.get("description", ""))
            self.tool_list.addItem(item)
        self.tool_list.blockSignals(False)
        self.tool_count.setText("共 %d 个工具" % len(tools))
        if self.tool_list.count():
            self.tool_list.setCurrentRow(0)

    def _filter(self, text: str) -> None:
        keyword = text.strip().lower()
        if not keyword:
            self._populate_tools(self._tools)
            return
        filtered = [
            tool
            for tool in self._tools
            if keyword in tool.get("name", "").lower()
            or keyword in tool.get("description", "").lower()
        ]
        self._populate_tools(filtered)

    def _show_tool(self, current, _previous) -> None:
        if current is None:
            self.tool_detail.clear()
            return
        tool = current.data(Qt.UserRole) or {}
        schema = {
            "name": tool.get("name", ""),
            "source": tool.get("source", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {}),
        }
        description = html.escape(str(tool.get("description", ""))).replace("\n", "<br>")
        params = html.escape(
            json.dumps(schema["parameters"], ensure_ascii=False, indent=2)
        )
        self.tool_detail.setHtml(
            "<div style='font-size:14px;font-weight:700;color:%s;'>%s</div>"
            "<div style='color:%s;font-size:9pt;margin:2px 0 10px 0;'>来源：%s</div>"
            "<div style='color:%s;line-height:150%%;'>%s</div>"
            "<div style='color:%s;font-size:9pt;margin:14px 0 6px 0;'>参数 Schema</div>"
            "<pre style='font-family:%s;font-size:9pt;color:%s;white-space:pre-wrap;'>%s</pre>"
            % (
                "#e8eff7",
                schema["name"],
                "#92a3b5",
                schema["source"],
                "#c7d3e0",
                description,
                "#92a3b5",
                "Consolas",
                "#a9bacb",
                params,
            )
        )
