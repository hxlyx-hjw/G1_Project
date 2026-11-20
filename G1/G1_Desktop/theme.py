"""集中式视觉主题：配色、字体、圆角与全局 QSS。

所有界面颜色都从这里取，页面代码不写死色值。想换主题只改本文件。
"""

from __future__ import annotations

from PyQt5.QtGui import QColor, QFont, QPalette
from PyQt5.QtWidgets import QApplication

# ----------------------------------------------------------------------
# 配色（深色控制台）
# ----------------------------------------------------------------------
BG = "#0c1117"
SURFACE = "#121922"
SURFACE_ALT = "#18212c"
CARD = "#1a2430"
CARD_HOVER = "#213143"
BORDER = "#2a3847"
BORDER_SOFT = "#202b37"

TEXT = "#e8eff7"
TEXT_MUTED = "#92a3b5"
TEXT_DIM = "#66788a"

ACCENT = "#4cc2ff"
ACCENT_DARK = "#1884b8"
ACCENT_SOFT = "#123549"

SUCCESS = "#3ecf8e"
SUCCESS_SOFT = "#11362a"
WARNING = "#e8b55a"
WARNING_SOFT = "#3a2d16"
DANGER = "#f2707c"
DANGER_SOFT = "#3d1b20"

USER_BUBBLE = "#143a50"
ASSISTANT_BUBBLE = "#18222e"
TOOL_BUBBLE = "#141d27"

FONT_UI = '"Microsoft YaHei UI", "Segoe UI", "PingFang SC", sans-serif'
FONT_MONO = '"Cascadia Mono", "Consolas", "DejaVu Sans Mono", monospace'

RADIUS = 10


def apply(app: QApplication) -> None:
    """设置全局字体、调色板与样式表。"""
    app.setFont(QFont("Microsoft YaHei UI", 10))
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(BG))
    palette.setColor(QPalette.WindowText, QColor(TEXT))
    palette.setColor(QPalette.Base, QColor(SURFACE_ALT))
    palette.setColor(QPalette.AlternateBase, QColor(SURFACE))
    palette.setColor(QPalette.Text, QColor(TEXT))
    palette.setColor(QPalette.Button, QColor(SURFACE_ALT))
    palette.setColor(QPalette.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.Highlight, QColor(ACCENT_DARK))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ToolTipBase, QColor(CARD))
    palette.setColor(QPalette.ToolTipText, QColor(TEXT))
    palette.setColor(QPalette.PlaceholderText, QColor(TEXT_DIM))
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)


STYLESHEET = f"""
/* ---------- 基础 ---------- */
QWidget {{
    color: {TEXT};
    font-family: {FONT_UI};
    font-size: 10pt;
}}
QMainWindow, QWidget#Root {{
    background: {BG};
}}
QToolTip {{
    background: {CARD};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 8px;
}}

/* ---------- 卡片 ---------- */
QFrame[card="true"] {{
    background: {CARD};
    border: 1px solid {BORDER_SOFT};
    border-radius: 12px;
}}
QFrame[card="flat"] {{
    background: {SURFACE};
    border: 1px solid {BORDER_SOFT};
    border-radius: 10px;
}}
QLabel[role="title"] {{
    font-size: 16pt;
    font-weight: 700;
    color: {TEXT};
}}
QLabel[role="pageTitle"] {{
    font-size: 15pt;
    font-weight: 700;
    color: {TEXT};
}}
QLabel[role="subtitle"] {{
    color: {TEXT_MUTED};
    font-size: 9pt;
}}
QLabel[role="section"] {{
    color: {TEXT_MUTED};
    font-size: 9pt;
    font-weight: 600;
    letter-spacing: 1px;
}}
QLabel[role="metricValue"] {{
    font-size: 17pt;
    font-weight: 600;
    color: {TEXT};
}}
QLabel[role="metricLabel"] {{
    color: {TEXT_MUTED};
    font-size: 9pt;
}}
QLabel[role="mono"] {{
    font-family: {FONT_MONO};
    color: {TEXT_MUTED};
    font-size: 9pt;
}}
QLabel[role="hint"] {{
    color: {TEXT_DIM};
    font-size: 9pt;
}}

/* ---------- 按钮 ---------- */
QPushButton {{
    background: {SURFACE_ALT};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 7px 14px;
}}
QPushButton:hover {{
    background: {CARD_HOVER};
    border-color: {ACCENT_DARK};
}}
QPushButton:pressed {{
    background: {ACCENT_SOFT};
}}
QPushButton:disabled {{
    color: {TEXT_DIM};
    background: {SURFACE};
    border-color: {BORDER_SOFT};
}}
QPushButton[variant="primary"] {{
    background: {ACCENT};
    color: #04121b;
    border: none;
    font-weight: 700;
}}
QPushButton[variant="primary"]:hover {{
    background: #6fd0ff;
}}
QPushButton[variant="primary"]:pressed {{
    background: {ACCENT_DARK};
}}
QPushButton[variant="danger"] {{
    background: {DANGER_SOFT};
    color: {DANGER};
    border: 1px solid #5c2a31;
    font-weight: 600;
}}
QPushButton[variant="danger"]:hover {{
    background: #55242b;
    border-color: {DANGER};
}}
QPushButton[variant="ghost"] {{
    background: transparent;
    border: 1px solid {BORDER};
}}
QPushButton[variant="ghost"]:hover {{
    background: {SURFACE_ALT};
}}
QPushButton[nav="true"] {{
    text-align: left;
    padding: 10px 16px;
    border: none;
    border-radius: 8px;
    background: transparent;
    color: {TEXT_MUTED};
    font-weight: 500;
}}
QPushButton[nav="true"]:hover {{
    background: {SURFACE_ALT};
    color: {TEXT};
}}
QPushButton[nav="true"]:checked {{
    background: {ACCENT_SOFT};
    color: {ACCENT};
    font-weight: 700;
}}
QPushButton[action="walk"] {{
    background: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 12px 10px;
    font-weight: 600;
}}
QPushButton[action="walk"]:hover {{
    background: {CARD_HOVER};
    border-color: {ACCENT};
    color: {ACCENT};
}}
QPushButton[action="stop"] {{
    background: {DANGER_SOFT};
    border: 1px solid #5c2a31;
    color: {DANGER};
    border-radius: 10px;
    padding: 12px 10px;
    font-weight: 700;
}}
QPushButton[action="stop"]:hover {{
    background: #55242b;
    border-color: {DANGER};
}}
QPushButton[chip="true"] {{
    background: {SURFACE};
    border: 1px solid {BORDER_SOFT};
    border-radius: 14px;
    padding: 5px 12px;
    color: {TEXT_MUTED};
    font-size: 9pt;
}}
QPushButton[chip="true"]:hover {{
    border-color: {ACCENT_DARK};
    color: {ACCENT};
}}

/* ---------- 输入 ---------- */
QLineEdit, QPlainTextEdit, QTextEdit, QTextBrowser, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {SURFACE_ALT};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 6px 9px;
    selection-background-color: {ACCENT_DARK};
    selection-color: #ffffff;
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QTextBrowser:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT_DARK};
}}
QPlainTextEdit[role="chatInput"] {{
    background: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 10px 12px;
    font-size: 10.5pt;
}}
QPlainTextEdit[role="chatInput"]:focus {{
    border-color: {ACCENT};
}}
QPlainTextEdit[role="log"] {{
    font-family: {FONT_MONO};
    font-size: 9pt;
    color: {TEXT_MUTED};
    background: #0f151d;
    border: 1px solid {BORDER_SOFT};
    border-radius: 8px;
    padding: 8px;
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox QAbstractItemView {{
    background: {CARD};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT_SOFT};
    selection-color: {ACCENT};
    outline: none;
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: 16px;
    background: transparent;
    border: none;
}}

/* ---------- 列表 ---------- */
QListWidget {{
    background: {SURFACE};
    border: 1px solid {BORDER_SOFT};
    border-radius: 10px;
    padding: 4px;
    outline: none;
    font-size: 9pt;
}}
QListWidget::item {{
    padding: 9px 10px;
    border-radius: 7px;
    color: {TEXT_MUTED};
    font-size: 9pt;
}}
QListWidget::item:hover {{
    background: {SURFACE_ALT};
    color: {TEXT};
}}
QListWidget::item:selected {{
    background: {ACCENT_SOFT};
    color: {ACCENT};
}}

/* ---------- 滚动条 ---------- */
QScrollArea {{
    background: transparent;
    border: none;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: #2c3b4b;
    border-radius: 5px;
    min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{
    background: #3b4e61;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: #2c3b4b;
    border-radius: 5px;
    min-width: 28px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ---------- 进度条 ---------- */
QProgressBar {{
    background: {SURFACE_ALT};
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 4px;
}}
QProgressBar[state="success"]::chunk {{
    background: {SUCCESS};
}}
QProgressBar[state="warning"]::chunk {{
    background: {WARNING};
}}
QProgressBar[state="danger"]::chunk {{
    background: {DANGER};
}}

/* ---------- 复选框 ---------- */
QCheckBox {{
    color: {TEXT_MUTED};
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid {BORDER};
    background: {SURFACE_ALT};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}

/* ---------- 分割线 ---------- */
QFrame[role="hline"] {{
    background: {BORDER_SOFT};
    max-height: 1px;
    border: none;
}}
QSplitter::handle {{
    background: transparent;
}}

/* ---------- 聊天气泡 ---------- */
QFrame[bubble="true"] {{
    border-radius: 12px;
}}
QFrame[bubble="true"][role="user"] {{
    background: {USER_BUBBLE};
    border: 1px solid #1d5674;
}}
QFrame[bubble="true"][role="assistant"] {{
    background: {ASSISTANT_BUBBLE};
    border: 1px solid {BORDER_SOFT};
}}
QFrame[bubble="true"][role="system"] {{
    background: {SURFACE};
    border: 1px dashed {BORDER};
}}
QFrame[bubble="true"][role="proactive"] {{
    background: {WARNING_SOFT};
    border: 1px solid #5a4520;
}}
QLabel[role="avatar"] {{
    border-radius: 14px;
    font-weight: 700;
    font-size: 9pt;
}}
QLabel[role="avatar"][who="user"] {{
    background: {ACCENT_DARK};
    color: #ffffff;
}}
QLabel[role="avatar"][who="g1"] {{
    background: {SUCCESS_SOFT};
    color: {SUCCESS};
}}
QLabel[role="avatar"][who="sys"] {{
    background: {SURFACE_ALT};
    color: {TEXT_MUTED};
}}

/* ---------- 工具调用卡 ---------- */
QFrame[tool="true"] {{
    background: {TOOL_BUBBLE};
    border: 1px solid {BORDER_SOFT};
    border-left: 3px solid {ACCENT_DARK};
    border-radius: 8px;
}}
QFrame[tool="true"][state="running"] {{
    border-left-color: {WARNING};
}}
QFrame[tool="true"][state="error"] {{
    border-left-color: {DANGER};
}}
QFrame[tool="true"][state="ok"] {{
    border-left-color: {SUCCESS};
}}
QLabel[toolName="true"] {{
    font-family: {FONT_MONO};
    font-weight: 600;
    color: {TEXT};
    font-size: 9.5pt;
}}
QToolButton {{
    background: transparent;
    color: {TEXT_MUTED};
    border: none;
    padding: 2px 6px;
    border-radius: 5px;
}}
QToolButton:hover {{
    background: {SURFACE_ALT};
    color: {TEXT};
}}

/* ---------- 状态胶囊 ---------- */
QLabel[chip="true"] {{
    border-radius: 9px;
    padding: 2px 8px;
    font-size: 8.5pt;
    font-weight: 600;
}}
QLabel[chip="true"][state="idle"] {{
    background: {SURFACE_ALT};
    color: {TEXT_MUTED};
}}
QLabel[chip="true"][state="ok"] {{
    background: {SUCCESS_SOFT};
    color: {SUCCESS};
}}
QLabel[chip="true"][state="warn"] {{
    background: {WARNING_SOFT};
    color: {WARNING};
}}
QLabel[chip="true"][state="danger"] {{
    background: {DANGER_SOFT};
    color: {DANGER};
}}
QLabel[chip="true"][state="info"] {{
    background: {ACCENT_SOFT};
    color: {ACCENT};
}}

/* ---------- 页头 ---------- */
QFrame[role="header"] {{
    background: {SURFACE};
    border-bottom: 1px solid {BORDER_SOFT};
}}
QFrame[role="sidebar"] {{
    background: {SURFACE};
    border-right: 1px solid {BORDER_SOFT};
}}
QFrame[role="statusbar"] {{
    background: {SURFACE};
    border-top: 1px solid {BORDER_SOFT};
}}
"""
