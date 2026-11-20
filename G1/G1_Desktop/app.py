"""G1 桌面控制台启动器。

用法::

    python -m G1_Desktop.app
    python -m G1_Desktop.app --no-viewer
    python -m G1_Desktop.app --check
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DESKTOP_ROOT = Path(__file__).resolve().parent
G1_ROOT = DESKTOP_ROOT.parent
G1_SOURCE = G1_ROOT / "G1_LLM_Robot" / "G1_LLM_Robot"
EXTENSION_ROOT = G1_ROOT.parent / "G1_DeepSeek_Extensions"
SUPERMIMIC_ROOT = G1_ROOT.parents[1] / "SuperMimic"
RAX_ROOT = G1_ROOT.parents[1] / "RAX"


def prepare_environment() -> None:
    """把 G1 源码和扩展包加入 import 路径，并统一工作目录。"""
    if not (G1_SOURCE / "deepseek_robot.py").is_file():
        raise SystemExit("G1 源码目录不完整：%s" % G1_SOURCE)
    if not (EXTENSION_ROOT / "g1_deepseek_ext").is_dir():
        raise SystemExit("DeepSeek 扩展目录不完整：%s" % EXTENSION_ROOT)

    for path in (str(EXTENSION_ROOT), str(G1_SOURCE), str(G1_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)

    os.environ.setdefault("G1_DEEPSEEK_G1_ROOT", str(G1_SOURCE))
    os.environ.setdefault("G1_DEEPSEEK_SUPERMIMIC_ROOT", str(SUPERMIMIC_ROOT))
    os.environ.setdefault("G1_DEEPSEEK_RAX_ROOT", str(RAX_ROOT))
    os.environ.setdefault("PYTHONUTF8", "1")
    os.chdir(G1_SOURCE)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def preload_robot_stack() -> tuple[bool, str]:
    """在导入 Qt 之前加载 torch / MuJoCo。

    Windows + Python 3.14 下实测：先导入 PyQt5 再导入 torch 或 mujoco，会触发
    ``WinError 1114（DLL 初始化例程失败）``。反过来先加载这两个库，Qt 再加载
    就正常。因此机器人模式下必须在创建 QApplication 之前完成预加载。
    """
    try:
        import mujoco  # noqa: F401
        import torch  # noqa: F401
        return True, ""
    except Exception as exc:  # DLL 缺失、CUDA 运行库不匹配等
        return False, "%s: %s" % (type(exc).__name__, exc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="G1_Desktop",
        description="G1 具身智能桌面控制台（对话 + 机器人 + 训练 + 环境）",
    )
    parser.add_argument("--key", help="DeepSeek API Key；缺省读取环境变量或 api_key.txt")
    parser.add_argument(
        "--model",
        default=os.environ.get("G1_MODEL_NAME", "deepseek-flash"),
        help="模型名，默认 deepseek-flash",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("G1_MODEL_BASE_URL", "https://api.deepseek.com"),
        help="OpenAI 兼容接口地址",
    )
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "max"),
        default="medium",
    )
    parser.add_argument("--no-robot", action="store_true", help="只启动对话，不加载 MuJoCo")
    parser.add_argument("--no-viewer", action="store_true", help="不打开 MuJoCo 3D 窗口")
    parser.add_argument("--headless", action="store_true", help="--no-viewer 的别名")
    parser.add_argument("--walker-dir", help="g1_walkfree 包所在目录")
    parser.add_argument("--speed-scale", type=float, default=1.0, help="动作安全限幅缩放")
    parser.add_argument("--no-proactive", action="store_true", help="关闭主动训练报告")
    parser.add_argument(
        "--proactive-interval-minutes", type=float, default=10.0, help="主动报告间隔（分钟）"
    )
    parser.add_argument("--no-search", action="store_true", help="不启用 Tavily 联网搜索")
    parser.add_argument("--check", action="store_true", help="只检查路径与依赖后退出")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    prepare_environment()

    if args.check:
        problems = []
        for name, path in (
            ("G1 源码", G1_SOURCE),
            ("扩展包", EXTENSION_ROOT),
            ("SuperMimic", SUPERMIMIC_ROOT),
            ("RAX", RAX_ROOT),
        ):
            print("%-12s %s  %s" % (name, "OK " if path.exists() else "缺失", path))
            if not path.exists():
                problems.append(name)
        try:
            import PyQt5  # noqa: F401

            print("%-12s OK" % "PyQt5")
        except ImportError:
            print("%-12s 缺失（pip install PyQt5）" % "PyQt5")
            problems.append("PyQt5")
        return 0 if not problems else 2

    enable_robot = not args.no_robot
    startup_warning = ""
    if enable_robot:
        # 关键：必须早于任何 PyQt5 导入。
        ok, error = preload_robot_stack()
        if not ok:
            enable_robot = False
            startup_warning = "MuJoCo/torch 预加载失败，已降级为纯对话模式：%s" % error
            print("警告：" + startup_warning, file=sys.stderr)

    try:
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QApplication
    except ImportError:
        print("缺少 PyQt5：pip install PyQt5", file=sys.stderr)
        return 2

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("G1 Desktop")
    app.setOrganizationName("G1")

    from . import theme

    theme.apply(app)

    from .main_window import MainWindow

    options = {
        "key": args.key,
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "reasoning_effort": args.reasoning_effort,
        "enable_robot": enable_robot,
        "viewer": enable_robot and not (args.no_viewer or args.headless),
        "walker_dir": args.walker_dir,
        "speed_scale": args.speed_scale,
        "enable_search": not args.no_search,
        "proactive": not args.no_proactive,
        "proactive_interval": max(6.0, args.proactive_interval_minutes * 60.0),
        "startup_warning": startup_warning,
    }

    window = MainWindow(options)
    window.show()
    window.start_backend()
    return app.exec_()


if __name__ == "__main__":
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from G1_Desktop.app import main as _main

        raise SystemExit(_main())
    raise SystemExit(main())
