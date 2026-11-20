"""Unified G1 entry: DeepSeek chat + MuJoCo viewer + research tools."""

from __future__ import annotations

import os
import sys
from pathlib import Path


G1_ROOT = Path(__file__).resolve().parent
G1_SOURCE = G1_ROOT / "G1_LLM_Robot" / "G1_LLM_Robot"
EXTENSION_ROOT = G1_ROOT.parent / "G1_DeepSeek_Extensions"
SUPERMIMIC_ROOT = G1_ROOT.parents[1] / "SuperMimic"
RAX_ROOT = G1_ROOT.parents[1] / "RAX"


def main(argv: list[str] | None = None) -> int:
    if not (G1_SOURCE / "deepseek_robot.py").is_file():
        print("G1 源码目录不完整：%s" % G1_SOURCE, file=sys.stderr)
        return 2
    if not (EXTENSION_ROOT / "start_chat.py").is_file():
        print("DeepSeek 扩展目录不完整：%s" % EXTENSION_ROOT, file=sys.stderr)
        return 2

    os.environ.setdefault("G1_DEEPSEEK_G1_ROOT", str(G1_SOURCE))
    os.environ.setdefault("G1_DEEPSEEK_SUPERMIMIC_ROOT", str(SUPERMIMIC_ROOT))
    os.environ.setdefault("G1_DEEPSEEK_RAX_ROOT", str(RAX_ROOT))
    os.environ.setdefault("PYTHONUTF8", "1")

    raw_args = list(sys.argv[1:] if argv is None else argv)
    headless = (
        "--headless" in raw_args
        or os.environ.get("G1_UNIFIED_HEADLESS", "").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    if not headless and "--viewer" not in raw_args:
        raw_args.insert(0, "--viewer")

    extension = str(EXTENSION_ROOT.resolve())
    if extension not in sys.path:
        sys.path.insert(0, extension)
    from start_chat import main as extension_main

    return extension_main(raw_args)


if __name__ == "__main__":
    raise SystemExit(main())
