"""Start G1 DeepSeek chat with external SuperMimic and RAX tools."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_G1_SOURCE = Path(
    r"E:\TEST_RECORD\G1_Project\G1\G1_LLM_Robot\G1_LLM_Robot"
)
G1_SOURCE = Path(
    os.environ.get("G1_DEEPSEEK_G1_ROOT", str(DEFAULT_G1_SOURCE))
).expanduser().resolve()

if not (G1_SOURCE / "deepseek_robot.py").is_file():
    raise SystemExit("G1 source root is invalid: %s" % G1_SOURCE)
if str(G1_SOURCE) not in sys.path:
    sys.path.insert(0, str(G1_SOURCE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(G1_SOURCE)

from deepseek_robot import build_parser  # noqa: E402
from g1_deepseek_ext.chat import ExtendedRobotChatBot  # noqa: E402
from g1_deepseek_ext.adapters import (  # noqa: E402
    DeepSeekIntegration,
    IntegrationConfig,
    integration_schemas,
)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    parser.add_argument(
        "--list-integration-tools",
        action="store_true",
        help="只打印 SuperMimic/RAX 扩展工具声明后退出",
    )
    parser.add_argument(
        "--integration-status",
        action="store_true",
        help="不连接 DeepSeek，只检查 SuperMimic/RAX 环境后退出",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="不打开 MuJoCo 窗口；统一启动器默认会开启窗口",
    )
    args = parser.parse_args(argv)

    if args.list_integration_tools:
        print(json.dumps(integration_schemas(), ensure_ascii=False, indent=2))
        return 0

    config = IntegrationConfig.from_env()
    integration = DeepSeekIntegration(config)
    if args.integration_status:
        print(integration.dispatch("robot_supermimic_status", {}))
        print(integration.dispatch("robot_rax_status", {}))
        return 0

    if args.list_tools:
        from robot_tools import RobotToolkit

        print(
            json.dumps(
                RobotToolkit(None).schemas() + integration_schemas(),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.selftest:
        from deepseek_robot import selftest

        return selftest(args)

    try:
        bot = ExtendedRobotChatBot(
            deepseek_api_key=args.key,
            tavily_api_key=args.tavily_key,
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            enable_search=args.search,
            enable_local_files=not args.no_files,
            enable_robot=not args.no_robot,
            robot_viewer=args.viewer and not args.headless,
            robot_realtime=(
                True
                if args.realtime
                else (True if args.viewer and not args.headless else None)
            ),
            walker_dir=args.walker_dir,
            robot_speed_scale=args.speed_scale,
            verbose=not args.quiet,
            enable_proactive=not args.no_proactive,
            proactive_interval_seconds=(
                args.proactive_interval_minutes * 60.0
                if args.proactive_interval_minutes is not None
                else None
            ),
            proactive_use_model=not args.proactive_no_model,
            integration=integration,
        )
    except Exception as exc:
        print("[启动失败] %s" % exc)
        return 2

    try:
        if args.task:
            print("任务：%s\n" % args.task)
            print(bot.chat(args.task))
        else:
            bot.run_cli()
    finally:
        bot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
