"""作为模块调用时，怎么把仿真画面显示出来 —— 四种写法 + 常见坑。

直接运行看效果：
    python example_module_usage.py              # 默认：RobotSession + 开窗口
    python example_module_usage.py --mode late  # 先无界面跑，再开窗口
    python example_module_usage.py --mode chat  # RobotChatBot 带窗口（需要 API Key）
    python example_module_usage.py --no-view    # 不开窗口（验证动作本身）

=========================== 核心结论（先看这里）===========================

1) 开窗口只有一个开关：``show_viewer=True`` / ``robot_viewer=True``
   打开后**会自动切成按真实时间跑**（realtime=True）——否则仿真全速跑，
   一个 2 秒的动作零点几秒就过去了，窗口里几乎什么都看不清。

2) MuJoCo 的被动查看器**不阻塞**：脚本跑完就退出、窗口跟着消失。
   想让窗口一直留着，末尾要调 ``wait_until_viewer_closed()``。

3) 窗口里的仿真**只在有动作执行时才推进**（空闲冻结）。
   所以"开着窗口但没下指令"时，画面会停在当前姿态——这是正常现象，
   不是卡死（鼠标仍然能转动视角）。

4) 打不开窗口也不会影响仿真：无图形界面/缺 OpenGL 时会自动降级，
   动作照常执行，只是没有画面。
==========================================================================
"""

from __future__ import annotations

# 支持从项目根目录、IDE及脚本所在目录启动。
from pathlib import Path as _ProjectPath
import sys as _project_sys
import os as _project_os
_project_root = _ProjectPath(__file__).resolve().parents[1]
_project_sys.path.insert(0, str(_project_root))
_project_os.chdir(_project_root)


import argparse
import sys

sys.path.insert(0, ".")

from robot_bridge import RobotSession


# ======================================================================
# 写法一：直接用 RobotSession（最省事）
# ======================================================================
def demo_session(show_viewer: bool, hold: bool) -> int:
    print("【写法一】RobotSession(show_viewer=%s)" % show_viewer)

    with RobotSession(show_viewer=show_viewer, verbose=True) as bot:
        print("  窗口是否开着：", bot.viewer_running)

        print(" ", bot.goto(0.8, 0.0, yaw=0, timeout=12.0).splitlines()[0])
        print(" ", bot.set_arm_preset("双手上举").splitlines()[0])
        print(" ", bot.turn(90, timeout=8.0).splitlines()[0])
        print(" ", bot.set_arm_preset("张开双臂").splitlines()[0])
        print(" ", bot.stop().splitlines()[0])
        print(" ", bot.describe().splitlines()[0])

        # 让窗口留一会儿，看完再退出（不调这句脚本一结束窗口就没了）
        if hold and bot.viewer_running:
            print("\n  窗口保持打开 8 秒……（也可以调 wait_until_viewer_closed() 一直等）")
            bot.wait_until_viewer_closed(timeout=8.0)
    return 0


# ======================================================================
# 写法二：先无界面跑，需要时再开窗口
# ======================================================================
def demo_late(hold: bool) -> int:
    print("【写法二】先无界面，运行中再 open_viewer()")

    with RobotSession(show_viewer=False, verbose=True) as bot:
        print("  一开始没窗口：", bot.viewer_running)

        # 先让机器人站好（无界面，飞快）
        bot.reset()

        # 想看了再开（默认顺便切到实时，否则看不清）
        print(" ", bot.open_viewer())
        print("  现在窗口开着：", bot.viewer_running)

        print(" ", bot.goto(0.8, 0.0, yaw=0, timeout=12.0).splitlines()[0])
        print(" ", bot.set_arm_preset("自然下垂（推荐）").splitlines()[0])
        print(" ", bot.turn(-90, timeout=8.0).splitlines()[0])

        if hold and bot.viewer_running:
            print("\n  窗口保持打开 8 秒……")
            bot.wait_until_viewer_closed(timeout=8.0)

        # 也可以中途关掉，仿真照常
        print(" ", bot.close_viewer())
    return 0


# ======================================================================
# 写法三：通过 RobotChatBot（大模型对话 + 窗口）
# ======================================================================
def demo_chat(hold: bool) -> int:
    print("【写法三】RobotChatBot(robot_viewer=True)")

    try:
        from deepseek_robot import RobotChatBot
    except Exception as exc:
        print("  导入失败：", exc)
        return 2

    try:
        bot = RobotChatBot(robot_viewer=True, verbose=True, enable_local_files=False)
    except Exception as exc:
        print("  启动失败（多半是没有 API Key）：", exc)
        return 2

    try:
        print("  窗口是否开着：", bot.viewer_running)
        print(bot.chat("走到(0.8, 0)然后举起双手，告诉我结果"))
        if hold and bot.viewer_running:
            bot.wait_until_viewer_closed(timeout=10.0)
    finally:
        bot.close()
    return 0


# ======================================================================
# 写法四：只要"每个动作都停留在画面里"的最简做法
# ======================================================================
def demo_minimal() -> int:
    """最短的一段可复制代码。"""
    print("【写法四】最短示例（就是这 5 行）")
    print("""
    from robot_bridge import RobotSession

    bot = RobotSession(show_viewer=True)      # 开窗口 + 自动实时
    bot.goto(0.8, 0.0, yaw=0)                 # 走过去
    bot.set_arm_preset("双手上举")             # 举手
    bot.wait_until_viewer_closed()            # 挂住，看完再关窗口
    bot.close()
""")
    return 0


# ======================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["session", "late", "chat", "minimal"],
                    default="session", help="演示哪种写法")
    ap.add_argument("--no-view", action="store_true", help="不开 3D 窗口")
    ap.add_argument("--no-hold", action="store_true", help="跑完立刻退出，不等窗口")
    args = ap.parse_args()

    hold = not args.no_hold
    if args.mode == "session":
        return demo_session(show_viewer=not args.no_view, hold=hold)
    if args.mode == "late":
        return demo_late(hold=hold)
    if args.mode == "chat":
        return demo_chat(hold=hold)
    return demo_minimal()


if __name__ == "__main__":
    raise SystemExit(main())
