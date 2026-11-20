"""演示：把"大模型会下达的那串指令"脚本化跑一遍，并录成 GIF。

这条脚本等价于用户说：
    "往前走一米 -> 举起双手 -> 向右转 90 度 -> 张开双臂 -> 回到原点 -> 停下"
不连大模型，纯粹为了**录一段能看的执行过程**，同时也当回归测试用。
"""

# 支持从项目根目录、IDE及脚本所在目录启动。
from pathlib import Path as _ProjectPath
import sys as _project_sys
import os as _project_os
_project_root = _ProjectPath(__file__).resolve().parents[1]
_project_sys.path.insert(0, str(_project_root))
_project_os.chdir(_project_root)


import json
import sys

sys.path.insert(0, ".")
from robot_bridge import RobotSession

# 等价于模型返回的一串 tool_calls：(工具名, 参数)
SCRIPT = [
    ("robot_status", {}),
    ("robot_goto", {"x": 0.9, "y": 0.0, "yaw": 0}),
    ("robot_set_arm", {"preset": "双手上举"}),
    ("robot_turn", {"degrees": 90}),
    ("robot_set_arm", {"preset": "张开双臂"}),
    ("robot_goto", {"x": 0.9, "y": 0.0, "yaw": -90, "timeout": 10}),
    ("robot_set_arm", {"preset": "自然下垂（推荐）"}),
    ("robot_stop", {}),
    ("robot_status", {}),
]


def main() -> int:
    bot = RobotSession(verbose=True, record_frames=True, record_fps=12,
                       record_size=(360, 270))

    # 直接调 session 的方法（等价于 RobotToolkit.dispatch 之后的调用）
    dispatch = {
        "robot_status": lambda p: bot.describe(),
        "robot_goto": lambda p: bot.goto(p["x"], p["y"], p.get("yaw")),
        "robot_set_arm": lambda p: (bot.set_arm_preset(p["preset"]) if p.get("preset")
                                    else bot.set_arm_joints(p.get("side", "both"),
                                                            **p.get("joints", {}))),
        "robot_sequence": lambda p: bot.sequence(p["steps"]),
        "robot_walk": lambda p: bot.walk_for(p.get("vx", 0), p.get("vy", 0),
                                             p.get("vyaw", 0), p.get("seconds", 1)),
        "robot_turn": lambda p: bot.turn(p["degrees"]),
        "robot_stop": lambda p: bot.stop(),
        "robot_reset": lambda p: bot.reset(),
    }

    for name, params in SCRIPT:
        print("\n>>> %s(%s)" % (name, json.dumps(params, ensure_ascii=False)))
        print(dispatch[name](params))
        bot._wait_sim(0.25)          # 每步之间稍微停一下，录出来看得清

    print("\n" + bot.save_gif("outputs/g1_llm_demo.gif"))
    print("动作历史：")
    for line in bot._action_log:
        print("   ", line)
    bot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
