
# 支持从项目根目录、IDE及脚本所在目录启动。
from pathlib import Path as _ProjectPath
import sys as _project_sys
import os as _project_os
_project_root = _ProjectPath(__file__).resolve().parents[1]
_project_sys.path.insert(0, str(_project_root))
_project_os.chdir(_project_root)

import sys, time
sys.path.insert(0, ".")
from robot_bridge import RobotSession, available_arm_presets

t0 = time.perf_counter()
bot = RobotSession(verbose=True)
print("启动耗时 %.2fs" % (time.perf_counter() - t0))
print("\n初始状态：")
print(bot.describe())

print("\n--- 1) 前进 2 仿真秒 ---")
t0 = time.perf_counter()
print(bot.walk_for(vx=0.5, seconds=2.0))
print("   (墙上耗时 %.2fs)" % (time.perf_counter() - t0))

print("\n--- 2) 闭环 goto 到 (1.0, 0.0) 朝向 0° ---")
t0 = time.perf_counter()
print(bot.goto(1.0, 0.0, yaw=0.0, timeout=15.0))
print("   (墙上耗时 %.2fs)" % (time.perf_counter() - t0))

print("\n--- 3) goto 回起点 (0, 0) ---")
print(bot.goto(0.0, 0.0, yaw=0.0, timeout=15.0))

print("\n--- 4) 原地左转 90° ---")
print(bot.turn(90.0, timeout=10.0))

print("\n--- 5) 手臂预设 ---")
print(bot.set_arm_preset("双手前平举"))
print(bot.set_arm_preset("双手上举"))

print("\n--- 6) 直接设手臂关节 ---")
print(bot.set_arm_joints(side="left", shoulder_pitch=-1.0, elbow=0.5))

print("\n--- 7) 腰部 ---")
print(bot.set_waist(yaw=0.25, pitch=0.1))

print("\n--- 8) 动作序列 ---")
seq = [
    {"type": "arm", "preset": "自然下垂（推荐）"},
    {"type": "goto", "x": 0.6, "y": 0.4, "yaw": 45},
    {"type": "arm", "preset": "张开双臂"},
    {"type": "stop"},
]
t0 = time.perf_counter()
print(bot.sequence(seq))
print("   (墙上耗时 %.2fs)" % (time.perf_counter() - t0))

print("\n--- 9) 安全限幅测试（故意给超大指令）---")
print(bot.walk_for(vx=9.9, vy=9.9, vyaw=9.9, seconds=0.5))

print("\n--- 10) 复位 ---")
print(bot.reset())
print("\n最终：")
print(bot.describe())
print("\n动作历史：")
for line in bot._action_log:
    print("   ", line)

bot.close()
print("\n总耗时 %.1fs" % (time.perf_counter() - t0))
