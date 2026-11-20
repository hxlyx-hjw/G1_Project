"""端到端验证：真连 DeepSeek，看模型能否自己生成并执行机器人命令。

Key 的来源（按优先级）：
    1. 环境变量 DEEPSEEK_API_KEY
    2. 当前目录的 api_key.txt
    3. 附件里的 PythonApplication2.py（仅本机有，找不到就跳过）
Key 不会打印到屏幕上。
"""

# 支持从项目根目录、IDE及脚本所在目录启动。
from pathlib import Path as _ProjectPath
import sys as _project_sys
import os as _project_os
_project_root = _ProjectPath(__file__).resolve().parents[1]
_project_sys.path.insert(0, str(_project_root))
_project_os.chdir(_project_root)

import os
import re
import sys

sys.path.insert(0, ".")


def load_key() -> str:
    """按优先级找 Key，返回 Key 本身（不打印）。"""
    # 1) 环境变量
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        print("Key 来源：环境变量  长度 %d" % len(key))
        return key.strip()

    # 2) 当前目录 api_key.txt
    if os.path.isfile("api_key.txt"):
        key = open("api_key.txt", encoding="utf-8").read().strip()
        if key:
            print("Key 来源：api_key.txt  长度 %d" % len(key))
            return key

    # 3) 附件（开发者本机才有）
    for cand in (r"..\attachments\c2b0df24faf8\PythonApplication2.py",
                 "attachments/c2b0df24faf8/PythonApplication2.py"):
        if os.path.isfile(cand):
            src = open(cand, encoding="utf-8", errors="ignore").read()
            m = re.search(r'iDEEPSEEK_API_KEY1\s*=\s*"([^"]+)"', src)
            if m:
                print("Key 来源：%s  长度 %d" % (cand, len(m.group(1))))
                return m.group(1)

    raise SystemExit(
        "找不到 API Key。请任选一种：\n"
        "  * 设环境变量 DEEPSEEK_API_KEY\n"
        "  * 在当前目录建 api_key.txt，里面只写 Key\n"
        "  * python e2e_test.py 之前先 export")


os.environ["DEEPSEEK_API_KEY"] = load_key()

from deepseek_robot import RobotChatBot                     # noqa: E402

bot = RobotChatBot(verbose=True, enable_local_files=False, enable_search=False)

TASKS = [
    "让机器人向前走到坐标 (0.8, 0) 的位置，然后举起双手，"
    "最后告诉我它现在在哪、朝向多少、手臂是什么姿态。",
]

for i, task in enumerate(TASKS, 1):
    print("\n" + "=" * 62)
    print("任务 %d：%s" % (i, task))
    print("=" * 62)
    reply = bot.chat(task)
    print("\n--- 模型最终回复 ---")
    print(reply)

print("\n" + "=" * 62)
print("机器人真实动作历史：")
for line in (bot.session._action_log if bot.session else []):
    print("   ", line)
bot.close()
