"""离屏 GUI 冒烟测试：不连接 DeepSeek、不启动 MuJoCo。

运行::

    cd E:\\TEST_RECORD\\G1_Project\\G1
    python -m unittest discover -s G1_Desktop/tests -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt5.QtWidgets import QApplication  # noqa: E402

from G1_Desktop import theme  # noqa: E402
from G1_Desktop.main_window import MainWindow  # noqa: E402
from G1_Desktop.widgets import elide_path, rich_text  # noqa: E402


class WidgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        theme.apply(cls.app)

    def test_rich_text_and_elide(self) -> None:
        html = rich_text("**粗体**\n- 项目\n`code`")
        self.assertIn("<b>粗体</b>", html)
        self.assertIn("• 项目", html)
        short = elide_path("E:\\A\\B", 20)
        self.assertEqual(short, "E:\\A\\B")
        long = elide_path("E:\\TEST_RECORD\\A\\B\\C\\D\\E", 20)
        self.assertIn("…", long)
        self.assertLessEqual(len(long), 20)

    def test_main_window_pages(self) -> None:
        window = MainWindow(
            {"enable_robot": False, "viewer": False, "proactive": False, "startup_warning": ""}
        )
        try:
            self.assertEqual(window.stack.count(), 4)
            self.assertEqual(len(window.nav_group.buttons()), 4)

            window.chat_page.add_user("测试")
            window.chat_page.tool_started("robot_status", "{}")
            window.chat_page.tool_finished(
                "robot_status",
                '{"time": 1.0, "x": 0.1, "reliability": {"quality": 0.9}}',
                0.2,
            )
            window.chat_page.add_assistant("回复", "推理")
            self.app.processEvents()

            window.robot_page.on_status(
                {
                    "time": 5.0,
                    "x": 0.25,
                    "y": -0.1,
                    "z": 0.78,
                    "yaw": 12.5,
                    "vx": 0.1,
                    "vy": 0.0,
                    "speed": 0.1,
                    "fallen": False,
                    "reliability": {"quality": 0.8},
                }
            )
            self.assertIn("0.25", window.robot_page.pos.value.text())

            window.training_page.on_jobs(
                [
                    {
                        "job_id": "20261104_193012_a1b2c3",
                        "state": "training",
                        "iteration": 10,
                        "total_iterations": 100,
                        "worker_running": True,
                        "output_dir": "E:\\runs\\x",
                    }
                ]
            )
            self.assertEqual(window.training_page.jobs.count(), 1)

            window.env_page.on_ready(
                {
                    "model": "deepseek-flash",
                    "base_url": "https://api.deepseek.com",
                    "temperature": 0.3,
                    "reasoning_effort": "medium",
                    "paths": {"g1_root": "E:\\TEST_RECORD\\G1_Project\\G1"},
                    "tools": [
                        {
                            "name": "robot_status",
                            "description": "查询状态",
                            "parameters": {},
                            "source": "机器人",
                        }
                    ],
                }
            )
            self.assertEqual(window.env_page.tool_list.count(), 1)
        finally:
            window.hide()
            window.service.shutdown()
            window.deleteLater()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
