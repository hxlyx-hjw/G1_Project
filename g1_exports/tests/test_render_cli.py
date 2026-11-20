"""render_policy_video / render_reference_video 的命令行契约测试。

这些测试不加载 MuJoCo、torch 或训练包，只验证参数解析和错误提示，
因此在没有部署训练包的环境中也能运行。
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_script(name, *args):
    return subprocess.run(
        [sys.executable, name, *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )


class RenderCliTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name)
        # 让脚本通过 reference 校验，只测试训练包缺失时的提示。
        (self.run_dir / "reference.npz").write_bytes(b"")
        self.missing_package = str(self.run_dir / "_no_such_package_")

    def tearDown(self):
        self._tmp.cleanup()

    def test_reference_help(self):
        result = run_script("render_reference_video.py", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--max-seconds", result.stdout)
        self.assertIn("--package", result.stdout)

    def test_policy_help(self):
        result = run_script("render_policy_video.py", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--checkpoint", result.stdout)
        self.assertIn("--package", result.stdout)

    def test_reference_missing_package(self):
        result = run_script(
            "render_reference_video.py",
            "--package",
            self.missing_package,
            "--run",
            str(self.run_dir),
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("训练包代码不完整", result.stderr)

    def test_policy_missing_package(self):
        result = run_script(
            "render_policy_video.py",
            "--package",
            self.missing_package,
            "--run",
            str(self.run_dir),
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("训练包代码不完整", result.stderr)


if __name__ == "__main__":
    unittest.main()
