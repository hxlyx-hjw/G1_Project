from __future__ import annotations

import contextlib
import io
import json
import importlib.util
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from g1_deepseek_ext.adapters import (
    DeepSeekIntegration,
    IntegrationConfig,
    IntegrationError,
    JobManager,
    SuperMimicAdapter,
    integration_schemas,
    resolve_under,
)
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
G1 = Path(r"E:\TEST_RECORD\G1_Project\G1\G1_LLM_Robot\G1_LLM_Robot")
SUPERMIMIC = Path(r"E:\TEST_RECORD\SuperMimic")
RAX = Path(r"E:\TEST_RECORD\RAX")
if str(G1) not in sys.path:
    sys.path.insert(0, str(G1))


class IntegrationTests(unittest.TestCase):
    def test_extension_names_are_unique_and_complete(self):
        schemas = integration_schemas()
        names = [item["function"]["name"] for item in schemas]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("robot_supermimic_evaluate", names)
        self.assertIn("robot_rax_run", names)
        self.assertEqual(set(names), DeepSeekIntegration(
            IntegrationConfig(
                g1_root=G1,
                supermimic_root=SUPERMIMIC,
                rax_root=RAX,
                jobs_root=ROOT / "runtime" / "test-jobs",
            )
        ).names())

    def test_path_guard_rejects_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "root"
            root.mkdir()
            child = root / "safe"
            child.mkdir()
            self.assertEqual(resolve_under(root, "safe"), child.resolve())
            with self.assertRaises(IntegrationError):
                resolve_under(root, "../outside")

    def test_status_tools_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            config = IntegrationConfig(
                g1_root=G1,
                supermimic_root=SUPERMIMIC,
                rax_root=RAX,
                jobs_root=Path(temp) / "jobs",
            )
            integration = DeepSeekIntegration(config)
            rax = json.loads(integration.dispatch("robot_rax_status", {}))
            self.assertTrue(rax["success"])
            self.assertTrue(rax["latest_scaling"]["runs"])
            supermimic = json.loads(
                integration.dispatch("robot_supermimic_status", {})
            )
            self.assertEqual(supermimic["backend"], "SuperMimic")
            self.assertIn("available", supermimic)

    def test_rax_list_returns_existing_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            config = IntegrationConfig(
                g1_root=G1,
                supermimic_root=SUPERMIMIC,
                rax_root=RAX,
                jobs_root=Path(temp) / "jobs",
            )
            integration = DeepSeekIntegration(config)
            result = json.loads(
                integration.dispatch("robot_rax_list", {"limit": 5})
            )
            self.assertTrue(result["success"])
            self.assertGreaterEqual(result["count"], 1)

    def test_extended_bot_registers_tools_without_robot_session(self):
        from g1_deepseek_ext.chat import ExtendedRobotChatBot

        with tempfile.TemporaryDirectory() as temp:
            config = IntegrationConfig(
                g1_root=G1,
                supermimic_root=SUPERMIMIC,
                rax_root=RAX,
                jobs_root=Path(temp) / "jobs",
            )
            integration = DeepSeekIntegration(config)
            bot = ExtendedRobotChatBot(
                deepseek_api_key="test-only",
                enable_robot=False,
                enable_local_files=False,
                verbose=False,
                enable_proactive=False,
                integration=integration,
            )
            try:
                registered = {item["function"]["name"] for item in bot.tools}
                self.assertIn("robot_supermimic_status", registered)
                self.assertIn("robot_rax_run", registered)
                self.assertIn("SuperMimic", bot.system_prompt)
                self.assertIn("RAX", bot.history[0]["content"])
            finally:
                bot.close()

    def test_detached_job_recorder_updates_status(self):
        with tempfile.TemporaryDirectory() as temp:
            jobs = JobManager(Path(temp) / "jobs")
            job_id = jobs.new_id("test")
            jobs.start(
                job_id=job_id,
                kind="unit_test",
                command=[
                    sys.executable,
                    "-c",
                    "import time; time.sleep(0.1); print('ok')",
                ],
                cwd=Path(temp),
            )
            deadline = time.monotonic() + 8
            status = jobs.status(job_id)
            while status["state"] in {"starting", "running"} and time.monotonic() < deadline:
                time.sleep(0.05)
                status = jobs.status(job_id)
            self.assertEqual(status["state"], "completed")
            self.assertEqual(status["returncode"], 0)
            self.assertIn("ok", status["log_tail"])

    def test_unified_entry_adds_viewer_and_honors_headless(self):
        launcher_path = ROOT.parent / "G1" / "Start_G1_Unified.py"
        spec = importlib.util.spec_from_file_location(
            "g1_unified_launcher", launcher_path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        received = []

        def fake_main(args):
            received.append(list(args))
            return 0

        fake_module = type(sys)("start_chat")
        fake_module.main = fake_main
        with patch.dict(sys.modules, {"start_chat": fake_module}):
            self.assertEqual(module.main(["--task", "测试"]), 0)
            self.assertIn("--viewer", received[-1])
            self.assertEqual(module.main(["--headless"]), 0)
            self.assertNotIn("--viewer", received[-1])

    def test_learn_video_schema_exposes_smplx_options(self):
        schema = next(
            item["function"]
            for item in integration_schemas()
            if item["function"]["name"] == "robot_supermimic_learn_video"
        )
        properties = schema["parameters"]["properties"]
        self.assertIn("smplx_model_dir", properties)
        self.assertIn("smplx_gender", properties)
        self.assertIn("smplx_device", properties)

    def test_learn_video_rejects_invalid_arguments_before_starting(self):
        video = ROOT.parent / "G1" / "vedio_self" / "test4.mp4"
        self.assertTrue(video.is_file())
        with tempfile.TemporaryDirectory() as temp:
            jobs_root = Path(temp) / "jobs"
            adapter = SuperMimicAdapter(
                IntegrationConfig(
                    g1_root=G1,
                    supermimic_root=SUPERMIMIC,
                    rax_root=RAX,
                    jobs_root=jobs_root,
                ),
                JobManager(jobs_root),
            )
            for bad in (
                {"root_mode": "floating"},
                {"min_visibility": 1.5},
                {"smplx_gender": "robot"},
                {"fps": 0},
            ):
                with self.assertRaises(IntegrationError):
                    adapter.learn_video(dict({"input": str(video)}, **bad))
            self.assertFalse(jobs_root.exists() and any(jobs_root.iterdir()))

    def test_job_status_surfaces_pipeline_progress(self):
        with tempfile.TemporaryDirectory() as temp:
            jobs_root = Path(temp) / "jobs"
            jobs = JobManager(jobs_root)
            output = Path(temp) / "output"
            output.mkdir()
            (output / "pipeline.json").write_text(
                json.dumps(
                    {
                        "state": "running",
                        "stage": "train",
                        "message": "SuperMimic 训练运行中",
                        "progress": 0.65,
                        "training_metrics": {"iteration": "3"},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            job_dir = jobs.root / "demo_job"
            job_dir.mkdir()
            (job_dir / "job.json").write_text(
                json.dumps(
                    {
                        "job_id": "demo_job",
                        "kind": "supermimic_video_to_training",
                        "state": "running",
                        "output_root": str(output),
                        "log": str(job_dir / "job.log"),
                    }
                ),
                encoding="utf-8",
            )
            adapter = SuperMimicAdapter(
                IntegrationConfig(
                    g1_root=G1,
                    supermimic_root=SUPERMIMIC,
                    rax_root=RAX,
                    jobs_root=jobs_root,
                ),
                jobs,
            )
            result = adapter.job_status({"job_id": "demo_job"})
            self.assertEqual(result["stage"], "train")
            self.assertEqual(result["progress"], 0.65)
            self.assertEqual(result["training_metrics"]["iteration"], "3")
            self.assertIsNone(result["success"])
            (output / "pipeline.json").write_text(
                json.dumps(
                    {
                        "state": "running",
                        "stage": "viewer_open",
                        "training_completed": True,
                    }
                ),
                encoding="utf-8",
            )
            result = adapter.job_status({"job_id": "demo_job"})
            self.assertEqual(result["state"], "viewer_open")
            self.assertTrue(result["success"])
            self.assertTrue(result["training_success"])

    def test_supermimic_logs_returns_live_metrics_and_tails(self):
        with tempfile.TemporaryDirectory() as temp:
            jobs_root = Path(temp) / "jobs"
            jobs = JobManager(jobs_root)
            output = Path(temp) / "output"
            training = output / "training"
            training.mkdir(parents=True)
            (training / "args.json").write_text(
                json.dumps({"iterations": 100}), encoding="utf-8"
            )
            (training / "train_log.csv").write_text(
                "iteration,wall_h,it_per_s,pi_loss\n"
                "10,0.01,8.0,-0.5\n"
                "20,0.02,10.0,-0.4\n",
                encoding="utf-8",
            )
            (output / "training.log").write_text(
                "it 20 | training running\n", encoding="utf-8"
            )
            (output / "viewer.log").write_text(
                "viewer ready\n", encoding="utf-8"
            )
            (output / "pipeline.json").write_text(
                json.dumps(
                    {
                        "state": "running",
                        "stage": "train",
                        "message": "training",
                        "progress": 0.65,
                        "training_completed": False,
                    }
                ),
                encoding="utf-8",
            )
            job_id = "log_job"
            job_dir = jobs.root / job_id
            job_dir.mkdir()
            (job_dir / "job.log").write_text(
                "job running\n", encoding="utf-8"
            )
            (job_dir / "job.json").write_text(
                json.dumps(
                    {
                        "job_id": job_id,
                        "kind": "supermimic_video_to_training",
                        "state": "running",
                        "output_root": str(output),
                        "log": str(job_dir / "job.log"),
                    }
                ),
                encoding="utf-8",
            )
            adapter = SuperMimicAdapter(
                IntegrationConfig(
                    g1_root=G1,
                    supermimic_root=SUPERMIMIC,
                    rax_root=RAX,
                    jobs_root=jobs_root,
                ),
                jobs,
            )
            result = adapter.logs(
                {"job_id": job_id, "lines": 10, "metrics_limit": 5}
            )
            self.assertEqual(result["training_progress"]["iteration"], 20)
            self.assertEqual(result["training_progress"]["iterations"], 100)
            self.assertAlmostEqual(
                result["training_progress"]["fraction"], 0.2
            )
            self.assertEqual(
                [row["iteration"] for row in result["recent_metrics"]],
                [10, 20],
            )
            self.assertIn("training running", result["logs"]["training_stdout"])
            self.assertIn("viewer ready", result["logs"]["viewer"])

    def test_supermimic_logs_supports_incremental_cursor(self):
        with tempfile.TemporaryDirectory() as temp:
            jobs_root = Path(temp) / "jobs"
            jobs = JobManager(jobs_root)
            output = Path(temp) / "output"
            training = output / "training"
            training.mkdir(parents=True)
            (training / "args.json").write_text(
                json.dumps({"iterations": 100}), encoding="utf-8"
            )
            (training / "train_log.csv").write_text(
                "iteration,wall_h\n20,0.02\n", encoding="utf-8"
            )
            first_line = "it 20 | training running\n"
            (output / "training.log").write_text(first_line, encoding="utf-8")
            (output / "pipeline.json").write_text(
                json.dumps(
                    {
                        "state": "running",
                        "stage": "train",
                        "progress": 0.65,
                    }
                ),
                encoding="utf-8",
            )
            job_id = "cursor_job"
            job_dir = jobs.root / job_id
            job_dir.mkdir()
            (job_dir / "job.json").write_text(
                json.dumps(
                    {
                        "job_id": job_id,
                        "kind": "supermimic_video_to_training",
                        "state": "running",
                        "output_root": str(output),
                        "log": str(job_dir / "job.log"),
                    }
                ),
                encoding="utf-8",
            )
            adapter = SuperMimicAdapter(
                IntegrationConfig(
                    g1_root=G1,
                    supermimic_root=SUPERMIMIC,
                    rax_root=RAX,
                    jobs_root=jobs_root,
                ),
                jobs,
            )
            first = adapter.logs({"job_id": job_id, "lines": 10})
            self.assertAlmostEqual(first["overall_progress"], 0.71, places=4)
            self.assertIn("20/100", first["summary"])
            self.assertIsNone(first["training_log_delta"])
            cursor = first["next_cursor"]
            self.assertEqual(cursor, (output / "training.log").stat().st_size)

            second_line = "it 21 | still running\n"
            with (output / "training.log").open("a", encoding="utf-8") as handle:
                handle.write(second_line)
            second = adapter.logs(
                {"job_id": job_id, "lines": 10, "cursor": cursor}
            )
            self.assertEqual(second["training_log_delta"], second_line)
            self.assertNotIn("it 20", second["training_log_delta"])
            self.assertGreater(second["next_cursor"], cursor)

            schema = next(
                item["function"]
                for item in integration_schemas()
                if item["function"]["name"] == "robot_supermimic_logs"
            )
            self.assertIn("cursor", schema["parameters"]["properties"])

    def test_supermimic_logs_reads_standalone_train_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            jobs_root = Path(temp) / "jobs"
            jobs = JobManager(jobs_root)
            output = Path(temp) / "output"
            output.mkdir()
            (output / "args.json").write_text(
                json.dumps({"iterations": 50}), encoding="utf-8"
            )
            (output / "train_log.csv").write_text(
                "iteration,wall_h,it_per_s\n25,0.01,10.0\n", encoding="utf-8"
            )
            job_id = "flat_train"
            job_dir = jobs.root / job_id
            job_dir.mkdir()
            (job_dir / "job.log").write_text(
                "it 25 | training running\n", encoding="utf-8"
            )
            (job_dir / "job.json").write_text(
                json.dumps(
                    {
                        "job_id": job_id,
                        "kind": "supermimic_train",
                        "state": "running",
                        "output_root": str(output),
                        "log": str(job_dir / "job.log"),
                    }
                ),
                encoding="utf-8",
            )
            adapter = SuperMimicAdapter(
                IntegrationConfig(
                    g1_root=G1,
                    supermimic_root=SUPERMIMIC,
                    rax_root=RAX,
                    jobs_root=jobs_root,
                ),
                jobs,
            )
            result = adapter.logs({"job_id": job_id, "lines": 10})
            self.assertEqual(result["training_progress"]["iteration"], 25)
            self.assertEqual(result["training_progress"]["iterations"], 50)
            self.assertAlmostEqual(result["overall_progress"], 0.5)
            self.assertEqual(result["stdout_source"], "job.log")
            self.assertIn("it 25", result["logs"]["training_stdout"])

    def test_reference_conversion_produces_supermimic_csv(self):
        source = G1 / "motions" / "fullbody" / "reference.npz"
        self.assertTrue(source.is_file())
        script = SUPERMIMIC / "scripts" / "learn_video.py"
        spec = importlib.util.spec_from_file_location(
            "supermimic_learn_video", script
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "motion.csv"
            result = module.convert_reference(source, output)
            rows = np.loadtxt(output, delimiter=",", ndmin=2)
            self.assertEqual(result["fps"], 30.0)
            self.assertEqual(rows.shape[1], 36)
            self.assertGreaterEqual(rows.shape[0], 2)
            self.assertTrue(np.isfinite(rows).all())

    def test_supermimic_progress_is_written_and_printed(self):
        script = SUPERMIMIC / "scripts" / "learn_video.py"
        spec = importlib.util.spec_from_file_location(
            "supermimic_progress", script
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                progress = module.Progress(
                    Path(temp) / "pipeline.json",
                    {
                        "state": "running",
                        "training": {"iterations": 10},
                    },
                )
                progress.update(
                    "train",
                    "训练运行中",
                    progress=0.65,
                    training_metrics={
                        "iteration": "4",
                        "wall_h": "0.01",
                    },
                )
            text = output.getvalue()
            self.assertIn("[SuperMimic][train]", text)
            self.assertIn("77.0%", text)
            self.assertIn("iteration=4", text)
            value = json.loads(
                (Path(temp) / "pipeline.json").read_text(encoding="utf-8")
            )
            self.assertEqual(value["stage"], "train")
            self.assertAlmostEqual(value["training_fraction"], 0.4)
            self.assertAlmostEqual(value["progress"], 0.77)


if __name__ == "__main__":
    unittest.main()
