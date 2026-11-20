"""Safe adapters that expose SuperMimic and RAX through G1 tools.

The adapters do not import either research package into the G1 process except
for reading result files. Long-running work is started in a child process and
reported through persistent job records.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .config import IntegrationConfig, PACKAGE_ROOT


class IntegrationError(RuntimeError):
    """A validation or adapter error that should be returned to the model."""


_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _tail(path: Path, limit: int = 6000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(text) > limit:
        return text[-limit:]
    return text


def _tail_lines(path: Path, lines: int, *, chars: int = 24000) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - max(chars, 1)))
            text = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    text = _ANSI_ESCAPE.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    values = text.splitlines()
    return "\n".join(values[-max(1, int(lines)):])


def _human_duration(seconds: float | None) -> str:
    if seconds is None:
        return "未知"
    total = int(max(0, float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return "%dh%02dm" % (hours, minutes)
    if minutes:
        return "%dm%02ds" % (minutes, secs)
    return "%ds" % secs


def _metric_value(value: str):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return int(number) if number.is_integer() and "." not in value else number


def _training_metrics(path: Path, limit: int = 8) -> tuple[list[dict], dict]:
    if not path.is_file():
        return [], {}
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
            rows = list(csv.reader(stream))
    except OSError:
        return [], {}
    if not rows:
        return [], {}
    headers = rows[0]
    history = []
    for row in rows[1:][-max(1, int(limit)):]:
        if len(row) != len(headers):
            continue
        history.append(
            {name: _metric_value(value) for name, value in zip(headers, row)}
        )
    return history, history[-1] if history else {}


def _read_log_since(
    path: Path, cursor: int, max_chars: int
) -> tuple[str, int, int, bool]:
    """Read a log incrementally.

    Returns ``(text, next_cursor, size, truncated)``. ``cursor`` is a byte
    offset returned by a previous call; pass ``0`` to receive the tail.
    """

    try:
        size = path.stat().st_size
    except OSError:
        return "", 0, 0, False
    if cursor < 0 or cursor > size:
        cursor = 0
    if cursor == 0:
        start = max(0, size - max_chars * 4)
    else:
        start = cursor
    try:
        with path.open("rb") as handle:
            handle.seek(start)
            data = handle.read(max(start, size) - start)
    except OSError:
        return "", cursor, size, False
    text = _ANSI_ESCAPE.sub("", data.decode("utf-8", errors="replace"))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    truncated = len(text) > max_chars or (cursor == 0 and start > 0)
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text, start + len(data), size, truncated


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if not drive:
        raise IntegrationError("无法把无盘符路径转换为 WSL 路径：%s" % resolved)
    parts = resolved.parts
    suffix = "/".join(parts[1:])
    return "/mnt/%s/%s" % (drive, suffix)


def _windows_path_from_wsl(value: str) -> Path | None:
    match = re.match(r"^/mnt/([A-Za-z])/(.*)$", value)
    if not match:
        return None
    return Path("%s:\\%s" % (match.group(1).upper(), match.group(2).replace("/", "\\")))


def resolve_under(
    root: Path,
    value: str | os.PathLike[str] | None,
    *,
    default: str | None = None,
    must_exist: bool = False,
) -> Path:
    """Resolve a user-supplied path and reject traversal outside *root*."""

    raw = str(value).strip() if value not in (None, "") else default
    if not raw:
        raise IntegrationError("缺少路径参数")
    converted = _windows_path_from_wsl(raw)
    candidate = converted if converted is not None else Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise IntegrationError(
            "路径必须位于 %s 内：%s" % (root_resolved, resolved)
        ) from exc
    if must_exist and not resolved.exists():
        raise IntegrationError("路径不存在：%s" % resolved)
    return resolved


def _safe_token(value: str, field: str = "name") -> str:
    text = str(value or "").strip()
    if not _SAFE_TOKEN.fullmatch(text):
        raise IntegrationError(
            "%s 只能包含字母、数字、点、下划线和连字符：%s" % (field, text)
        )
    return text


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


class JobManager:
    """Start child jobs and persist their state and bounded logs."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._processes: dict[str, subprocess.Popen[Any]] = {}

    def new_id(self, prefix: str) -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return "%s_%s_%s" % (prefix, stamp, uuid.uuid4().hex[:8])

    def start(
        self,
        *,
        job_id: str,
        kind: str,
        command: Sequence[str],
        cwd: Path,
        output_root: Path | None = None,
        environment: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        job_dir = self.root / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        log_path = job_dir / "job.log"
        record_path = job_dir / "job.json"
        request_path = job_dir / "request.json"
        flags = 0
        if os.name == "nt":
            flags = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )
        record: dict[str, Any] = {
            "job_id": job_id,
            "kind": kind,
            "state": "starting",
            "pid": None,
            "worker_pid": None,
            "started_at": None,
            "finished_at": None,
            "returncode": None,
            "cwd": str(cwd.resolve()),
            "command": list(command),
            "output_root": str(output_root.resolve()) if output_root else None,
            "log": str(log_path),
        }
        _atomic_json(record_path, record)
        _atomic_json(
            request_path,
            {
                "command": list(command),
                "cwd": str(cwd.resolve()),
            },
        )
        runner = PACKAGE_ROOT / "job_runner.py"
        try:
            process = subprocess.Popen(
                [sys.executable, str(runner), str(job_dir)],
                cwd=str(PACKAGE_ROOT),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=environment,
                creationflags=flags,
                start_new_session=os.name != "nt",
            )
        except Exception as exc:
            record.update(
                state="failed",
                error="%s: %s" % (type(exc).__name__, exc),
                finished_at=datetime.now().astimezone().isoformat(),
            )
            _atomic_json(record_path, record)
            raise
        record.update(
            state="running",
            pid=process.pid,
            started_at=datetime.now().astimezone().isoformat(),
        )
        _atomic_json(record_path, record)
        with self._lock:
            self._processes[job_id] = process
        return record

    def status(self, job_id: str, *, tail_chars: int = 6000) -> dict[str, Any]:
        job_id = _safe_token(job_id, "job_id")
        job_dir = self.root / job_id
        record_path = job_dir / "job.json"
        if not record_path.is_file():
            raise IntegrationError("找不到任务：%s" % job_id)
        record = json.loads(record_path.read_text(encoding="utf-8"))
        with self._lock:
            process = self._processes.get(job_id)
            if process is not None and process.poll() is not None:
                self._processes.pop(job_id, None)
        log_path = Path(record.get("log", job_dir / "job.log"))
        record["log_tail"] = _tail(log_path, max(0, int(tail_chars)))
        record["success"] = record.get("state") == "completed"
        if record.get("state") == "running":
            record["success"] = None
        return record


class SuperMimicAdapter:
    commands = (
        "robot_supermimic_status",
        "robot_supermimic_learn_video",
        "robot_supermimic_logs",
        "robot_supermimic_evaluate",
        "robot_supermimic_train",
        "robot_supermimic_render",
        "robot_supermimic_job_status",
    )

    def __init__(self, config: IntegrationConfig, jobs: JobManager):
        self.config = config
        self.jobs = jobs
        self.root = config.supermimic_root

    def _wsl_command(self, script: Path, *args: str) -> list[str]:
        return [
            "wsl.exe",
            "-d",
            self.config.wsl_distro,
            "--exec",
            "env",
            "G1_DEEPSEEK_SUPERMIMIC_VENV=" + self.config.supermimic_venv,
            "SUPERMIMIC_EXPERIMENT_ROOT=" + _wsl_path(self.root / "runs"),
            "bash",
            _wsl_path(script),
            *args,
        ]

    def _wsl_available(self) -> tuple[bool, str]:
        python = (self.config.supermimic_venv.rstrip("/") + "/bin/python")
        command = [
            "wsl.exe",
            "-d",
            self.config.wsl_distro,
            "--exec",
            "sh",
            "-c",
            'test -x "$1" && test -f "$2" && test -f "$3"',
            "g1-deepseek-check",
            python,
            _wsl_path(
                self.root / "upstream/checkpoints/g1_29dof/policy_098000.pt"
            ),
            _wsl_path(self.root / "scripts/run_supermimic_eval.sh"),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
                creationflags=(
                    getattr(subprocess, "CREATE_NO_WINDOW", 0)
                    if os.name == "nt"
                    else 0
                ),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, str(exc)
        return result.returncode == 0, (result.stderr or result.stdout).strip()

    def status(self) -> dict[str, Any]:
        available, error = self._wsl_available()
        summary_path = self.root / "results" / "supermimic_smoke.json"
        summary: dict[str, Any] | None = None
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                summary = {"error": "结果 JSON 无法解析"}
        return {
            "success": available,
            "backend": "SuperMimic",
            "available": available,
            "error": error or None,
            "root": str(self.root),
            "wsl_distro": self.config.wsl_distro,
            "venv": self.config.supermimic_venv,
            "checkpoint": str(
                self.root / "upstream/checkpoints/g1_29dof/policy_098000.pt"
            ),
            "latest_metrics": summary,
            "capabilities": [
                "single or dual-view video extraction and automatic training",
                "CPU MuJoCo evaluation with the bundled 98k G1 policy",
                "training or resumable training in the independent WSL environment",
                "rollout rendering to MP4",
            ],
            "note": "The published 98k checkpoint is an inference policy, not a resumable training checkpoint.",
        }

    def _resolve_video_input(self, value: str) -> Path:
        raw = str(value or "").strip()
        if not raw:
            raise IntegrationError("缺少视频文件或双视角目录")
        converted = _windows_path_from_wsl(raw)
        candidate = converted if converted is not None else Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self.config.g1_root / candidate
        candidate = candidate.resolve()
        allowed = [
            self.config.g1_root.resolve(),
            self.root.resolve(),
            Path(r"E:\TEST_RECORD").resolve(),
        ]
        dataset_root = Path(r"D:\G1_Datasets")
        if dataset_root.is_dir():
            allowed.append(dataset_root.resolve())
        if not any(
            candidate == root or candidate.is_relative_to(root) for root in allowed
        ):
            raise IntegrationError(
                "视频输入必须位于 G1 数据根或 SuperMimic 目录内：%s"
                % candidate
            )
        if not candidate.exists():
            raise IntegrationError("视频输入不存在：%s" % candidate)
        return candidate

    def learn_video(self, args: dict[str, Any]) -> dict[str, Any]:
        """Extract one or two views, convert to SuperMimic CSV and train."""

        source = self._resolve_video_input(str(args.get("input", "")))
        view_mode = str(args.get("view_mode", "auto")).lower()
        if view_mode not in {"auto", "single", "multi"}:
            raise IntegrationError("view_mode 只能是 auto、single 或 multi")
        quality = str(args.get("quality", "full")).lower()
        if quality not in {"lite", "full", "heavy"}:
            raise IntegrationError("quality 只能是 lite、full 或 heavy")
        pose_backend = str(args.get("pose_backend", "auto")).lower()
        multi_backend = str(args.get("multiview_backend", "auto")).lower()
        if pose_backend not in {"auto", "mediapipe", "smplx"}:
            raise IntegrationError("pose_backend 非法")
        if multi_backend not in {"auto", "dlt", "smplx"}:
            raise IntegrationError("multiview_backend 非法")
        fps = float(args.get("fps", 30.0))
        max_seconds = float(args.get("max_seconds", 60.0))
        iterations = int(args.get("iterations", 3))
        num_envs = int(args.get("num_envs", 2))
        mode = str(args.get("mode", "smoke")).lower()
        backend = str(args.get("backend", "mujoco")).lower()
        device = str(args.get("device", "cpu")).lower()
        root_mode = str(args.get("root_mode", "contact_grounded")).lower()
        min_visibility = float(args.get("min_visibility", 0.5))
        smplx_gender = str(args.get("smplx_gender", "neutral")).lower()
        smplx_device = str(args.get("smplx_device", "auto")).lower()
        if not 1 <= fps <= 60:
            raise IntegrationError("fps 必须在 1~60")
        if not 0 < max_seconds <= 180:
            raise IntegrationError("max_seconds 必须在 (0,180]")
        if not 1 <= iterations <= 320_000:
            raise IntegrationError("iterations 必须在 1~320000")
        if not 1 <= num_envs <= 16:
            raise IntegrationError("num_envs 必须在 1~16")
        if mode not in {"smoke", "paper"}:
            raise IntegrationError("mode 只能是 smoke 或 paper")
        if backend not in {"mujoco", "mjlab"}:
            raise IntegrationError("backend 只能是 mujoco 或 mjlab")
        if device != "cpu" and device != "cuda" and not device.startswith("cuda:"):
            raise IntegrationError("device 只能是 cpu、cuda 或 cuda:N")
        if backend == "mjlab" and device == "cpu":
            raise IntegrationError("mjlab 训练需要 CUDA")
        if root_mode not in {"in_place", "contact_grounded", "auto"}:
            raise IntegrationError("root_mode 只能是 in_place、contact_grounded 或 auto")
        if not 0.0 <= min_visibility <= 1.0:
            raise IntegrationError("min_visibility 必须在 0~1")
        if smplx_gender not in {"neutral", "male", "female"}:
            raise IntegrationError("smplx_gender 只能是 neutral、male 或 female")
        if (
            smplx_device != "auto"
            and smplx_device != "cpu"
            and smplx_device != "cuda"
            and not smplx_device.startswith("cuda:")
        ):
            raise IntegrationError("smplx_device 只能是 auto、cpu、cuda 或 cuda:N")
        dual_only = ("calibration", "list_file", "pair")
        if source.is_file() and any(name in args for name in dual_only):
            raise IntegrationError("双视角/相机标定参数不能用于单个视频文件")
        if view_mode == "single" and any(name in args for name in dual_only):
            raise IntegrationError("单视角模式不能设置双视角或相机标定参数")

        smplx_model_dir = None
        if args.get("smplx_model_dir"):
            smplx_model_dir = self._resolve_video_input(
                str(args["smplx_model_dir"])
            )

        job_id = self.jobs.new_id("sm-video")
        output = self.root / "runs" / "g1_deepseek" / job_id
        script = self.root / "scripts" / "learn_video.py"
        command = [
            sys.executable,
            str(script),
            "--input",
            str(source),
            "--output-root",
            str(output),
            "--view-mode",
            view_mode,
            "--fps",
            str(fps),
            "--max-seconds",
            str(max_seconds),
            "--quality",
            quality,
            "--pose-backend",
            pose_backend,
            "--multiview-backend",
            multi_backend,
            "--root-mode",
            root_mode,
            "--min-visibility",
            str(min_visibility),
            "--iterations",
            str(iterations),
            "--num-envs",
            str(num_envs),
            "--mode",
            mode,
            "--backend",
            backend,
            "--device",
            device,
        ]
        if args.get("list_file"):
            list_file = Path(str(args["list_file"])).expanduser()
            if not list_file.is_absolute():
                list_file = source / list_file
            list_file = list_file.resolve()
            if source.is_dir() and not list_file.is_relative_to(source):
                raise IntegrationError("list_file 必须位于输入目录内")
            if not list_file.is_file():
                raise IntegrationError("list_file 不存在：%s" % list_file)
            command.extend(["--list-file", str(list_file)])
        if args.get("pair") not in (None, ""):
            command.extend(["--pair", str(args["pair"])])
        if smplx_model_dir is not None:
            command.extend(["--smplx-model-dir", str(smplx_model_dir)])
        if args.get("smplx_gender"):
            command.extend(["--smplx-gender", smplx_gender])
        if args.get("smplx_device"):
            command.extend(["--smplx-device", smplx_device])
        if args.get("calibration"):
            calibration = self._resolve_video_input(str(args["calibration"]))
            command.extend(["--calibration", str(calibration)])
        for name in (
            "fov",
            "baseline",
            "azimuth_b",
            "offset_a",
            "offset_b",
            "min_valid_joint_fraction",
            "max_reprojection",
            "target_leg_length",
        ):
            if name in args:
                command.extend(
                    ["--" + name.replace("_", "-"), str(args[name])]
                )
        if args.get("auto_sync") is False:
            command.append("--no-auto-sync")
        if args.get("viewer", True) is False:
            command.append("--no-viewer")
        if args.get("keep_viewer_after_training", True) is False:
            command.append("--viewer-no-hold")
        environment = os.environ.copy()
        environment["G1_DEEPSEEK_G1_ROOT"] = str(self.config.g1_root)
        environment["G1_DEEPSEEK_WSL_DISTRO"] = self.config.wsl_distro
        environment["G1_DEEPSEEK_SUPERMIMIC_VENV"] = self.config.supermimic_venv
        environment["PYTHONUTF8"] = "1"
        job = self.jobs.start(
            job_id=job_id,
            kind="supermimic_video_to_training",
            command=command,
            cwd=self.root,
            output_root=output,
            environment=environment,
        )
        return {
            "success": True,
            "started": True,
            "job_id": job["job_id"],
            "input": str(source),
            "output_root": str(output),
            "pipeline": [
                "single/dual-view pose extraction",
                "G1 full-body retargeting",
                "SuperMimic 30 Hz CSV conversion",
                "automatic policy training",
                "live MuJoCo reference viewer",
            ],
            "training": {
                "mode": mode,
                "backend": backend,
                "device": device,
                "iterations": iterations,
                "num_envs": num_envs,
            },
            "viewer": {
                "enabled": bool(args.get("viewer", True)),
                "keep_open_after_training": bool(
                    args.get("keep_viewer_after_training", True)
                ),
            },
            "note": self._training_note(mode, iterations, device),
            "next_step": "调用 robot_supermimic_job_status 检查阶段、进度和训练指标。",
        }

    @staticmethod
    def _training_note(mode: str, iterations: int, device: str) -> str:
        if mode == "smoke" or iterations < 1000:
            return (
                "当前配置只验证“提取→重定向→CSV→训练”链路（mode=%s，"
                "iterations=%d），迭代数很少，不能得到可用策略。真实训练请传 "
                "mode=paper 并提高 iterations，或用 robot_supermimic_train 指向本次 "
                "output_root/motion 继续训练。"
                % (mode, iterations)
            )
        if device == "cpu":
            return (
                "paper 模式按论文网络配置训练，CPU 上耗时会很长；"
                "有条件时建议指定 device=cuda 后重新启动。"
            )
        return "paper 模式按论文网络配置训练，请通过 job_status 跟踪迭代指标。"

    def evaluate(self, args: dict[str, Any]) -> dict[str, Any]:
        motion_dir = resolve_under(
            self.root,
            args.get("motion_dir"),
            default="data/eval_subset",
            must_exist=True,
        )
        job_id = self.jobs.new_id("sm-eval")
        run_dir = self.root / "runs" / "g1_deepseek" / job_id
        run_dir.mkdir(parents=True, exist_ok=False)
        record = run_dir / "rollout.npz" if args.get("record", True) else None
        script = self.root / "scripts" / "run_supermimic_eval.sh"
        command_args = [_wsl_path(motion_dir), _wsl_path(run_dir)]
        if record is not None:
            command_args.append(_wsl_path(record))
        job = self.jobs.start(
            job_id=job_id,
            kind="supermimic_evaluate",
            command=self._wsl_command(script, *command_args),
            cwd=PACKAGE_ROOT,
            output_root=run_dir,
        )
        return {
            "success": True,
            "started": True,
            "job_id": job["job_id"],
            "output_root": str(run_dir),
            "motion_dir": str(motion_dir),
            "record": str(record) if record else None,
            "next_step": (
                "调用 robot_supermimic_job_status 查询进度；完成后可调用 "
                "robot_supermimic_render，clip 名称取 motion 文件名且不含 .csv。"
            ),
        }

    def train(self, args: dict[str, Any]) -> dict[str, Any]:
        motion_dir = resolve_under(
            self.root,
            args.get("motion_dir"),
            default="data/lafan1_g1",
            must_exist=True,
        )
        mode = str(args.get("mode", "smoke")).lower()
        if mode not in {"smoke", "paper"}:
            raise IntegrationError("mode 只能是 smoke 或 paper")
        iterations = int(args.get("iterations", 3 if mode == "smoke" else 1000))
        num_envs = int(args.get("num_envs", 2 if mode == "smoke" else 4))
        if not 1 <= iterations <= 320_000:
            raise IntegrationError("iterations 必须在 1~320000")
        if not 1 <= num_envs <= 16:
            raise IntegrationError("num_envs 必须在 1~16")
        job_id = self.jobs.new_id("sm-train")
        run_dir = self.root / "runs" / "g1_deepseek" / job_id
        run_dir.mkdir(parents=True, exist_ok=False)
        script = (self.root / "scripts" / "supermimic_train.sh").resolve()
        command = self._wsl_command(
            script,
            _wsl_path(motion_dir),
            _wsl_path(run_dir),
            str(iterations),
            str(num_envs),
            mode,
        )
        job = self.jobs.start(
            job_id=job_id,
            kind="supermimic_train",
            command=command,
            cwd=PACKAGE_ROOT,
            output_root=run_dir,
        )
        return {
            "success": True,
            "started": True,
            "job_id": job["job_id"],
            "output_root": str(run_dir),
            "motion_dir": str(motion_dir),
            "iterations": iterations,
            "num_envs": num_envs,
            "mode": mode,
            "note": (
                "smoke 使用已验证的小型 CPU 参数；paper 使用论文网络配置，"
                "完整 320k 训练耗时很长。"
            ),
        }

    def _run_from_args(self, args: dict[str, Any]) -> tuple[str, Path]:
        job_id = str(args.get("job_id", "")).strip()
        if job_id:
            job = self.jobs.status(job_id, tail_chars=0)
            if not job.get("output_root"):
                raise IntegrationError("任务没有记录输出目录：%s" % job_id)
            return job_id, Path(job["output_root"]).resolve()
        run_dir = resolve_under(
            self.root,
            args.get("run_dir"),
            default="runs/eval_smoke",
            must_exist=True,
        )
        return _safe_token(run_dir.name, "run_dir.name"), run_dir

    def render(self, args: dict[str, Any]) -> dict[str, Any]:
        _, run_dir = self._run_from_args(args)
        record = run_dir / "rollout.npz"
        if not record.is_file():
            raise IntegrationError("运行目录缺少 rollout.npz：%s" % run_dir)
        clip = _safe_token(str(args.get("clip", "")), "clip")
        gl = str(args.get("gl", "egl")).lower()
        if gl not in {"egl", "osmesa", "glfw"}:
            raise IntegrationError("gl 只能是 egl、osmesa 或 glfw")
        render_job_id = self.jobs.new_id("sm-render")
        output = run_dir / ("g1_deepseek_%s_%s.mp4" % (render_job_id[-8:], clip))
        script = self.root / "scripts" / "render_supermimic.sh"
        job = self.jobs.start(
            job_id=render_job_id,
            kind="supermimic_render",
            command=self._wsl_command(
                script,
                _wsl_path(record),
                clip,
                _wsl_path(output),
                gl,
            ),
            cwd=PACKAGE_ROOT,
            output_root=run_dir,
        )
        return {
            "success": True,
            "started": True,
            "job_id": job["job_id"],
            "source_run": str(run_dir),
            "clip": clip,
            "output": str(output),
        }

    def job_status(self, args: dict[str, Any]) -> dict[str, Any]:
        result = self.jobs.status(str(args.get("job_id", "")))
        output_root = result.get("output_root")
        if output_root:
            pipeline_path = Path(output_root) / "pipeline.json"
            if pipeline_path.is_file():
                try:
                    pipeline = json.loads(
                        pipeline_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    pipeline = {"parse_error": str(exc)}
                result["pipeline"] = pipeline
                if pipeline.get("training_completed"):
                    result["training_success"] = True
                if (
                    pipeline.get("stage") == "viewer_open"
                    and pipeline.get("training_completed")
                ):
                    result["state"] = "viewer_open"
                    result["success"] = True
                for key in (
                    "stage",
                    "message",
                    "progress",
                    "training_metrics",
                    "actual_mode",
                    "quality_warning",
                    "retarget_rejected_fraction",
                    "multiview",
                ):
                    if key in pipeline:
                        result[key] = pipeline[key]
                if pipeline.get("state") == "failed":
                    result["success"] = False
            viewer_path = Path(output_root) / "viewer.json"
            if viewer_path.is_file():
                try:
                    result["viewer"] = json.loads(
                        viewer_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    result["viewer"] = {"parse_error": str(exc)}
        return result

    def logs(self, args: dict[str, Any]) -> dict[str, Any]:
        waiting = float(args.get("wait_seconds", 0.0) or 0.0)
        if not 0 <= waiting <= 5:
            raise IntegrationError("wait_seconds 必须在 0~5")
        if waiting:
            time.sleep(waiting)
        lines = int(args.get("lines", 80))
        if not 10 <= lines <= 500:
            raise IntegrationError("lines 必须在 10~500")
        history_limit = int(args.get("metrics_limit", 8))
        if not 1 <= history_limit <= 100:
            raise IntegrationError("metrics_limit 必须在 1~100")
        cursor = int(args.get("cursor", 0) or 0)
        if cursor < 0:
            raise IntegrationError("cursor 必须是非负整数")
        job = self.job_status({"job_id": str(args.get("job_id", ""))})
        output_root = Path(str(job.get("output_root", "")))
        if not output_root.is_dir():
            raise IntegrationError("任务输出目录不存在：%s" % output_root)
        training_root = output_root / "training"
        # learn_video 输出到 <root>/training/，独立 robot_supermimic_train
        # 直接输出到 <root>/；两种布局都支持。
        metrics_path = training_root / "train_log.csv"
        if not metrics_path.is_file():
            metrics_path = output_root / "train_log.csv"
        history, latest = _training_metrics(metrics_path, history_limit)
        training_args = {}
        args_path = training_root / "args.json"
        if not args_path.is_file():
            args_path = output_root / "args.json"
        if args_path.is_file():
            try:
                training_args = json.loads(args_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                training_args = {}
        target = int(training_args.get("iterations", 0) or 0)
        iteration = 0
        try:
            iteration = int(float(latest.get("iteration", 0)))
        except (TypeError, ValueError):
            pass
        fraction = (
            max(0.0, min(1.0, iteration / target)) if target > 0 else None
        )
        elapsed_seconds = None
        eta_seconds = None
        try:
            elapsed_seconds = float(latest.get("wall_h", 0.0)) * 3600.0
            if fraction is not None and iteration > 0 and elapsed_seconds > 0:
                eta_seconds = elapsed_seconds * (1.0 - fraction) / fraction
        except (TypeError, ValueError):
            pass
        stdout_path = output_root / "training.log"
        stdout_source = "training.log"
        if not stdout_path.is_file():
            stdout_path = Path(str(job.get("log", "") or ""))
            stdout_source = "job.log"
        paths = {
            "job": Path(str(job.get("log", "") or "")),
            "training_stdout": stdout_path,
            "training_metrics": metrics_path,
            "viewer": output_root / "viewer.log",
        }
        tails = {
            "job": _tail_lines(paths["job"], lines),
            "training_stdout": _tail_lines(paths["training_stdout"], lines),
            "viewer": _tail_lines(paths["viewer"], lines),
        }
        delta_text, next_cursor, log_size, delta_truncated = _read_log_since(
            paths["training_stdout"],
            cursor,
            max(1000, min(24000, lines * 400)),
        )
        pipeline = job.get("pipeline", {})
        stage = job.get("stage") or pipeline.get("stage")
        progress = job.get("progress", pipeline.get("progress"))
        overall_progress = progress
        if fraction is not None:
            if stage in {"train", "train_start"} and pipeline:
                overall_progress = round(0.65 + 0.30 * fraction, 4)
            elif progress is None:
                overall_progress = round(fraction, 4)
        if fraction is None:
            summary = "训练尚未产生 train_log.csv 指标"
        else:
            summary = "训练迭代 %d/%d（%.2f%%），预计剩余 %s" % (
                iteration,
                target,
                fraction * 100.0,
                _human_duration(eta_seconds),
            )
        return {
            "success": job.get("success"),
            "job_id": job.get("job_id"),
            "state": job.get("state"),
            "stage": stage,
            "message": job.get("message") or pipeline.get("message"),
            "progress": progress,
            "overall_progress": overall_progress,
            "summary": summary,
            "training_progress": {
                "iteration": iteration,
                "iterations": target or None,
                "fraction": fraction,
                "elapsed_seconds": elapsed_seconds,
                "eta_seconds": eta_seconds,
                "it_per_s": latest.get("it_per_s"),
            },
            "latest_metrics": latest,
            "recent_metrics": history,
            "viewer": job.get("viewer"),
            "quality_warning": pipeline.get("quality_warning"),
            "logs": tails,
            "stdout_source": stdout_source,
            "training_log_delta": delta_text if cursor else None,
            "training_log_delta_truncated": delta_truncated if cursor else False,
            "next_cursor": next_cursor,
            "training_log_size": log_size,
            "log_files": [
                {
                    "name": name,
                    "path": str(path),
                    "size": path.stat().st_size,
                    "modified": datetime.fromtimestamp(
                        path.stat().st_mtime
                    ).astimezone().isoformat(),
                }
                for name, path in paths.items()
                if path.is_file()
            ],
            "snapshot_at": datetime.now().astimezone().isoformat(),
            "note": (
                "结构化指标以 train_log.csv 为准；训练结束且窗口保持时，"
                "state 可以是 viewer_open。把本次 next_cursor 作为 cursor 再次调用，"
                "只会返回新增的训练日志，适合实时轮询。"
            ),
        }

    def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "robot_supermimic_status":
            return self.status()
        if name == "robot_supermimic_learn_video":
            return self.learn_video(args)
        if name == "robot_supermimic_logs":
            return self.logs(args)
        if name == "robot_supermimic_evaluate":
            return self.evaluate(args)
        if name == "robot_supermimic_train":
            return self.train(args)
        if name == "robot_supermimic_render":
            return self.render(args)
        if name == "robot_supermimic_job_status":
            return self.job_status(args)
        raise IntegrationError("未知 SuperMimic 工具：%s" % name)


class RAXAdapter:
    commands = (
        "robot_rax_status",
        "robot_rax_run",
        "robot_rax_report",
        "robot_rax_list",
        "robot_rax_job_status",
    )

    def __init__(
        self,
        config: IntegrationConfig,
        jobs: JobManager,
        python_executable: str | None = None,
    ):
        self.config = config
        self.jobs = jobs
        self.root = config.rax_root
        self.python = python_executable or os.sys.executable

    def status(self) -> dict[str, Any]:
        result_path = self.root / "results" / "rax_scaling.json"
        result: dict[str, Any] | None = None
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                result = {"error": "结果 JSON 无法解析"}
        backend = self.root / "rax_min" / "backend" / "g1_reach.py"
        available = (
            backend.is_file()
            and self.config.g1_root.is_dir()
            and (self.root / "rax_min" / "cli.py").is_file()
        )
        return {
            "success": available,
            "backend": "RAX",
            "available": available,
            "root": str(self.root),
            "g1_root": str(self.config.g1_root),
            "python": self.python,
            "latest_scaling": result,
            "capabilities": [
                "contract-bound exploration",
                "controller compile, freeze and rehearsal",
                "one-shot formal held-out evaluation",
                "operator-only report and persisted evidence",
            ],
            "note": "The current reproduction is a minimal local harness, not the full official multi-branch implementation.",
        }

    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        budget = int(args.get("budget", 8))
        seed = int(args.get("seed", 0))
        timeout = float(args.get("timeout", 180.0))
        isolation = str(args.get("isolation", "process")).lower()
        if not 1 <= budget <= 1024:
            raise IntegrationError("budget 必须在 1~1024")
        if not -2_147_483_648 <= seed <= 2_147_483_647:
            raise IntegrationError("seed 超出 32 位整数范围")
        if not 10 <= timeout <= 1800:
            raise IntegrationError("timeout 必须在 10~1800 秒")
        if isolation not in {"process", "bwrap"}:
            raise IntegrationError("isolation 只能是 process 或 bwrap")
        if isolation == "bwrap":
            raise IntegrationError("当前 Windows 环境只验证了 process 隔离")
        if not self.config.g1_root.is_dir():
            raise IntegrationError("G1 源码目录不存在：%s" % self.config.g1_root)
        job_id = self.jobs.new_id("rax")
        run_name = "g1_deepseek_" + job_id
        runs_root = self.root / "rax_min" / "runs"
        run_root = runs_root / run_name
        environment = os.environ.copy()
        old_path = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            str(self.root)
            if not old_path
            else str(self.root) + os.pathsep + old_path
        )
        environment.setdefault("MUJOCO_GL", "disable")
        command = [
            self.python,
            "-m",
            "rax_min",
            "run",
            "--budget",
            str(budget),
            "--seed",
            str(seed),
            "--parent",
            str(runs_root),
            "--name",
            run_name,
            "--isolation",
            isolation,
            "--timeout",
            str(timeout),
            "--g1-root",
            str(self.config.g1_root),
        ]
        job = self.jobs.start(
            job_id=job_id,
            kind="rax_run",
            command=command,
            cwd=self.root,
            output_root=run_root,
            environment=environment,
        )
        return {
            "success": True,
            "started": True,
            "job_id": job["job_id"],
            "run": run_name,
            "run_root": str(run_root),
            "budget": budget,
            "seed": seed,
            "timeout_seconds": timeout,
            "next_step": (
                "调用 robot_rax_job_status 查询进度；完成后调用 "
                "robot_rax_report，参数 run 可传 job_id。"
            ),
        }

    def _resolve_run(self, value: str | None) -> Path:
        raw = str(value or "").strip()
        if not raw:
            raise IntegrationError("缺少 run 或 job_id")
        if raw.startswith("rax_"):
            try:
                job = self.jobs.status(raw, tail_chars=0)
            except IntegrationError:
                job = None
            if job is not None:
                if not job.get("output_root"):
                    raise IntegrationError("任务没有记录运行目录：%s" % raw)
                return Path(job["output_root"]).resolve()
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.root / "rax_min" / "runs" / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to((self.root / "rax_min" / "runs").resolve())
        except ValueError as exc:
            raise IntegrationError("RAX run 必须位于项目 runs 目录内") from exc
        return resolved

    def report(self, args: dict[str, Any]) -> dict[str, Any]:
        run = self._resolve_run(args.get("run") or args.get("job_id"))
        report_path = run / "report.json"
        state_path = run / "state.json"
        if not report_path.is_file():
            state = (
                json.loads(state_path.read_text(encoding="utf-8"))
                if state_path.is_file()
                else None
            )
            return {
                "success": False,
                "state": "incomplete",
                "run_root": str(run),
                "current_state": state,
                "message": "正式报告尚未生成；请先检查任务状态或日志。",
            }
        return {
            "success": True,
            "run_root": str(run),
            "report_path": str(report_path),
            "report": json.loads(report_path.read_text(encoding="utf-8")),
        }

    def list_runs(self, args: dict[str, Any]) -> dict[str, Any]:
        limit = max(1, min(100, int(args.get("limit", 20))))
        runs_root = self.root / "rax_min" / "runs"
        rows: list[dict[str, Any]] = []
        if runs_root.is_dir():
            for path in sorted(
                (p for p in runs_root.iterdir() if p.is_dir()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ):
                state_path = path / "state.json"
                if not state_path.is_file():
                    continue
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    state = {}
                rows.append(
                    {
                        "run": path.name,
                        "phase": state.get("phase"),
                        "budget": state.get("budget"),
                        "seed": state.get("seed"),
                        "formal": state.get("formal"),
                        "report": str(path / "report.json")
                        if (path / "report.json").is_file()
                        else None,
                    }
                )
                if len(rows) >= limit:
                    break
        return {"success": True, "count": len(rows), "runs": rows}

    def job_status(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.jobs.status(str(args.get("job_id", "")))

    def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "robot_rax_status":
            return self.status()
        if name == "robot_rax_run":
            return self.run(args)
        if name == "robot_rax_report":
            return self.report(args)
        if name == "robot_rax_list":
            return self.list_runs(args)
        if name == "robot_rax_job_status":
            return self.job_status(args)
        raise IntegrationError("未知 RAX 工具：%s" % name)


def _schema(name: str, description: str, properties: dict, required=()) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(required),
            },
        },
    }


def integration_schemas() -> list[dict]:
    path_prop = {
        "type": "string",
        "description": "Windows 路径或项目内相对路径；必须位于对应项目目录内。",
    }
    job_prop = {
        "type": "string",
        "description": "由 *_run/evaluate/train/render 工具返回的 job_id。",
    }
    return [
        _schema(
            "robot_supermimic_status",
            "检查独立 SuperMimic 环境、98k G1 推理策略、最近评测指标与可用能力。开始 SuperMimic 工作前先调用。",
            {},
        ),
        _schema(
            "robot_supermimic_learn_video",
            "SuperMimic 视频到训练流水线：单视频或含 list 的双视角目录→2D/DLT/SMPL-X→G1全身重定向→30Hz动作→自动训练，并另开持续运行的MuJoCo参考渲染。训练结束窗口默认保持，直到用户关闭。",
            {
                "input": {
                    "type": "string",
                    "description": "单个视频文件，或包含 list/list.txt 与 a/b 视频的双视角目录。",
                },
                "view_mode": {
                    "type": "string",
                    "enum": ["auto", "single", "multi"],
                },
                "list_file": path_prop,
                "pair": {
                    "type": "string",
                    "description": "双视角目录中的 pair 名或从0开始的索引；省略选第一对。",
                },
                "fps": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 60,
                    "description": "提取采样率，默认30；转换后统一重采样到SuperMimic的30Hz。",
                },
                "max_seconds": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": 180,
                },
                "quality": {"type": "string", "enum": ["lite", "full", "heavy"]},
                "pose_backend": {
                    "type": "string",
                    "enum": ["auto", "mediapipe", "smplx"],
                },
                "multiview_backend": {
                    "type": "string",
                    "enum": ["auto", "dlt", "smplx"],
                },
                "smplx_model_dir": {
                    "type": "string",
                    "description": "pose_backend 或 multiview_backend 使用 smplx 时的模型目录；必须位于允许的项目目录内。",
                },
                "smplx_gender": {
                    "type": "string",
                    "enum": ["neutral", "male", "female"],
                },
                "smplx_device": {
                    "type": "string",
                    "description": "auto、cpu、cuda 或 cuda:N。",
                },
                "calibration": {
                    "type": "string",
                    "description": "可选 OpenCV 相机标定 JSON；双视角训练应优先提供。",
                },
                "fov": {"type": "number", "minimum": 5, "maximum": 170},
                "baseline": {"type": "number", "exclusiveMinimum": 0, "maximum": 20},
                "azimuth_b": {"type": "number", "minimum": -180, "maximum": 180},
                "offset_a": {"type": "number"},
                "offset_b": {"type": "number"},
                "auto_sync": {"type": "boolean"},
                "min_valid_joint_fraction": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "max_reprojection": {"type": "number", "exclusiveMinimum": 0},
                "target_leg_length": {
                    "type": "number",
                    "minimum": 0.2,
                    "maximum": 2,
                },
                "root_mode": {
                    "type": "string",
                    "enum": ["in_place", "contact_grounded", "auto"],
                },
                "min_visibility": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "iterations": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 320000,
                    "description": "自动训练迭代数；smoke默认为3。",
                },
                "num_envs": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 16,
                },
                "mode": {"type": "string", "enum": ["smoke", "paper"]},
                "backend": {"type": "string", "enum": ["mujoco", "mjlab"]},
                "device": {
                    "type": "string",
                    "description": "cpu、cuda 或 cuda:N；mjlab必须使用CUDA。",
                },
                "viewer": {
                    "type": "boolean",
                    "description": "默认true，启动独立MuJoCo实时参考渲染。",
                },
                "keep_viewer_after_training": {
                    "type": "boolean",
                    "description": "默认true，训练结束后继续显示，直到用户关闭窗口。",
                },
            },
            ("input",),
        ),
        _schema(
            "robot_supermimic_evaluate",
            "在独立 WSL CPU MuJoCo 环境中异步评测 SuperMimic 的 98k G1 策略。默认使用两条 10 秒 LAFAN1 子集并保存轨迹。返回 job_id。",
            {
                "motion_dir": path_prop,
                "record": {
                    "type": "boolean",
                    "description": "是否保存 rollout.npz 供后续渲染，默认 true。",
                },
            },
        ),
        _schema(
            "robot_supermimic_train",
            "异步启动 SuperMimic 训练。smoke 使用仓库已验证的小型 CPU 参数；paper 使用论文网络配置，耗时和资源开销明显更高。每次写入新的隔离目录。",
            {
                "motion_dir": path_prop,
                "mode": {"type": "string", "enum": ["smoke", "paper"]},
                "iterations": {"type": "integer", "minimum": 1, "maximum": 320000},
                "num_envs": {"type": "integer", "minimum": 1, "maximum": 16},
            },
        ),
        _schema(
            "robot_supermimic_render",
            "从 SuperMimic 评测产生的 rollout.npz 异步渲染带参考虚影的 MP4。",
            {
                "job_id": {
                    "type": "string",
                    "description": "SuperMimic evaluate 返回的任务号；优先使用。",
                },
                "run_dir": {
                    "type": "string",
                    "description": "也可直接指定 SuperMimic 内的已有运行目录。",
                },
                "clip": {
                    "type": "string",
                    "description": "轨迹中的 clip 名，例如 walk1_subject1_10s。",
                },
                "gl": {"type": "string", "enum": ["egl", "osmesa", "glfw"]},
            },
            ("clip",),
        ),
        _schema(
            "robot_supermimic_job_status",
            "查询 SuperMimic 后台任务状态、退出码、输出目录和日志尾部。",
            {"job_id": job_prop},
            ("job_id",),
        ),
        _schema(
            "robot_supermimic_logs",
            "实时读取 SuperMimic 视频训练任务：返回流水线阶段、训练迭代百分比、耗时、预计剩余、最新及最近指标、训练 stdout、任务日志和查看器日志尾部。把上次返回的 next_cursor 作为 cursor 再次调用，只返回新增训练日志，适合持续轮询。",
            {
                "job_id": job_prop,
                "cursor": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "上次返回的 next_cursor（字节偏移）；省略或0表示取当前日志尾部。",
                },
                "lines": {
                    "type": "integer",
                    "minimum": 10,
                    "maximum": 500,
                    "description": "每份日志读取尾部行数，默认80。",
                },
                "metrics_limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "返回最近多少条结构化训练指标，默认8。",
                },
                "wait_seconds": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 5,
                    "description": "读取前最多等待5秒，便于短间隔轮询。",
                },
            },
            ("job_id",),
        ),
        _schema(
            "robot_rax_status",
            "检查独立 RAX harness、G1 只读后端和最近缩放结果。开始 RAX 工作前先调用。",
            {},
        ),
        _schema(
            "robot_rax_run",
            "异步运行一次完整 RAX 生命周期：探索、编译、SHA256 冻结、排练、一次提交和正式 held-out 评测。返回 job_id。",
            {
                "budget": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1024,
                    "description": "探索 episode 预算，默认 8。",
                },
                "seed": {"type": "integer"},
                "timeout": {
                    "type": "number",
                    "minimum": 10,
                    "maximum": 1800,
                    "description": "单个控制器进程超时秒数。",
                },
            },
        ),
        _schema(
            "robot_rax_report",
            "读取 RAX 已完成的正式报告。优先传 run 参数为 robot_rax_run 返回的 job_id；未完成时返回当前 phase。",
            {
                "run": {
                    "type": "string",
                    "description": "RAX run 名称、job_id 或 runs 内的绝对路径。",
                }
            },
        ),
        _schema(
            "robot_rax_list",
            "列出最近的 RAX 运行、预算、阶段和正式结果摘要。",
            {"limit": {"type": "integer", "minimum": 1, "maximum": 100}},
        ),
        _schema(
            "robot_rax_job_status",
            "查询 RAX 后台任务状态、退出码、运行目录和日志尾部。",
            {"job_id": job_prop},
            ("job_id",),
        ),
    ]


class DeepSeekIntegration:
    """Dispatch object shared by the reflected G1 chat bot tools."""

    PROMPT = (
        "\n本进程已通过外部扩展接入两个独立研究工具。涉及 SuperMimic 的评估、"
        "训练、渲染或状态时使用 robot_supermimic_*。用户给出单个动作视频或含 "
        "list 的双视角视频目录并要求学习时，优先使用 "
        "robot_supermimic_learn_video；它会先提取和重定向，再自动训练，并默认保持"
        "实时 MuJoCo 渲染窗口打开；查询实时训练日志和进度使用 "
        "robot_supermimic_logs。涉及 RAX 的探索/冻结/"
        "held-out 评测、报告或状态时使用 robot_rax_*。长时间任务会返回 job_id，"
        "必须随后调用对应 job_status 查询，不能把“已启动”说成“已完成”。"
        "工具只能写入各自项目的独立运行目录，不得改写 G1、SuperMimic 或 RAX "
        "源码。读取报告时以 report.json/evaluation.json 为事实来源。"
    )

    def __init__(
        self,
        config: IntegrationConfig | None = None,
        *,
        python_executable: str | None = None,
    ):
        self.config = config or IntegrationConfig.from_env()
        self.jobs = JobManager(self.config.jobs_root)
        self.supermimic = SuperMimicAdapter(self.config, self.jobs)
        self.rax = RAXAdapter(
            self.config,
            self.jobs,
            python_executable=python_executable,
        )

    def schemas(self) -> list[dict]:
        return integration_schemas()

    def names(self) -> set[str]:
        return set(SuperMimicAdapter.commands + RAXAdapter.commands)

    def dispatch(self, name: str, args: dict[str, Any] | None = None) -> str:
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise IntegrationError("工具参数必须是 JSON 对象")
        try:
            if name in SuperMimicAdapter.commands:
                value = self.supermimic.dispatch(name, args)
            elif name in RAXAdapter.commands:
                value = self.rax.dispatch(name, args)
            else:
                raise IntegrationError("未知集成工具：%s" % name)
            return _json(value)
        except Exception as exc:  # The model receives errors as tool results.
            return _json(
                {
                    "success": False,
                    "tool": name,
                    "error": "%s: %s" % (type(exc).__name__, exc),
                }
            )
