"""Detached completion recorder for G1 DeepSeek extension jobs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def main(argument: str) -> int:
    job_dir = Path(argument).resolve()
    record_path = job_dir / "job.json"
    request_path = job_dir / "request.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    request = json.loads(request_path.read_text(encoding="utf-8"))
    log_path = Path(record["log"])
    flags = 0
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                request["command"],
                cwd=request["cwd"],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
            record.update(
                state="running",
                worker_pid=process.pid,
                started_at=datetime.now().astimezone().isoformat(),
            )
            _atomic_json(record_path, record)
            returncode = process.wait()
        record.update(
            state="completed" if returncode == 0 else "failed",
            returncode=returncode,
            finished_at=datetime.now().astimezone().isoformat(),
        )
        _atomic_json(record_path, record)
        return 0
    except BaseException as exc:
        record.update(
            state="failed",
            error="%s: %s" % (type(exc).__name__, exc),
            finished_at=datetime.now().astimezone().isoformat(),
        )
        _atomic_json(record_path, record)
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: job_runner.py JOB_DIRECTORY")
    raise SystemExit(main(sys.argv[1]))
