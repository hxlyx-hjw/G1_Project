"""Runtime paths and configurable integration defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser().resolve() if value else default.resolve()


@dataclass(frozen=True)
class IntegrationConfig:
    """Paths used by the external adapters.

    All paths are configurable for a moved installation, but the defaults match
    the current TEST_RECORD layout.
    """

    g1_root: Path
    supermimic_root: Path
    rax_root: Path
    wsl_distro: str = "Ubuntu-22.04"
    supermimic_venv: str = "/mnt/d/G1_Envs/SuperMimic"
    jobs_root: Path = PACKAGE_ROOT / "runtime" / "jobs"

    @classmethod
    def from_env(cls) -> "IntegrationConfig":
        default_g1 = Path(
            r"E:\TEST_RECORD\G1_Project\G1\G1_LLM_Robot\G1_LLM_Robot"
        )
        default_supermimic = Path(r"E:\TEST_RECORD\SuperMimic")
        default_rax = Path(r"E:\TEST_RECORD\RAX")
        return cls(
            g1_root=_path_from_env("G1_DEEPSEEK_G1_ROOT", default_g1),
            supermimic_root=_path_from_env(
                "G1_DEEPSEEK_SUPERMIMIC_ROOT", default_supermimic
            ),
            rax_root=_path_from_env("G1_DEEPSEEK_RAX_ROOT", default_rax),
            wsl_distro=os.environ.get(
                "G1_DEEPSEEK_WSL_DISTRO", "Ubuntu-22.04"
            ).strip(),
            supermimic_venv=os.environ.get(
                "G1_DEEPSEEK_SUPERMIMIC_VENV",
                "/mnt/d/G1_Envs/SuperMimic",
            ).strip(),
            jobs_root=_path_from_env(
                "G1_DEEPSEEK_JOBS_ROOT",
                PACKAGE_ROOT / "runtime" / "jobs",
            ),
        )
