"""Registro del entorno de ejecución para cada run (para el apéndice de reproducibilidad)."""
from __future__ import annotations

import datetime
import platform
import socket
import subprocess
import sys
from pathlib import Path

import torch


def _git(*args: str, cwd: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def collect_provenance(repo_dir: str | Path | None = None) -> dict:
    repo_dir = Path(repo_dir or Path(__file__).resolve().parent)
    status = _git("status", "--porcelain", cwd=repo_dir)
    return {
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "command": " ".join(sys.argv),
        "hostname": socket.gethostname(),
        "git": {
            "commit": _git("rev-parse", "HEAD", cwd=repo_dir),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo_dir),
            "dirty": bool(status) if status is not None else None,
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
