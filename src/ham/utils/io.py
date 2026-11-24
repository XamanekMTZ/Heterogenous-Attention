from __future__ import annotations

import json
from pathlib import Path

import torch

from ham.config import ExperimentConfig, config_from_dict
from ham.models import build_model


def write_json(obj, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def run_dir_for(cfg: ExperimentConfig) -> Path:
    return Path(cfg.train.output_dir) / cfg.data.name / cfg.name / f"seed{cfg.seed}"


def load_run(run_dir: str | Path, checkpoint: str = "best.pt", map_location="cpu"):
    """Reconstruye ``(cfg, model)`` desde un directorio de run."""
    import yaml

    run_dir = Path(run_dir)
    with open(run_dir / "config.resolved.yaml") as f:
        cfg = config_from_dict(yaml.safe_load(f))
    model = build_model(cfg.model)
    model.load_state_dict(torch.load(run_dir / checkpoint, map_location=map_location, weights_only=True))
    return cfg, model
