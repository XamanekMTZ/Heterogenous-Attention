"""Evaluación en test de un run, o de un experimento de ``202506_Transformers``.

    uv run python -m ham.evaluate --run runs/cifar10/cifar10_ham_concat/seed0
    uv run python -m ham.evaluate --legacy ../202506_Transformers/experiments/exp_20251206_185442_heterogeneous_400e

En modo ``--legacy`` el tipo de atención se infiere de las claves del checkpoint
(los experimentos viejos no lo guardaban) y se usa la normalización original.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ham.config import AttentionConfig, DataConfig, ModelConfig
from ham.datasets import build_dataloaders
from ham.models import build_model
from ham.train import evaluate, resolve_device
from ham.utils.io import load_run, read_json

LEGACY_CHECKPOINTS = ("model_final.pt", "Checkpoints/model_final.pt")


def infer_legacy_branches(state_dict) -> list[str]:
    """202506_Transformers: heterogeneous = attn1 (dot) + attn2 (additive) + fusion."""
    keys = list(state_dict)
    if any(".fusion.weight" in k for k in keys):
        return ["dot", "additive"]
    if any(".attn.q_proj.weight" in k for k in keys):
        return ["additive"]
    return ["dot"]


def legacy_model_config(cfg: dict, branches: list[str]) -> ModelConfig:
    return ModelConfig(
        image_size=cfg["image_size"],
        patch_size=cfg["patch_size"],
        num_channels=cfg["num_channels"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg.get("num_layers", cfg.get("num_hidden_layers")),
        num_attention_heads=cfg["num_attention_heads"],
        intermediate_size=cfg["intermediate_size"],
        hidden_dropout_prob=cfg["hidden_dropout_prob"],
        attention_probs_dropout_prob=cfg["hidden_dropout_prob"],  # el código viejo usaba ésta en la atención
        qkv_bias=cfg["qkv_bias"],
        num_classes=cfg["num_classes"],
        attention=AttentionConfig(branches=branches, fusion="concat_linear"),
    )


def load_legacy(exp_dir: Path):
    ckpt = next((exp_dir / c for c in LEGACY_CHECKPOINTS if (exp_dir / c).exists()), None)
    if ckpt is None:
        raise FileNotFoundError(f"No se encontró {LEGACY_CHECKPOINTS} en {exp_dir}")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    branches = infer_legacy_branches(state)
    model_cfg = legacy_model_config(read_json(exp_dir / "config.json"), branches)
    model = build_model(model_cfg)
    model.load_state_dict(state, strict=True)
    train_cfg = read_json(exp_dir / "train_config.json") if (exp_dir / "train_config.json").exists() else {}
    dataset = train_cfg.get("dataset_name") or "cifar10"
    return model_cfg, model, dataset, ckpt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", type=Path, help="directorio de un run de ham.train")
    g.add_argument("--legacy", type=Path, help="directorio de experiments/ de 202506_Transformers")
    p.add_argument("--checkpoint", default="best.pt", help="(sólo --run) best.pt | last.pt")
    p.add_argument("--data-root", default="data")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args(argv)
    device = resolve_device(args.device)

    if args.run:
        cfg, model = load_run(args.run, args.checkpoint)
        data_cfg, model_cfg, source = cfg.data, cfg.model, args.run / args.checkpoint
        data_cfg.num_workers = args.num_workers
    else:
        model_cfg, model, dataset, source = load_legacy(args.legacy)
        data_cfg = DataConfig(
            name=dataset, root=args.data_root, normalization="legacy_cifar", num_workers=args.num_workers
        )

    data = build_dataloaders(data_cfg, model_cfg.image_size, args.batch_size, seed=0)
    result = evaluate(model.to(device), data.test, device)
    result.update(
        checkpoint=str(source),
        dataset=data_cfg.name,
        branches=model_cfg.attention.branches,
        fusion=model_cfg.attention.fusion,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
