"""Construcción de DataLoaders con splits train/val/test explícitos y reproducibles."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Subset

from ham.config import DataConfig
from ham.datasets.sources import DATASET_REGISTRY, DATASET_STATS, DatasetSource, register_dataset
from ham.datasets.transforms import build_transforms, resolve_normalization
from ham.utils.seed import seed_worker


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    num_classes: int
    # Índices sobre el train oficial (y sobre él mismo para test si no hay test oficial).
    split: dict[str, list[int]]


def _split_indices(n: int, fractions: dict[str, float], seed: int) -> dict[str, list[int]]:
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    out, start = {}, 0
    for name, frac in fractions.items():
        size = int(round(frac * n))
        out[name] = sorted(perm[start : start + size])
        start += size
    out["train"] = sorted(perm[start:])
    return out


def build_dataloaders(cfg: DataConfig, image_size: int, batch_size: int, seed: int) -> DataBundle:
    name = cfg.name.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{cfg.name}' no soportado: {sorted(DATASET_REGISTRY)}")

    norm = resolve_normalization(cfg.normalization, DATASET_STATS[name])
    train_tf = build_transforms(image_size, train=True, augment=cfg.augment, normalization=norm)
    eval_tf = build_transforms(image_size, train=False, normalization=norm)
    src: DatasetSource = DATASET_REGISTRY[name](cfg.root, train_tf, eval_tf)

    fractions = {"val": cfg.val_fraction}
    if src.test is None:
        fractions["test"] = cfg.val_fraction
    split = _split_indices(len(src.train_aug), fractions, cfg.split_seed)
    if cfg.train_fraction < 1.0:  # subconjunto fijo (split_seed) del train, igual para todas las semillas
        keep = torch.randperm(len(split["train"]), generator=torch.Generator().manual_seed(cfg.split_seed + 1))
        split["train"] = sorted(split["train"][i] for i in keep[: int(round(cfg.train_fraction * len(keep)))].tolist())

    train_set = Subset(src.train_aug, split["train"])
    val_set = Subset(src.train_eval, split["val"])
    test_set = src.test if src.test is not None else Subset(src.train_eval, split["test"])

    common = dict(
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        persistent_workers=cfg.num_workers > 0,
    )
    return DataBundle(
        train=DataLoader(
            train_set, shuffle=True, generator=torch.Generator().manual_seed(seed), **common
        ),
        val=DataLoader(val_set, shuffle=False, **common),
        test=DataLoader(test_set, shuffle=False, **common),
        num_classes=src.num_classes,
        split=split,
    )


__all__ = ["DataBundle", "build_dataloaders", "register_dataset", "DATASET_REGISTRY"]
