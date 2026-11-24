"""Entrenamiento de un experimento.

    uv run python -m ham.train --config configs/cifar10_ham_concat.yaml --seed 0 [clave=valor ...]

Salida en ``runs/<dataset>/<name>/seed<k>/``:
    config.resolved.yaml  configuración exacta usada
    provenance.json       commit de git, versiones, GPU, comando
    split.json            índices de train/val(/test) sobre el train oficial
    metrics.json          métricas por época (train/val loss y acc, lr, tiempo; λ si es aprendible)
    best.pt / last.pt     pesos del mejor modelo (val acc) y de la última época
    test.json             evaluación final del mejor modelo en test (una sola vez)
    train.log
"""
from __future__ import annotations

import argparse
import copy
import logging
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from ham.config import ExperimentConfig, load_config, save_config
from ham.datasets import build_dataloaders
from ham.models import build_model
from ham.utils.io import run_dir_for, write_json
from ham.utils.provenance import collect_provenance
from ham.utils.seed import set_seed

log = logging.getLogger("ham")


def cosine_with_warmup(optimizer, warmup_epochs: int, total_epochs: int) -> LambdaLR:
    """Warmup lineal + decaimiento coseno, paso por época.

    A diferencia de ``utils.get_scheduler`` original, la época 0 no arranca con lr=0.
    """

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


class BatchMixer:
    """CutMix (Yun et al. 2019) y Mixup (Zhang et al. 2018) por lote, con un generador propio.

    Devuelve ``(imágenes, y_a, y_b, λ)``; la pérdida es ``λ·CE(y_a) + (1−λ)·CE(y_b)``. En CutMix
    λ se corrige al área real del recorte. ``mode=cutmix_mixup`` elige una de las dos por lote.
    """

    MODES = ("none", "cutmix", "mixup", "cutmix_mixup")

    def __init__(self, mode: str, alpha: float, prob: float, seed: int):
        if mode not in self.MODES:
            raise ValueError(f"mix '{mode}' no soportado ({' | '.join(self.MODES)})")
        self.mode, self.alpha, self.prob = mode, alpha, prob
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator().manual_seed(seed)

    def __call__(self, images, labels):
        if self.mode == "none" or self.rng.random() >= self.prob:
            return images, labels, labels, 1.0
        mode = self.mode if self.mode != "cutmix_mixup" else ("cutmix", "mixup")[int(self.rng.random() < 0.5)]
        lam = float(self.rng.beta(self.alpha, self.alpha))
        perm = torch.randperm(images.size(0), generator=self.gen).to(images.device)
        if mode == "mixup":
            return lam * images + (1 - lam) * images[perm], labels, labels[perm], lam
        h, w = images.shape[-2:]
        ch, cw = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
        cy, cx = int(self.rng.integers(h)), int(self.rng.integers(w))
        y0, y1 = max(cy - ch // 2, 0), min(cy + ch // 2, h)
        x0, x1 = max(cx - cw // 2, 0), min(cx + cw // 2, w)
        images = images.clone()
        images[..., y0:y1, x0:x1] = images[perm][..., y0:y1, x0:x1]
        return images, labels, labels[perm], 1 - (y1 - y0) * (x1 - x0) / (h * w)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def evaluate(model, loader, device, criterion=None, amp=False):
    criterion = criterion or nn.CrossEntropyLoss()
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)
        loss_sum += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return {"loss": loss_sum / total, "acc": 100.0 * correct / total, "n": total}


def fusion_lambdas(model) -> list[list[float]]:
    """λ de cada cabeza ``[capa][cabeza]`` si la fusión los aprende; ``[]`` si no."""
    out = []
    for layer in getattr(getattr(model, "encoder", None), "layers", []):  # la ResNet no tiene encoder
        heads = [h.fusion for h in layer.attention.attn_heads]
        if heads and all(getattr(f, "learnable", False) for f in heads):
            out.append([round(f.lambda_value.item(), 6) for f in heads])
    return out


class Trainer:
    def __init__(self, cfg: ExperimentConfig, model, data, device, run_dir: Path):
        self.cfg, self.model, self.data, self.device, self.run_dir = cfg, model.to(device), data, device, run_dir
        t = cfg.train
        self.criterion = nn.CrossEntropyLoss()
        self.train_criterion = nn.CrossEntropyLoss(label_smoothing=t.label_smoothing)
        self.mixer = BatchMixer(t.mix, t.mix_alpha, t.mix_prob, cfg.seed)
        # λ aprendible, escalas de zscore_score y α de enrich.global sin weight decay: lo empujaría
        # hacia λ=0.5 / escala 1 / α=0.
        no_wd = ("lambda_logit", "log_scale", "alpha")
        decay = [p for n, p in model.named_parameters() if not n.endswith(no_wd)]
        no_decay = [p for n, p in model.named_parameters() if n.endswith(no_wd)]
        groups = [{"params": decay}] + ([{"params": no_decay, "weight_decay": 0.0}] if no_decay else [])
        self.optimizer = torch.optim.AdamW(groups, lr=t.learning_rate, weight_decay=t.weight_decay)
        if t.scheduler == "cosine":
            self.scheduler = cosine_with_warmup(self.optimizer, t.warmup_epochs, t.epochs)
        elif t.scheduler == "none":
            self.scheduler = None
        else:
            raise ValueError(f"scheduler '{t.scheduler}' no soportado (cosine | none)")
        self.amp = t.amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.amp)

    def train_one_epoch(self, epoch):
        self.model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for images, labels in tqdm(self.data.train, desc=f"Epoch {epoch + 1}/{self.cfg.train.epochs}", leave=False):
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            images, y_a, y_b, lam = self.mixer(images, labels)
            with torch.autocast(self.device.type, enabled=self.amp):
                logits = self.model(images)
                loss = self.train_criterion(logits, y_a)
                if lam < 1.0:
                    loss = lam * loss + (1 - lam) * self.train_criterion(logits, y_b)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        return {"loss": loss_sum / total, "acc": 100.0 * correct / total}

    def fit(self):
        history, best_acc, best_epoch, best_state = [], -1.0, -1, None
        for epoch in range(self.cfg.train.epochs):
            t0 = time.time()
            lr = self.optimizer.param_groups[0]["lr"]
            tr = self.train_one_epoch(epoch)
            va = evaluate(self.model, self.data.val, self.device, self.criterion, self.amp)
            if self.scheduler is not None:
                self.scheduler.step()

            row = {
                "epoch": epoch + 1,
                "lr": lr,
                "train_loss": tr["loss"],
                "train_acc": tr["acc"],
                "val_loss": va["loss"],
                "val_acc": va["acc"],
                "time_s": time.time() - t0,
            }
            lambdas = fusion_lambdas(self.model)
            if lambdas:
                row["fusion_lambda"] = lambdas
            history.append(row)
            log.info(
                "epoch %d | lr %.2e | train %.4f / %.2f%% | val %.4f / %.2f%%",
                epoch + 1, lr, tr["loss"], tr["acc"], va["loss"], va["acc"],
            )
            if va["acc"] > best_acc:
                best_acc, best_epoch = va["acc"], epoch + 1
                best_state = copy.deepcopy(self.model.state_dict())
                torch.save(best_state, self.run_dir / "best.pt")
            write_json(history, self.run_dir / "metrics.json")

        torch.save(self.model.state_dict(), self.run_dir / "last.pt")
        self.model.load_state_dict(best_state)
        return history, best_acc, best_epoch


def setup_logging(run_dir: Path):
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(), logging.FileHandler(run_dir / "train.log")):
        h.setFormatter(fmt)
        log.addHandler(h)


def prepare_run_dir(cfg: ExperimentConfig) -> Path:
    run_dir = run_dir_for(cfg)
    if run_dir.exists() and any(run_dir.iterdir()):
        if not cfg.train.overwrite:
            raise FileExistsError(f"{run_dir} ya existe. Usa train.overwrite=true o cambia name/seed.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    return run_dir


def run(cfg: ExperimentConfig) -> dict:
    set_seed(cfg.seed, cfg.train.deterministic)
    device = resolve_device(cfg.train.device)

    data = build_dataloaders(cfg.data, cfg.model.image_size, cfg.train.batch_size, cfg.seed)
    cfg.model.num_classes = data.num_classes
    model = build_model(cfg.model)

    run_dir = prepare_run_dir(cfg)
    setup_logging(run_dir)
    save_config(cfg, run_dir / "config.resolved.yaml")
    write_json(collect_provenance(), run_dir / "provenance.json")
    write_json(data.split, run_dir / "split.json")

    n_params = sum(p.numel() for p in model.parameters())
    log.info("run: %s", run_dir)
    log.info(
        "dataset=%s train/val/test=%d/%d/%d classes=%d | branches=%s fusion=%s | params=%d | device=%s",
        cfg.data.name, len(data.train.dataset), len(data.val.dataset), len(data.test.dataset),
        data.num_classes, cfg.model.attention.branches, cfg.model.attention.fusion, n_params, device,
    )

    trainer = Trainer(cfg, model, data, device, run_dir)
    _, best_val, best_epoch = trainer.fit()
    test = evaluate(trainer.model, data.test, device, amp=trainer.amp)
    result = {
        "test_acc": test["acc"],
        "test_loss": test["loss"],
        "test_n": test["n"],
        "best_val_acc": best_val,
        "best_epoch": best_epoch,
        "num_params": n_params,
    }
    write_json(result, run_dir / "test.json")
    log.info("test acc (best val epoch %d): %.2f%%", best_epoch, test["acc"])
    return result


def resume_action(cfg: ExperimentConfig) -> str:
    """Para ``--skip-done``: ``"skip"`` si el run ya terminó (tiene ``test.json``),
    ``"restart"`` si quedó incompleto (se borra y se repite) o ``"new"``."""
    run_dir = run_dir_for(cfg)
    if (run_dir / "test.json").exists():
        return "skip"
    if run_dir.exists() and any(run_dir.iterdir()):
        return "restart"
    return "new"


def main(argv=None):
    p = argparse.ArgumentParser(description="Entrena un ViT con atención homogénea o heterogénea (HAM).")
    p.add_argument("--config", required=True, help="YAML del experimento")
    p.add_argument("--seed", type=int, default=None, help="sobrescribe 'seed' del YAML")
    p.add_argument(
        "--skip-done", action="store_true",
        help="no repetir runs con test.json y reiniciar los que quedaron incompletos (para lotes largos)",
    )
    p.add_argument("overrides", nargs="*", help="sobrescrituras clave.sub=valor")
    args = p.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.skip_done:
        action = resume_action(cfg)
        if action == "skip":
            print(f"ya terminado, se omite: {run_dir_for(cfg)}")
            return
        if action == "restart":
            print(f"run incompleto, se reinicia: {run_dir_for(cfg)}")
            cfg.train.overwrite = True
    run(cfg)


if __name__ == "__main__":
    main()
