"""Curvas de entrenamiento por experimento (media ± desviación entre semillas).

    uv run python scripts/plot_curves.py [--runs runs] [--out figuras/]

Genera ``curves_<dataset>.png`` (val acc y train loss por época) y, si algún
experimento aprende λ (``fusion_lambda`` en ``metrics.json``), ``lambda_<dataset>.png``.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def collect(runs: Path):
    """``{dataset: {experimento: [historia por semilla]}}``; sólo runs terminados."""
    groups = defaultdict(lambda: defaultdict(list))
    for metrics in sorted(runs.glob("*/*/seed*/metrics.json")):
        if not (metrics.parent / "test.json").exists():
            continue
        dataset, name = metrics.parts[-4], metrics.parts[-3]
        groups[dataset][name].append(json.loads(metrics.read_text()))
    return groups


def _band(ax, histories, key, label):
    n = min(len(h) for h in histories)
    values = np.array([[row[key] for row in h[:n]] for h in histories])
    epochs = np.arange(1, n + 1)
    mean, std = values.mean(0), values.std(0)
    (line,) = ax.plot(epochs, mean, label=f"{label} (n={len(histories)})")
    ax.fill_between(epochs, mean - std, mean + std, color=line.get_color(), alpha=0.2)


def plot_curves(dataset, experiments, out: Path):
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, histories in sorted(experiments.items()):
        _band(ax_acc, histories, "val_acc", name)
        _band(ax_loss, histories, "train_loss", name)
    ax_acc.set(title=f"{dataset}: accuracy de validación", xlabel="época", ylabel="val acc (%)")
    ax_loss.set(title=f"{dataset}: pérdida de entrenamiento", xlabel="época", ylabel="train loss")
    for ax in (ax_acc, ax_loss):
        ax.grid(alpha=0.3)
    ax_acc.legend(fontsize=8)
    fig.tight_layout()
    path = out / f"curves_{dataset}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_lambdas(dataset, experiments, out: Path):
    learned = {n: hs for n, hs in experiments.items() if "fusion_lambda" in hs[0][0]}
    if not learned:
        return None
    fig, axes = plt.subplots(1, len(learned), figsize=(6 * len(learned), 4.5), squeeze=False)
    for ax, (name, histories) in zip(axes[0], sorted(learned.items())):
        n = min(len(h) for h in histories)
        # [semilla, época, capa] promediando cabezas
        lam = np.array([[np.mean(row["fusion_lambda"], axis=1) for row in h[:n]] for h in histories])
        epochs = np.arange(1, n + 1)
        for layer in range(lam.shape[2]):
            mean, std = lam[:, :, layer].mean(0), lam[:, :, layer].std(0)
            (line,) = ax.plot(epochs, mean, label=f"capa {layer + 1}")
            ax.fill_between(epochs, mean - std, mean + std, color=line.get_color(), alpha=0.2)
        ax.set(title=f"{dataset} / {name}: λ (media de cabezas)", xlabel="época", ylabel="λ (peso de dot)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / f"lambda_{dataset}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--out", type=Path, default=Path("."))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    groups = collect(args.runs)
    if not groups:
        print(f"No hay runs terminados bajo {args.runs}")
        return
    for dataset, experiments in sorted(groups.items()):
        for path in (plot_curves(dataset, experiments, args.out), plot_lambdas(dataset, experiments, args.out)):
            if path:
                print(f"Guardado en {path}")


if __name__ == "__main__":
    main()
