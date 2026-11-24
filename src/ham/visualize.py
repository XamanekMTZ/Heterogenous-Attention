"""Mapas de atención CLS→parches: por rama, por capa y comparando variantes.

    # un run: una columna por rama (y la atención efectiva en HAM)
    uv run python -m ham.visualize --run runs/cifar10/cifar10_ham_concat/seed0
    # un run, capa por capa (filas = imagen × capa, columnas = ramas)
    uv run python -m ham.visualize --run runs/cifar10/cifar10_ham_weighted/seed0 --layers --num-images 2
    # varias variantes sobre las mismas imágenes; la primera es la referencia (dot)
    uv run python -m ham.visualize --compare runs/cifar10/cifar10_{dot,ham_concat,ham_weighted}/seed0 \
        --select disagree --out comparativa.png

Modos (``--mode``):
- ``rollout`` (por defecto): attention rollout (Abnar y Zuidema 2020) a través de las
  capas. En HAM, cada rama se muestra con un rollout sólo por esa rama, más la atención
  efectiva: ``fused`` en las fusiones de scores, o la media de ramas (``efectiva``) en
  ``concat_linear``, donde es una aproximación porque esa fusión mezcla salidas.
- ``mean``: la reducción de ``utils_visual.visualize_attention`` del código original
  (fila del CLS, media sobre cabezas y capas).

La rama aditiva tipo pooling (``[B,H,N]``) se trata como una matriz de filas iguales.
``--select disagree`` toma imágenes donde la referencia falla y las demás aciertan, y
viceversa; requiere ``predictions.npz`` (``python -m ham.analyze``) en cada run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ham import analysis as A  # noqa: E402
from ham.datasets import build_dataloaders  # noqa: E402
from ham.datasets.sources import DATASET_STATS  # noqa: E402
from ham.datasets.transforms import resolve_normalization  # noqa: E402
from ham.train import resolve_device  # noqa: E402
from ham.utils.io import load_run  # noqa: E402


def cls_attention_maps(attentions, branches=("attention",), mode="mean") -> dict[str, torch.Tensor]:
    """Lista por capa de pesos -> ``{rama: [B, N_patches]}``."""
    layers = A.layer_dicts(attentions, list(branches))
    names = list(layers[0])
    if mode == "mean":
        maps = {n: torch.stack([layer[n].mean(1)[:, 0, 1:] for layer in layers]).mean(0) for n in names}
        if len(names) > 1 and "fused" not in names:
            maps["efectiva"] = torch.stack(list(maps.values())).mean(0)
        return maps
    if mode != "rollout":
        raise ValueError(f"mode '{mode}' no soportado (rollout | mean)")
    maps = {n: A.rollout([layer[n].mean(1) for layer in layers])[:, 0, 1:] for n in names}
    if len(names) > 1 and "fused" not in names:
        maps["efectiva"] = A.rollout([A.effective_attention(layer) for layer in layers])[:, 0, 1:]
    return maps


def effective_map(maps: dict[str, torch.Tensor]) -> torch.Tensor:
    """El mapa de la atención que el modelo aplica: la única rama, ``fused`` o ``efectiva``."""
    for key in ("fused", "efectiva"):
        if key in maps:
            return maps[key]
    return next(iter(maps.values()))


def layer_maps(attentions, branches) -> list[dict[str, torch.Tensor]]:
    """Fila del CLS por capa (media de cabezas): ``[{rama: [B, N_patches]}]``."""
    return [{n: w.mean(1)[:, 0, 1:] for n, w in layer.items()} for layer in A.layer_dicts(attentions, branches)]


def denormalize(x, norm):
    if norm is None:
        return x
    mean, std = (torch.tensor(v, device=x.device).view(1, 3, 1, 1) for v in norm)
    return x * std + mean


def _overlay(ax, img, m, side):
    g = int(m.numel() ** 0.5)
    m = F.interpolate(m.view(1, 1, g, g).float(), size=(side, side), mode="bilinear", align_corners=False)[0, 0]
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    ax.imshow(img)
    ax.imshow(m.cpu().numpy(), cmap="jet", alpha=0.5)


def plot_grid(images, columns, row_titles, path, cell_titles=None):
    """``images``: ``[n,3,S,S]`` en [0,1]; ``columns``: ``[(título, mapas [n,P])]``."""
    n, side = images.size(0), images.size(-1)
    ncol = 1 + len(columns)
    fig, axes = plt.subplots(n, ncol, figsize=(2.2 * ncol, 2.3 * n), squeeze=False)
    for i in range(n):
        img = images[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(row_titles[i], fontsize=8)
        for j, (title, maps) in enumerate(columns, start=1):
            _overlay(axes[i, j], img, maps[i], side)
            sub = cell_titles[i][j - 1] if cell_titles else ""
            if i == 0 or sub:
                axes[i, j].set_title(f"{title}\n{sub}" if i == 0 and sub else (title if i == 0 else sub), fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


class TestImages:
    """Imágenes del test de un run, por índice (mismo orden que ``predictions.npz``)."""

    def __init__(self, cfg, device):
        cfg.data.num_workers = 0
        data = build_dataloaders(cfg.data, cfg.model.image_size, batch_size=1, seed=0)
        self.dataset = data.test.dataset
        self.device = device
        self.norm = resolve_normalization(cfg.data.normalization, DATASET_STATS[cfg.data.name.lower()])
        base = getattr(self.dataset, "dataset", self.dataset)  # Subset -> dataset
        self.classes = getattr(base, "classes", None)

    def __len__(self):
        return len(self.dataset)

    def get(self, indices):
        batch = [self.dataset[int(i)] for i in indices]
        return torch.stack([b[0] for b in batch]).to(self.device), [b[1] for b in batch]

    def name(self, label):
        return str(self.classes[label])[:14] if self.classes else str(label)


def random_indices(n_total, n, seed):
    return torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))[:n].tolist()


def disagree_indices(runs, n, seed):
    """Mitad: la referencia (``runs[0]``) falla y la mayoría de las demás acierta; mitad: al revés."""
    correct = []
    for run in runs:
        p = np.load(run / "predictions.npz")
        correct.append(p["logits"].astype(np.float32).argmax(1) == p["labels"])
    ref, others = correct[0], np.mean(correct[1:], axis=0)
    rng = np.random.default_rng(seed)
    a = rng.permutation(np.flatnonzero(~ref & (others > 0.5)))
    b = rng.permutation(np.flatnonzero(ref & (others < 0.5)))
    picked = list(a[: (n + 1) // 2]) + list(b[: n // 2])
    rest = [int(i) for i in rng.permutation(len(ref)) if i not in set(picked)]
    return [int(i) for i in picked] + rest[: n - len(picked)]


@torch.no_grad()
def forward(model, images):
    model.eval()
    return model(images, output_attentions=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", type=Path)
    g.add_argument("--compare", type=Path, nargs="+", help="runs del mismo dataset; el primero es la referencia")
    p.add_argument("--layers", action="store_true", help="(con --run) un mapa por capa y rama")
    p.add_argument("--mode", default="rollout", choices=["rollout", "mean"])
    p.add_argument("--select", default="random", choices=["random", "disagree"])
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--num-images", type=int, default=None, help="default: 8 (2 con --layers)")
    p.add_argument("--index-seed", type=int, default=42, help="semilla para elegir imágenes de test")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    runs = args.compare or [args.run]
    n = args.num_images or (2 if args.layers else 8)
    models = [load_run(r, args.checkpoint) for r in runs]
    cfg0 = models[0][0]
    if len({c.data.name for c, _ in models}) > 1:
        raise ValueError("--compare requiere runs del mismo dataset")
    test = TestImages(cfg0, device)
    idx = disagree_indices(runs, n, args.index_seed) if args.select == "disagree" else \
        random_indices(len(test), n, args.index_seed)
    images, labels = test.get(idx)
    shown = denormalize(images, test.norm)

    if args.compare:
        columns, cells = [], [[] for _ in range(n)]
        for cfg, model in models:
            logits, attentions = forward(model.to(device), images)
            maps = cls_attention_maps(attentions, cfg.model.attention.branches, args.mode)
            columns.append((cfg.name.removeprefix(f"{cfg.data.name}_"), effective_map(maps)))
            for i, pred in enumerate(logits.argmax(1).tolist()):
                cells[i].append(f"{'✓' if pred == labels[i] else '✗'} {test.name(pred)}")
            model.cpu()
        rows = [f"#{i} {test.name(y)}" for i, y in zip(idx, labels)]
        out = args.out or Path("attention_compare.png")
        plot_grid(shown, columns, rows, out, cells)
    elif args.layers:
        cfg, model = models[0]
        _, attentions = forward(model.to(device), images)
        per_layer = layer_maps(attentions, cfg.model.attention.branches)
        names = list(per_layer[0])
        nl = len(per_layer)
        rep = shown.repeat_interleave(nl, 0)
        columns = [(name, torch.stack([per_layer[l][name][i] for i in range(n) for l in range(nl)])) for name in names]
        rows = [f"{test.name(labels[i])} · capa {l + 1}" for i in range(n) for l in range(nl)]
        out = args.out or args.run / "attention_layers.png"
        plot_grid(rep, columns, rows, out)
    else:
        cfg, model = models[0]
        logits, attentions = forward(model.to(device), images)
        maps = cls_attention_maps(attentions, cfg.model.attention.branches, args.mode)
        rows = [f"y={test.name(y)} ŷ={test.name(p)}" for y, p in zip(labels, logits.argmax(1).tolist())]
        out = args.out or args.run / "attention_maps.png"
        plot_grid(shown, list(maps.items()), rows, out)
    print(f"Guardado en {out}")


if __name__ == "__main__":
    main()
