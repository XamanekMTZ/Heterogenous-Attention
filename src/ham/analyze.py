"""Análisis post-entrenamiento de un run (sólo inferencia sobre ``best.pt``).

    uv run python -m ham.analyze --run runs/cifar10/cifar10_ham_weighted/seed0 [--skip-done]

Escribe en el directorio del run:
    analysis.json     métricas de clasificación (F1, ECE, ...), atención por capa y rama,
                      deletion, robustez y eficiencia; incluye su propio provenance
    predictions.npz   logits (float16) y etiquetas del test, en el orden del loader,
                      para pruebas pareadas entre variantes (``scripts/compare.py``)

Todo es determinista: subconjuntos y perturbaciones usan semillas fijas.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from ham import analysis as A
from ham.datasets import build_dataloaders
from ham.datasets.sources import DATASET_STATS
from ham.datasets.transforms import resolve_normalization
from ham.train import resolve_device
from ham.utils.io import load_run, read_json, write_json
from ham.utils.provenance import collect_provenance
from ham.utils.seed import set_seed


class Normalizer:
    """Pasa de tensores normalizados a píxeles en [0, 1] y de vuelta."""

    def __init__(self, norm, device):
        self.mean = self.std = None
        if norm is not None:
            self.mean, self.std = (torch.tensor(v, device=device).view(1, 3, 1, 1) for v in norm)

    def to_pixels(self, x):
        return x if self.mean is None else x * self.std + self.mean

    def from_pixels(self, x):
        return x if self.mean is None else (x - self.mean) / self.std


@torch.no_grad()
def predict(model, loader, device, transform=None):
    logits, labels = [], []
    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        if transform is not None:
            x = transform(x, i)
        logits.append(model(x).float().cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def robustness(model, loader, device, normalizer, patch_size, clean_acc, seed):
    def accuracy(transform):
        logits, labels = predict(model, loader, device, transform)
        return 100.0 * (logits.argmax(1) == labels).float().mean().item()

    out = {"clean": clean_acc}
    for sigma in A.ROBUSTNESS["noise"]:
        out[f"noise_{sigma}"] = accuracy(lambda x, i, s=sigma: normalizer.from_pixels(
            A.gaussian_noise(normalizer.to_pixels(x), s, torch.Generator().manual_seed(seed + i))))
    for sigma in A.ROBUSTNESS["blur"]:
        out[f"blur_{sigma}"] = accuracy(lambda x, i, s=sigma: normalizer.from_pixels(
            A.gaussian_blur(normalizer.to_pixels(x), s)))
    for frac in A.ROBUSTNESS["occlusion"]:
        def occlude(x, i, f=frac):
            p = (x.size(-1) // patch_size) ** 2
            mask = A.random_patch_mask(x.size(0), p, f, torch.Generator().manual_seed(seed + i))
            return A.delete_patches(x, mask.to(x.device), patch_size)
        out[f"occlusion_{frac}"] = accuracy(occlude)
    return out


@torch.no_grad()
def attention_and_deletion(model, images, branches, patch_size, batch, n_deletion, seed):
    """Métricas de atención sobre ``images`` y curvas de deletion sobre las primeras ``n_deletion``."""
    stats = A.AttentionStats(patch_size)
    curves = {"rollout": [], "random": []}
    contrib_sum, contrib_n = None, 0
    enrich_parts = []  # (resultado de enrich_contributions, tamaño del lote)
    gen = torch.Generator().manual_seed(seed)
    for start in range(0, images.size(0), batch):
        x = images[start:start + batch]
        logits, attentions = model(x, output_attentions=True)
        layers = A.layer_dicts(attentions, branches)
        stats.update(layers)

        contrib = A.branch_contributions(model, x)
        if contrib is not None:
            if contrib_sum is None:
                contrib_sum = {k: np.zeros(len(v)) for k, v in contrib.items()}
            for k, v in contrib.items():
                contrib_sum[k] += np.array(v) * x.size(0)
            contrib_n += x.size(0)
        enrich = A.enrich_contributions(model, x)
        if enrich is not None:
            enrich_parts.append((enrich, x.size(0)))

        n_del = min(x.size(0), max(0, n_deletion - start))
        if n_del == 0:
            continue
        x, target = x[:n_del], logits[:n_del].argmax(1)
        cls_map = A.rollout([A.effective_attention(layer)[:n_del] for layer in layers])[:, 0, 1:]
        num_patches = cls_map.size(1)
        orders = {
            "rollout": cls_map.argsort(1, descending=True).cpu(),
            "random": torch.rand(n_del, num_patches, generator=gen).argsort(1),
        }
        for name, order in orders.items():
            probs = []
            for frac in A.DELETION_FRACTIONS:
                k = int(round(frac * num_patches))
                mask = torch.zeros(n_del, num_patches, dtype=torch.bool)
                mask.scatter_(1, order[:, :k], True)
                p = model(A.delete_patches(x, mask.to(x.device), patch_size)).softmax(-1)
                probs.append(p.gather(1, target[:, None]).squeeze(1).cpu())
            curves[name].append(torch.stack(probs, 1))  # [b, pasos]

    deletion = {"fractions": A.DELETION_FRACTIONS, "n_images": min(n_deletion, images.size(0))}
    for name, parts in curves.items():
        mean = torch.cat(parts).mean(0)
        deletion[name] = mean.tolist()
        deletion[f"auc_{name}"] = float(np.trapezoid(mean.numpy(), A.DELETION_FRACTIONS))
    contributions = None if contrib_sum is None else {k: (v / contrib_n).tolist() for k, v in contrib_sum.items()}
    attention = stats.result()
    if enrich_parts:
        attention["enrich_contribution"] = mean_enrich(enrich_parts)
    return attention, deletion, contributions


def mean_enrich(parts):
    """Media ponderada por lote de los resultados de ``enrich_contributions`` (listas por capa)."""
    first, n = parts[0][0], sum(w for _, w in parts)
    out = {"enrich": first["enrich"]}
    for section in ("terms", "gate"):
        if section in first:
            out[section] = {k: (sum(np.array(r[section][k]) * w for r, w in parts) / n).tolist()
                            for k in first[section]}
    return out


def efficiency(model, cfg, device, seed):
    from torch.utils.flop_counter import FlopCounterMode

    s = cfg.model.image_size
    g = torch.Generator().manual_seed(seed)
    out = {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(torch.randn(1, 3, s, s, generator=g).to(device))
    out["gflops_per_image"] = fc.get_total_flops() / 1e9
    # FlopCounterMode sólo cuenta matmul/conv. Las ramas por pares hacen además
    # suma + tanh + reducción elemento a elemento sobre [N, N, d] por cabeza y capa.
    m = cfg.model
    n_tok = (s // m.patch_size) ** 2 + 1
    heads = m.attention.per_head(m.num_attention_heads)
    pairwise = sum(b in ("additive_pairwise", "additive_free", "additive_free_scaled") for h in heads for b in h) / len(heads)
    out["pairwise_elementwise_gflops"] = 3 * pairwise * m.num_layers * m.hidden_size * n_tok**2 / 1e9

    bs = cfg.train.batch_size
    x = torch.randn(bs, 3, s, s, generator=g).to(device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    with torch.no_grad():
        for _ in range(3):
            model(x)
        sync()
        t0, iters = time.perf_counter(), 10
        for _ in range(iters):
            model(x)
        sync()
    out["throughput_img_s"] = bs * iters / (time.perf_counter() - t0)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model.train()
        loss = model(x).logsumexp(-1).mean()
        loss.backward()
        sync()
        out["train_peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
        model.zero_grad(set_to_none=True)
        model.eval()
    return out


def analyze(run_dir: Path, device, n_attention=1000, n_deletion=500, attn_batch=100, seed=0, num_workers=4):
    set_seed(seed, deterministic=True)
    cfg, model = load_run(run_dir)
    model.to(device).eval()
    cfg.data.num_workers = num_workers
    data = build_dataloaders(cfg.data, cfg.model.image_size, cfg.train.batch_size, seed=0)
    norm = resolve_normalization(cfg.data.normalization, DATASET_STATS[cfg.data.name.lower()])
    normalizer = Normalizer(norm, device)
    patch = cfg.model.patch_size
    branches = cfg.model.attention.branches
    is_ham_fusion = any(len(h) > 1 for h in cfg.model.attention.per_head(cfg.model.num_attention_heads))

    logits, labels = predict(model, data.test, device)
    cls = A.classification_metrics(logits, labels, data.num_classes)
    test = read_json(run_dir / "test.json") if (run_dir / "test.json").exists() else {}
    if test and abs(test["test_acc"] - cls["acc"]) > 0.02:
        print(f"AVISO: accuracy {cls['acc']:.2f} difiere de test.json {test['test_acc']:.2f}")

    testset = data.test.dataset
    idx = torch.randperm(len(testset), generator=torch.Generator().manual_seed(seed))[:n_attention]
    images = torch.stack([testset[int(i)][0] for i in idx]).to(device)
    if cfg.model.arch == "vit":
        attention, deletion, contributions = attention_and_deletion(
            model, images, branches, patch, attn_batch, n_deletion, seed)
        attention["layers"] = cfg.model.num_layers
        attention["n_images"] = len(idx)
        attention["branch_contribution"] = contributions
    else:  # CNN de referencia: sin atención ni rollout
        attention, deletion = None, None

    result = {
        "run": str(run_dir),
        "dataset": cfg.data.name,
        "experiment": cfg.name,
        "seed": cfg.seed,
        "branches": branches,
        "head_branches": cfg.model.attention.head_branches,
        "enrich": cfg.model.attention.enrich,
        "fusion": cfg.model.attention.fusion if is_ham_fusion else None,
        "num_classes": data.num_classes,
        "classification": cls,
        "attention": attention,
        "deletion": deletion,
        "robustness": robustness(model, data.test, device, normalizer, patch, cls["acc"], seed),
        "efficiency": efficiency(model, cfg, device, seed),
        "analysis_seed": seed,
        "provenance": collect_provenance(),
    }
    np.savez_compressed(run_dir / "predictions.npz", logits=logits.half().numpy(), labels=labels.numpy())
    write_json(result, run_dir / "analysis.json")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--skip-done", action="store_true", help="no repetir si ya existe analysis.json")
    p.add_argument("--n-attention", type=int, default=1000, help="imágenes de test para métricas de atención")
    p.add_argument("--n-deletion", type=int, default=500, help="imágenes para las curvas de deletion")
    p.add_argument("--attn-batch", type=int, default=100)
    p.add_argument("--seed", type=int, default=0, help="semilla del análisis (subconjuntos y perturbaciones)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    if args.skip_done and (args.run / "analysis.json").exists():
        print(f"ya analizado, se omite: {args.run}")
        return
    t0 = time.time()
    r = analyze(args.run, resolve_device(args.device), args.n_attention, args.n_deletion,
                args.attn_batch, args.seed, args.num_workers)
    c, d = r["classification"], r["deletion"]
    auc = f"deletion AUC {d['auc_rollout']:.3f} (azar {d['auc_random']:.3f})" if d else "sin atención"
    print(f"{args.run}: acc {c['acc']:.2f} | F1 {c['f1_macro']:.2f} | ECE {c['ece']:.2f} | {auc} | {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
