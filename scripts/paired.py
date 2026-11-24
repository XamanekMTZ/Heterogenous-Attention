"""Comparaciones pareadas por semilla contra una referencia, desde ``test.json`` y ``metrics.json``.

    uv run python scripts/paired.py --dataset cifar10 --ref dot dot_local dot_gate [--roots runs runs_seeds]
    uv run python scripts/paired.py --dataset cifar10 --ref dot_wide dot_local --threshold 79

Junta las semillas de cada experimento en todas las ``--roots`` (p. ej. las 0-4 de ``runs/`` y las
5-15 de ``runs_seeds/``) y, para cada variante, sólo usa las semillas que también tiene la referencia.
Columnas: accuracy media ± desv., Δ pareada media ± desv. con IC 95 % (t), semillas con Δ>0,
p de la t pareada, p con Holm dentro de la tabla, p de Wilcoxon (n ≥ 6), parámetros, época en que
val_acc alcanza ``--threshold`` (media de las semillas que llegan / cuántas llegan) y s/época.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from scipy import stats


def load(roots, dataset, name):
    """``{semilla: (test.json, metrics.json)}`` de ``<root>/<dataset>/<dataset>_<name>/seed*``."""
    out = {}
    for root in roots:
        for f in sorted(Path(root, dataset, f"{dataset}_{name}").glob("seed*/test.json")):
            seed = int(f.parent.name.removeprefix("seed"))
            if seed in out:
                raise ValueError(f"Semilla {seed} de {name} repetida en varias raíces")
            out[seed] = (json.loads(f.read_text()), json.loads((f.parent / "metrics.json").read_text()))
    return out


def holm(ps):
    order = np.argsort(ps)
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def epoch_reaching(metrics, threshold):
    return next((r["epoch"] for r in metrics if r["val_acc"] >= threshold), None)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--ref", required=True, help="experimento de referencia, sin el prefijo del dataset")
    p.add_argument("variants", nargs="+")
    p.add_argument("--roots", nargs="+", default=["runs", "runs_seeds"])
    p.add_argument("--threshold", type=float, default=None, help="val_acc para medir épocas hasta alcanzarla")
    args = p.parse_args(argv)

    ref = load(args.roots, args.dataset, args.ref)
    if not ref:
        raise SystemExit(f"Sin runs de {args.ref} en {args.roots}")
    acc = lambda runs, s: runs[s][0]["test_acc"]  # noqa: E731
    ref_acc = [acc(ref, s) for s in ref]
    print(f"{args.ref}: {statistics.mean(ref_acc):.2f} ± {statistics.stdev(ref_acc):.2f} (n={len(ref)})")

    rows = []
    for name in args.variants:
        runs = load(args.roots, args.dataset, name)
        seeds = sorted(set(runs) & set(ref))
        if len(seeds) < 2:
            print(f"{name}: menos de 2 semillas pareadas, se omite")
            continue
        a = np.array([acc(runs, s) for s in seeds])
        d = a - np.array([acc(ref, s) for s in seeds])
        half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        epochs = [epoch_reaching(runs[s][1], args.threshold) for s in seeds] if args.threshold else []
        rows.append({
            "name": name, "n": len(seeds), "acc": a.mean(), "sd": a.std(ddof=1), "delta": d.mean(),
            "delta_sd": d.std(ddof=1), "ci": (d.mean() - half, d.mean() + half), "wins": int((d > 0).sum()),
            "p": stats.ttest_1samp(d, 0).pvalue,
            "p_w": stats.wilcoxon(d).pvalue if len(d) >= 6 and np.any(d != 0) else None,
            "params": runs[seeds[0]][0].get("num_params"),
            "epoch": [e for e in epochs if e is not None],
            "s_epoch": statistics.mean(statistics.mean(r["time_s"] for r in runs[s][1][1:]) for s in seeds),
        })
    for row, p_holm in zip(rows, holm([r["p"] for r in rows])):
        row["p_holm"] = p_holm

    header = f"{'variante':<24}{'n':>3} {'acc':>13} {'Δ':>13} {'IC 95 %':>17} {'Δ>0':>6} {'p':>7} {'Holm':>7} {'Wilc.':>7} {'params':>8}"
    header += f" {'época≥' + format(args.threshold, 'g'):>14}" if args.threshold else ""
    print(header + f" {'s/época':>8}")
    for r in sorted(rows, key=lambda r: -r["delta"]):
        line = (f"{r['name']:<24}{r['n']:>3} {r['acc']:6.2f} ± {r['sd']:.2f} {r['delta']:+6.2f} ± {r['delta_sd']:.2f} "
                f"[{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] {r['wins']:>3}/{r['n']:<2} {r['p']:7.3f} {r['p_holm']:7.3f} "
                f"{'—' if r['p_w'] is None else format(r['p_w'], '.3f'):>7} {r['params'] or '—':>8}")
        if args.threshold:
            reached = f"{statistics.mean(r['epoch']):.0f} ({len(r['epoch'])}/{r['n']})" if r["epoch"] else "—"
            line += f" {reached:>14}"
        print(line + f" {r['s_epoch']:8.2f}")


if __name__ == "__main__":
    main()
