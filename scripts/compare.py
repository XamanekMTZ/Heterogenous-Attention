"""Compara variantes con ``analysis.json`` y ``predictions.npz`` (``python -m ham.analyze``).

    uv run python scripts/compare.py [--runs runs] [--out results/final] [--ref dot]

Genera en ``--out``:
    metrics_extra.csv/.tex   media ± desv. entre semillas: F1/precision/recall macro, ECE, NLL,
                             top-5, AUC de deletion, robustez y eficiencia
    significance.csv         cada variante contra la referencia (``<dataset>_<ref>``), por semilla:
                             McNemar exacto e IC 95% bootstrap de la diferencia de accuracy;
                             fila ``media``: prueba t pareada entre semillas (poca potencia con n=3)
    attn_layers_<ds>.png     entropía (CLS y todas las filas), distancia media y diversidad de cabezas por capa
    attn_js_<ds>.png         JS entre ramas por capa
    branch_contrib_<ds>.png  contribución de cada rama en fusiones de salida (concat_linear)
    reliability_<ds>.png     diagramas de calibración
    perclass_delta_<ds>.png  accuracy por clase menos la de la referencia
    deletion_<ds>.png        curvas de deletion (rollout contra orden aleatorio)
    robustness_<ds>.png      accuracy contra severidad de cada perturbación
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ham.analysis import ROBUSTNESS, bootstrap_diff, mcnemar  # noqa: E402

COLUMNS = [  # (sección de analysis.json, clave, nombre en la tabla)
    ("classification", "acc", "acc"),
    ("classification", "f1_macro", "f1_macro"),
    ("classification", "precision_macro", "precision_macro"),
    ("classification", "recall_macro", "recall_macro"),
    ("classification", "top5", "top5"),
    ("classification", "ece", "ece"),
    ("classification", "nll", "nll"),
    ("deletion", "auc_rollout", "deletion_auc"),
    ("deletion", "auc_random", "deletion_auc_random"),
    *[("robustness", f"{kind}_{s}", f"{kind}_{s}") for kind, sev in ROBUSTNESS.items() for s in sev],
    ("efficiency", "gflops_per_image", "gflops"),
    ("efficiency", "pairwise_elementwise_gflops", "pairwise_elementwise_gflops"),
    ("efficiency", "throughput_img_s", "throughput_img_s"),
    ("efficiency", "train_peak_mem_mb", "train_peak_mem_mb"),
]
LATEX = [("f1_macro", "F1 macro", 2), ("ece", "ECE", 2), ("nll", "NLL", 3), ("deletion_auc", "AUC del.", 3),
         ("noise_0.1", "Ruido 0.1", 2), ("occlusion_0.25", "Ocl. 25\\%", 2), ("gflops", "GFLOPs", 4)]


def collect(runs: Path):
    """``{dataset: {experimento: {semilla: analysis}}}``."""
    groups = defaultdict(lambda: defaultdict(dict))
    for f in sorted(runs.glob("*/*/seed*/analysis.json")):
        a = json.loads(f.read_text())
        a["_dir"] = f.parent
        groups[f.parts[-4]][f.parts[-3]][a["seed"]] = a
    return groups


def short(dataset, name):
    return name.removeprefix(f"{dataset}_")


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def metrics_table(groups):
    rows = []
    for ds, exps in sorted(groups.items()):
        for name, seeds in sorted(exps.items()):
            row = {"dataset": ds, "experiment": name, "n_seeds": len(seeds)}
            for section, key, col in COLUMNS:
                m, s = mean_std([a[section].get(key) for a in seeds.values()])
                if m is not None:
                    row[f"{col}_mean"], row[f"{col}_std"] = m, s
            rows.append(row)
    return rows


def write_metrics(rows, csv_path: Path, tex_path: Path):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    cols = [c for c in LATEX if any(f"{c[0]}_mean" in r for r in rows)]
    lines = [r"\begin{tabular}{ll" + "r" * len(cols) + "}", r"\toprule",
             "Dataset & Experimento & " + " & ".join(c[1] for c in cols) + r" \\", r"\midrule"]
    for r in rows:
        cells = []
        for key, _, dec in cols:
            m = r.get(f"{key}_mean")
            cells.append("--" if m is None else f"${m:.{dec}f} \\pm {r[f'{key}_std']:.{dec}f}$")
        lines.append(f"{r['dataset']} & {short(r['dataset'], r['experiment']).replace('_', chr(92) + '_')} & "
                     + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path.write_text("\n".join(lines) + "\n")


def correct_vector(a):
    p = np.load(a["_dir"] / "predictions.npz")
    return p["logits"].astype(np.float32).argmax(1) == p["labels"], p["labels"]


def significance(groups, ref: str):
    from scipy.stats import ttest_rel

    rows = []
    for ds, exps in sorted(groups.items()):
        ref_name = f"{ds}_{ref}"
        if ref_name not in exps:
            print(f"AVISO: no hay {ref_name}; se omite la significancia de {ds}")
            continue
        for name, seeds in sorted(exps.items()):
            if name == ref_name:
                continue
            accs_ref, accs = [], []
            for seed in sorted(set(seeds) & set(exps[ref_name])):
                c_exp, y_exp = correct_vector(seeds[seed])
                c_ref, y_ref = correct_vector(exps[ref_name][seed])
                if not np.array_equal(y_exp, y_ref):
                    raise ValueError(f"{ds}/{name} seed{seed}: el test no coincide con la referencia")
                mc, bs = mcnemar(c_exp, c_ref), bootstrap_diff(c_exp, c_ref, seed=seed)
                accs.append(100.0 * c_exp.mean())
                accs_ref.append(100.0 * c_ref.mean())
                rows.append({"dataset": ds, "experiment": name, "reference": ref_name, "seed": seed,
                             "acc": accs[-1], "acc_ref": accs_ref[-1], "diff": bs["diff"],
                             "ci_low": bs["ci_low"], "ci_high": bs["ci_high"],
                             "only_exp_correct": mc["n10"], "only_ref_correct": mc["n01"],
                             "p_mcnemar": mc["p_value"]})
            if len(accs) > 1:
                t = ttest_rel(accs, accs_ref)
                rows.append({"dataset": ds, "experiment": name, "reference": ref_name, "seed": "media",
                             "acc": statistics.mean(accs), "acc_ref": statistics.mean(accs_ref),
                             "diff": statistics.mean(a - b for a, b in zip(accs, accs_ref)),
                             "p_ttest_seeds": float(t.pvalue)})
    return rows


# --------------------------------------------------------------------------- figuras

def _band(ax, x, series, label, **kw):
    arr = np.array(series, dtype=float)
    mean, std = arr.mean(0), arr.std(0)
    (line,) = ax.plot(x, mean, label=label, **kw)
    ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.15)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado en {path}")


def plot_attention(ds, exps, out: Path):
    metrics = [("entropy_cls", "entropía CLS→parches (1 = uniforme)"), ("entropy_all", "entropía, todas las filas"),
               ("distance_px", "distancia media de atención (px)"), ("head_js", "diversidad entre cabezas (JS)")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, (key, title) in zip(axes.flat, metrics):
        for ci, (name, seeds) in enumerate(sorted(exps.items())):
            branches = next(iter(seeds.values()))["attention"]["branches"]
            for b in branches:
                # un color por variante; en HAM: dot ':' , aditiva '--', atención efectiva (fused) '-'
                ls = "-" if b == "fused" or len(branches) == 1 else (":" if b == "dot" else "--")
                series = [a["attention"]["branches"][b][key] for a in seeds.values()]
                _band(ax, range(1, len(series[0]) + 1), series, f"{short(ds, name)}:{b}", marker="o", ms=3,
                      ls=ls, color=colors[ci % len(colors)])
        ax.set(title=title, xlabel="capa")
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.grid(alpha=0.3)
    axes[0, 1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle(f"{ds}: atención por capa (media ± desv. entre semillas)")
    _save(fig, out / f"attn_layers_{ds}.png")

    pairs = [(n, s) for n, s in sorted(exps.items()) if next(iter(s.values()))["attention"]["js_between"]]
    if pairs:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for name, seeds in pairs:
            for pair in next(iter(seeds.values()))["attention"]["js_between"]:
                series = [a["attention"]["js_between"][pair] for a in seeds.values()]
                _band(ax, range(1, len(series[0]) + 1), series, f"{short(ds, name)}: {pair.replace('|', ' vs ')}",
                      marker="o", ms=3)
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.set(title=f"{ds}: divergencia JS entre ramas", xlabel="capa", ylabel="JS (0 = iguales, 1 = disjuntas)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
        _save(fig, out / f"attn_js_{ds}.png")

    contrib = [(n, s) for n, s in sorted(exps.items()) if next(iter(s.values()))["attention"].get("branch_contribution")]
    if contrib:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name, seeds in contrib:
            for b in next(iter(seeds.values()))["attention"]["branch_contribution"]:
                series = [a["attention"]["branch_contribution"][b] for a in seeds.values()]
                _band(ax, range(1, len(series[0]) + 1), series, f"{short(ds, name)}:{b}", marker="o", ms=3)
        ax.axhline(0.5, color="gray", lw=0.8, ls=":")
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.set(title=f"{ds}: contribución de cada rama (‖W_b·out_b‖ relativa)", xlabel="capa", ylabel="fracción")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        _save(fig, out / f"branch_contrib_{ds}.png")


def plot_reliability(ds, exps, out: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=1)
    for name, seeds in sorted(exps.items()):
        rel = [a["classification"]["reliability"] for a in seeds.values()]
        count = np.sum([r["count"] for r in rel], 0)
        conf = np.sum([np.array(r["conf"]) * r["count"] for r in rel], 0)
        acc = np.sum([np.array(r["acc"]) * r["count"] for r in rel], 0)
        nz = count > 0
        ece = statistics.mean(a["classification"]["ece"] for a in seeds.values())
        ax.plot(conf[nz] / count[nz], acc[nz] / count[nz], marker="o", ms=3,
                label=f"{short(ds, name)} (ECE {ece:.2f})")
    ax.set(title=f"{ds}: calibración", xlabel="confianza", ylabel="accuracy", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, out / f"reliability_{ds}.png")


def plot_perclass(ds, exps, ref, out: Path):
    ref_name = f"{ds}_{ref}"
    if ref_name not in exps:
        return
    ref_acc = np.mean([a["classification"]["per_class_acc"] for a in exps[ref_name].values()], 0)
    others = [(n, s) for n, s in sorted(exps.items()) if n != ref_name]
    n_cls, width = len(ref_acc), 0.8 / max(1, len(others))
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * n_cls * max(1, len(others)) / 2), 4.5))
    for i, (name, seeds) in enumerate(others):
        delta = np.mean([a["classification"]["per_class_acc"] for a in seeds.values()], 0) - ref_acc
        ax.bar(np.arange(n_cls) + i * width - 0.4 + width / 2, delta, width, label=short(ds, name))
    ax.axhline(0, color="black", lw=0.8)
    ax.set(title=f"{ds}: accuracy por clase − {ref}", xlabel="clase", ylabel="Δ accuracy (puntos)")
    ax.set_xticks(range(n_cls))
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    _save(fig, out / f"perclass_delta_{ds}.png")


def plot_deletion(ds, exps, out: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, seeds in sorted(exps.items()):
        x = next(iter(seeds.values()))["deletion"]["fractions"]
        auc = statistics.mean(a["deletion"]["auc_rollout"] for a in seeds.values())
        line = ax.plot(x, np.mean([a["deletion"]["rollout"] for a in seeds.values()], 0),
                       label=f"{short(ds, name)} (AUC {auc:.3f})")[0]
        ax.plot(x, np.mean([a["deletion"]["random"] for a in seeds.values()], 0),
                color=line.get_color(), ls=":", lw=0.8)
    ax.set(title=f"{ds}: deletion por rollout (punteado: orden aleatorio)",
           xlabel="fracción de parches borrados", ylabel="prob. de la clase predicha")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, out / f"deletion_{ds}.png")


def plot_robustness(ds, exps, out: Path):
    titles = {"noise": "ruido gaussiano (σ)", "blur": "desenfoque gaussiano (σ px)", "occlusion": "oclusión (fracción)"}
    fig, axes = plt.subplots(1, len(ROBUSTNESS), figsize=(5 * len(ROBUSTNESS), 4.2))
    for ax, (kind, sev) in zip(axes, ROBUSTNESS.items()):
        for name, seeds in sorted(exps.items()):
            series = [[a["robustness"]["clean"]] + [a["robustness"][f"{kind}_{s}"] for s in sev]
                      for a in seeds.values()]
            _band(ax, [0, *sev], series, short(ds, name), marker="o", ms=3)
        ax.set(title=titles[kind], xlabel="severidad", ylabel="accuracy (%)")
        ax.grid(alpha=0.3)
    axes[-1].legend(fontsize=8)
    fig.suptitle(f"{ds}: robustez")
    _save(fig, out / f"robustness_{ds}.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--out", type=Path, default=Path("."))
    p.add_argument("--ref", default="dot", help="variante de referencia: <dataset>_<ref>")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    groups = collect(args.runs)
    if not groups:
        print(f"No hay analysis.json bajo {args.runs} (correr python -m ham.analyze)")
        return
    write_metrics(metrics_table(groups), args.out / "metrics_extra.csv", args.out / "metrics_extra.tex")
    sig = significance(groups, args.ref)
    if sig:
        fields = list(dict.fromkeys(k for r in sig for k in r))
        with open(args.out / "significance.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(sig)
        print(f"{'dataset':<14}{'experimento':<28}{'semilla':>8}{'Δacc':>8}  {'IC 95%':<16}{'p':>8}")
        for r in sig:
            ci = f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]" if "ci_low" in r else ""
            pv = r.get("p_mcnemar", r.get("p_ttest_seeds"))
            print(f"{r['dataset']:<14}{short(r['dataset'], r['experiment']):<28}{str(r['seed']):>8}"
                  f"{r['diff']:>+8.2f}  {ci:<16}{pv:>8.3f}")
    for ds, exps in sorted(groups.items()):
        plot_attention(ds, exps, args.out)
        plot_reliability(ds, exps, args.out)
        plot_perclass(ds, exps, args.ref, args.out)
        plot_deletion(ds, exps, args.out)
        plot_robustness(ds, exps, args.out)


if __name__ == "__main__":
    main()
