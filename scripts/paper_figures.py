"""Figuras del artículo (``paper/figuras/``) y la tabla de similitud de mapas (``paper/tablas/mapas.tex``).

    uv run python scripts/paper_figures.py [--out paper] [--device auto]
    uv run python scripts/paper_figures.py --lang en --out paper/ieee_en   
    curvas.pdf exactitud de validación por época (media ± desv. de 16 semillas)
    semillas.pdf exactitud de test de cada semilla: los dos modos de las variantes con local kernel.pdf Δ contra dot según el tamaño del kernel local (semillas 0-4) mecanismo.pdf  entropía, distancia y diversidad de la atención por capa, y peso del término local mapas.png rollout del CLS con dos semillas de dot como control, misma escala por fila mapas.tex JS entre mapas de rollout: entre semillas de una variante y contra dot

Los colores siguen a la variante en todas las figuras (paleta categórica validada con la skill de visualización: azul, naranja y aqua; grises para las referencias).
"""
from __future__ import annotations

import argparse
import itertools
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from paper_tables import FIRST5, HEADER, HIGH_MODE, THRESHOLD, Variant, paired, tt  # noqa: E402

from ham import analysis as A  # noqa: E402
from ham.train import resolve_device  # noqa: E402
from ham.utils.io import load_run  # noqa: E402
from ham.visualize import TestImages, cls_attention_maps, denormalize, effective_map, forward  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {  # variante: (color, estilo de línea, etiqueta)
    "dot": ("#52514e", "-", "dot"),
    "dot_wide": ("#a8a7a1", "--", "dot_wide"),
    "dot_local_k1": ("#a8a7a1", ":", "dot_local_k1"),
    "dot_local": ("#2a78d6", "-", "dot_local (3×3)"),
    "dot_local_k5": ("#eb6834", "-", "dot_local_k5"),
    "ham_concat_local_k5": ("#1baf7a", "-", "ham_concat_local_k5"),
}

LANG = "es"
EN = {  # textos de las figuras en inglés (--lang en)
    "Época": "Epoch", "Exactitud de validación (%)": "Validation accuracy (%)", "Exactitud de test (%)": "Test accuracy (%)",
    "modo alto": "high mode", "sin\nlocal": "no\nlocal", "Término local (kernel)": "Local term (kernel)",
    "Δ exactitud vs dot (puntos)": "Δ accuracy vs dot (points)", "Entropía normalizada": "Normalized entropy",
    "Distancia media (px)": "Mean distance (px)", "JS entre cabezas": "JS between heads",
    "Peso del término local": "Weight of the local term", "Capa": "Layer", "dot (semilla 0)": "dot (seed 0)",
    "dot (semilla 1)": "dot (seed 1)", "Imagen": "Image",
}


def T(text):
    return EN.get(text, text) if LANG == "en" else text


plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"escrito {path}")


def curvas(out):
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for name in ["dot", "dot_wide", "dot_local", "dot_local_k5", "ham_concat_local_k5"]:
        v = Variant(name)
        acc = np.array([[r["val_acc"] for r in v.metrics(s)] for s in v.seeds])
        mean, sd = acc.mean(0), acc.std(0)
        epochs = np.arange(1, acc.shape[1] + 1)
        color, ls, label = STYLE[name]
        ax.fill_between(epochs, mean - sd, mean + sd, color=color, alpha=0.15, linewidth=0)
        ax.plot(epochs, mean, color=color, ls=ls, lw=1.5, label=label)
    ax.axhline(THRESHOLD, color=INK2, lw=0.8, ls=":")
    ax.text(8, THRESHOLD + 0.4, f"{THRESHOLD:.0f} %", color=INK2, fontsize=7)
    ax.set_xlim(0, 400)
    ax.set_ylim(50, 89)
    ax.set_xlabel(T("Época"))
    ax.set_ylabel(T("Exactitud de validación (%)"))
    ax.legend(loc="lower right", frameon=False, handlelength=2.2)
    save(fig, out / "curvas.pdf")


def semillas(out):
    names = ["dot", "dot_wide", "ham_concat", "dot_local", "dot_local_gate", "dot_local_k5", "ham_concat_local_k5"]
    fig, ax = plt.subplots(figsize=(6.8, 2.4))
    rng = np.random.default_rng(0)
    for i, name in enumerate(names):
        a = Variant(name).accs()
        color = STYLE.get(name, ("#a8a7a1",))[0]
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(a)), a, s=14, color=color, edgecolor="white", linewidth=0.5, zorder=3)
        ax.plot([i - 0.22, i + 0.22], [a.mean()] * 2, color=INK, lw=1.2, zorder=4)
        ax.text(i, a.max() + 0.35, f"{int((a >= HIGH_MODE).sum())}/{len(a)}", ha="center", fontsize=6.5, color=INK2)
    ax.axhline(HIGH_MODE, color=INK2, lw=0.8, ls="--")
    ax.text(-0.45, HIGH_MODE + 0.15, f"{T('modo alto')} ($\\geq${HIGH_MODE})", ha="left", fontsize=6.5, color=INK2)
    ax.set_xticks(range(len(names)), names, fontsize=6.5, rotation=15, ha="right")
    ax.set_xlim(-0.5, len(names) - 0.5)
    ax.set_ylabel(T("Exactitud de test (%)"))
    ax.grid(axis="x", visible=False)
    save(fig, out / "semillas.pdf")


def kernel(out):
    dot = Variant("dot", seeds=FIRST5)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    families = [("dot + local", ["dot_local_k1", "dot_local", "dot_local_k5", "dot_local_k7"], [1, 3, 5, 7], STYLE["dot_local"][0], -0.12),
                ("ham_concat + local", ["ham_concat", "ham_concat_local", "ham_concat_local_k5"], [0, 3, 5], STYLE["ham_concat_local_k5"][0], 0.12)]
    for label, names, ks, color, dx in families:
        rs = [paired(Variant(n, seeds=FIRST5), dot) for n in names]
        x = np.array(ks) + dx
        y = np.array([r["delta"] for r in rs])
        err = np.array([[r["delta"] - r["lo"] for r in rs], [r["hi"] - r["delta"] for r in rs]])
        ax.errorbar(x, y, yerr=err, color=color, marker="o", ms=4.5, lw=1.5, capsize=2, label=label,
                    markeredgecolor="white", markeredgewidth=0.6)
    wide = paired(Variant("dot_wide", seeds=FIRST5), dot)["delta"]
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axhline(wide, color=STYLE["dot_wide"][0], lw=0.8, ls="--")
    ax.text(7.4, wide - 0.45, "dot_wide", fontsize=6.5, color=INK2, ha="right")
    ax.set_xticks([0, 1, 3, 5, 7], [T("sin\nlocal"), "1×1", "3×3", "5×5", "7×7"])
    ax.set_xlabel(T("Término local (kernel)"))
    ax.set_ylabel(T("Δ exactitud vs dot (puntos)"))
    ax.legend(loc="upper left", frameon=False)
    save(fig, out / "kernel.pdf")


def mecanismo(out):
    names = ["dot", "dot_local_k1", "dot_local", "dot_local_k5", "ham_concat_local_k5"]
    panels = [("entropy_all", T("Entropía normalizada")), ("distance_px", T("Distancia media (px)")),
              ("head_js", T("JS entre cabezas"))]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9))
    layers = np.arange(1, 5)
    for name in names:
        v = Variant(name)
        color, ls, label = STYLE[name]
        for ax, (key, title) in zip(axes, panels):
            ax.plot(layers, v.mean_analysis("attention", "branches", "dot", key), color=color, ls=ls, lw=1.5,
                    marker="o", ms=3.5, label=label)
            ax.set_title(title)
        if name != "dot":
            contrib = v.mean_analysis("attention", "enrich_contribution", "terms", "local")
            axes[3].plot(layers, contrib, color=color, ls=ls, lw=1.5, marker="o", ms=3.5)
    axes[3].set_title(T("Peso del término local"))
    axes[3].set_ylim(0, 0.7)
    for ax in axes:
        ax.set_xticks(layers)
        ax.set_xlabel(T("Capa"))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, out / "mecanismo.pdf")


@torch.no_grad()
def rollout_maps(run_dir, images, device):
    cfg, model = load_run(run_dir)
    _, attentions = forward(model.to(device), images)
    maps = effective_map(cls_attention_maps(attentions, cfg.model.attention.branches, "rollout"))
    model.cpu()
    return maps / maps.sum(-1, keepdim=True)


def mapas(out_fig, out_tab, device, n_images=500, n_show=6):
    variants = ["dot", "dot_local", "dot_local_k5", "ham_concat_local_k5"]
    dirs = {v: Variant(v, seeds=FIRST5).dirs for v in variants}
    cfg0, _ = load_run(dirs["dot"][0])
    test = TestImages(cfg0, device)
    idx = torch.randperm(len(test), generator=torch.Generator().manual_seed(42))[:n_images].tolist()
    images, labels = test.get(idx)
    maps = {(v, s): rollout_maps(d, images, device) for v in variants for s, d in dirs[v].items()}

    js = lambda a, b: A.js_divergence(a, b).mean().item()  # noqa: E731
    rows = []
    within_dot = statistics.mean(js(maps["dot", i], maps["dot", j]) for i, j in itertools.combinations(range(5), 2))
    for v in variants:
        within = statistics.mean(js(maps[v, i], maps[v, j]) for i, j in itertools.combinations(range(5), 2))
        vs_dot = "---" if v == "dot" else f"{statistics.mean(js(maps[v, s], maps['dot', s]) for s in range(5)):.4f}"
        ratio = "---" if v == "dot" else f"{float(vs_dot) / within_dot:.2f}"
        rows.append(f"{tt(v)} & {within:.4f} & {vs_dot} & {ratio} \\\\")
    if out_tab is not None:
        out_tab.write_text(HEADER + "\n".join([
            r"\begin{table}[tbp]", r"\centering", r"\small",
            rf"\caption{{Divergencia de Jensen--Shannon entre mapas de \emph{{rollout}} del CLS en CIFAR-10 "
            rf"({n_images} imágenes de test, semillas 0--4). Entre semillas: media sobre los 10 pares de semillas de la "
            rf"variante. Contra \texttt{{dot}}: misma semilla. Razón: contra \texttt{{dot}} dividido entre la variación "
            rf"entre semillas de \texttt{{dot}}.}}",
            r"\label{tab:mapas}", r"\begin{adjustbox}{max width=\columnwidth}", r"\begin{tabular}{lccc}", r"\toprule",
            r"Variante & JS entre semillas & JS contra \texttt{dot} & Razón \\", r"\midrule", *rows,
            r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]) + "\n")
        print(f"escrito {out_tab}")

    columns = [(T("dot (semilla 0)"), maps["dot", 0]), (T("dot (semilla 1)"), maps["dot", 1]),
               ("dot_local_k5", maps["dot_local_k5", 0]), ("ham_concat_local_k5", maps["ham_concat_local_k5", 0])]
    shown = denormalize(images[:n_show], test.norm).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    side = shown.shape[1]
    fig, axes = plt.subplots(n_show, 1 + len(columns), figsize=(1.25 * (1 + len(columns)), 1.3 * n_show))
    for i in range(n_show):
        row = [m[i].view(1, 1, *(2 * [int(m.size(-1) ** 0.5)])) for _, m in columns]
        row = [torch.nn.functional.interpolate(m.float(), size=(side, side), mode="bilinear", align_corners=False)[0, 0].cpu()
               for m in row]
        vmin, vmax = min(m.min() for m in row).item(), max(m.max() for m in row).item()  # misma escala en la fila
        axes[i, 0].imshow(shown[i])
        axes[i, 0].set_ylabel(test.name(labels[i]), fontsize=7)
        for j, m in enumerate(row, start=1):
            axes[i, j].imshow(shown[i])
            axes[i, j].imshow(m.numpy(), cmap="viridis", alpha=0.6, vmin=vmin, vmax=vmax)
            if i == 0:
                axes[i, j].set_title(columns[j - 1][0], fontsize=6.5)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
    axes[0, 0].set_title(T("Imagen"), fontsize=6.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_fig / "mapas.png", dpi=220)
    plt.close(fig)
    print(f"escrito {out_fig / 'mapas.png'}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("paper"))
    p.add_argument("--device", default="auto")
    p.add_argument("--lang", choices=["es", "en"], default="es")
    p.add_argument("--sin-mapas", action="store_true",
                   help="omitir mapas.png y mapas.tex, que necesitan los pesos (best.pt) de los runs")
    args = p.parse_args(argv)
    global LANG
    LANG = args.lang
    figs, tabs = args.out / "figuras", args.out / "tablas"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    curvas(figs)
    semillas(figs)
    kernel(figs)
    mecanismo(figs)
    if not args.sin_mapas:
        mapas(figs, None if LANG == "en" else tabs / "mapas.tex", resolve_device(args.device))


if __name__ == "__main__":
    main()
