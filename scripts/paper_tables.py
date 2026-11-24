"""Tablas LaTeX del artículo (``paper/tablas/*.tex``) a partir de los runs.

    uv run python scripts/paper_tables.py [--out paper/tablas]

Lee ``test.json``, ``metrics.json`` y ``analysis.json`` (``python -m ham.analyze``) de
``runs/``, ``runs_seeds/``, ``runs_corta/`` y ``runs_noaug/``. Todas las comparaciones son
pareadas por semilla (t de Student, IC 95 %, Wilcoxon con n >= 6) y Holm dentro de cada tabla;
las mismas convenciones que ``scripts/paired.py``. Además imprime las cifras que cita el texto
y que no salen en ninguna tabla (modo alto, comparaciones sin parear).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from paired import epoch_reaching, holm  # noqa: E402

ROOTS = ("runs", "runs_seeds")
THRESHOLD = 79.0  # val_acc: cerca del final de dot (79.9)
SEEDS = range(16)  # semillas de las tablas; las 16-47 de fase17 se analizan aparte (ieee_tables.tab_evidencia)
HIGH_MODE = 85.5  # test_acc que separa los dos modos de las variantes con local (sección 3.3)
HEADER = "% Generado por scripts/paper_tables.py a partir de los runs; no editar a mano.\n"
ENV = "table"  # entorno de las tablas; ieee_tables.py usa table* (ancho de página) salvo en ENV_BY_LABEL
ENV_BY_LABEL: dict[str, str] = {}


# --------------------------------------------------------------------------- datos

def run_dirs(name, dataset="cifar10", roots=ROOTS) -> dict[int, Path]:
    """``{semilla: directorio}`` de ``<root>/<dataset>/<dataset>_<name>/seed*`` con ``test.json``."""
    out = {}
    for root in roots:
        for f in sorted(Path(root, dataset, f"{dataset}_{name}").glob("seed*/test.json")):
            seed = int(f.parent.name.removeprefix("seed"))
            if seed in out:
                raise ValueError(f"Semilla {seed} de {name} repetida en {roots}")
            out[seed] = f.parent
    if not out:
        raise SystemExit(f"Sin runs de {dataset}_{name} en {roots}")
    return out


def read(d: Path, f: str):
    return json.loads((d / f).read_text())


class Variant:
    """Los runs de un experimento, con accesos cacheados a sus métricas."""

    def __init__(self, name, dataset="cifar10", roots=ROOTS, seeds=None):
        dirs = run_dirs(name, dataset, roots)
        seeds = SEEDS if seeds is None else seeds
        self.name, self.dirs = name, {s: d for s, d in dirs.items() if s in seeds}
        self.acc = {s: read(d, "test.json")["test_acc"] for s, d in self.dirs.items()}
        self.params = read(next(iter(self.dirs.values())), "test.json").get("num_params")
        self._metrics, self._analysis = {}, {}

    def metrics(self, s):
        if s not in self._metrics:
            self._metrics[s] = read(self.dirs[s], "metrics.json")
        return self._metrics[s]

    def analysis(self, s):
        if s not in self._analysis:
            self._analysis[s] = read(self.dirs[s], "analysis.json")
        return self._analysis[s]

    @property
    def seeds(self):
        return sorted(self.dirs)

    def accs(self):
        return np.array([self.acc[s] for s in self.seeds])

    def s_epoch(self):
        """Segundos por época, media de las semillas (sin la primera época, que carga workers)."""
        return statistics.mean(statistics.mean(r["time_s"] for r in self.metrics(s)[1:]) for s in self.seeds)

    def epochs_to(self, threshold=THRESHOLD):
        reached = [epoch_reaching(self.metrics(s), threshold) for s in self.seeds]
        return [e for e in reached if e is not None], len(reached)

    def mean_analysis(self, *keys):
        """Media sobre semillas de ``analysis[k0][k1]...``."""
        vals = []
        for s in self.seeds:
            x = self.analysis(s)
            for k in keys:
                x = x[k]
            vals.append(x)
        return np.mean(np.array(vals, dtype=float), axis=0)


def paired(a: Variant, b: Variant) -> dict:
    seeds = sorted(set(a.acc) & set(b.acc))
    d = np.array([a.acc[s] - b.acc[s] for s in seeds])
    n = len(d)
    half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return {
        "n": n, "delta": d.mean(), "lo": d.mean() - half, "hi": d.mean() + half, "wins": int((d > 0).sum()),
        "p": stats.ttest_1samp(d, 0).pvalue,
        "p_w": stats.wilcoxon(d).pvalue if n >= 6 and np.any(d != 0) else None,
    }


def with_holm(rows: list[dict]) -> list[dict]:
    for r, p in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = p
    return rows


# --------------------------------------------------------------------------- formato

def tt(name):
    return r"\texttt{" + name.replace("_", r"\_") + "}"


def signed(x, nd=2):
    return f"${x:+.{nd}f}$"


def ci(r):
    return f"{signed(r['delta'])} [{signed(r['lo'])}, {signed(r['hi'])}]"


def pfmt(p):
    if p is None:
        return "---"
    return r"$<0.001$" if p < 0.001 else f"{p:.3f}"


def pm(a):
    return f"{a.mean():.2f} $\\pm$ {a.std(ddof=1):.2f}"


def thousands(n):
    return f"{n:,}".replace(",", r"\,")


def wins(r):
    return f"{r['wins']}/{r['n']}"


def epochs_cell(v: Variant):
    reached, n = v.epochs_to()
    if not reached:
        return "---"
    cell = f"{statistics.mean(reached):.0f}"
    return cell if len(reached) == n else f"{cell} ({len(reached)}/{n})"


def table(path: Path, caption, label, cols, header, rows, note=None, size=r"\small"):
    """Tabla ``booktabs`` (reducida al ancho de línea si no cabe); ``rows``: listas de celdas o ``"mid"``."""
    env = ENV_BY_LABEL.get(label, ENV)
    lines = [HEADER, rf"\begin{{{env}}}[{'!t' if env.endswith('*') else 'tbp'}]", r"\centering", size,
             rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\begin{adjustbox}{max width=\linewidth}",
             rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(r"\midrule" if row == "mid" else " & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    if note:
        lines.append(rf"\par\smallskip\parbox{{\linewidth}}{{\footnotesize {note}}}")
    lines.append(rf"\end{{{env}}}")
    path.write_text("\n".join(lines) + "\n")
    print(f"escrito {path}")


# --------------------------------------------------------------------------- tablas

DESCRIPTION = {  # variante: (ramas por cabeza, fusión, término local)
    "dot": ("dot", "---", "---"),
    "dot_wide": ("dot (MLP 217)", "---", "---"),
    "additive": ("aditiva \\emph{pooling}", "---", "---"),
    "additive_free": ("aditiva por pares sin parámetros", "---", "---"),
    "ham_concat": ("dot + aditiva \\emph{pooling}", "concat", "---"),
    "ham_pairwise": ("dot + aditiva por pares", "concat", "---"),
    "ham_weighted": ("dot + aditiva libre", "scores, $\\lambda=0.85$", "---"),
    "ham_weighted_learn": ("dot + aditiva libre", "scores, $\\lambda$ aprendido", "---"),
    "ham_sum": ("dot + aditiva libre", "scores, suma", "---"),
    "ham_zscore": ("dot + aditiva libre", "scores estandarizados", "---"),
    "ham_wout": ("dot + aditiva \\emph{pooling}", "salidas, $\\lambda$ aprendido", "---"),
    "ham_gated": ("dot + aditiva \\emph{pooling}", "salidas, compuerta", "---"),
    "ham_weighted_learn_out": ("dot + aditiva libre", "salidas, $\\lambda$ aprendido", "---"),
    "ham_heads": ("2 dot, 2 aditiva \\emph{pooling}", "entre cabezas", "---"),
    "dot_local_k1": ("dot", "---", "$1\\times1$"),
    "dot_local": ("dot", "---", "$3\\times3$"),
    "dot_local_k5": ("dot", "---", "$5\\times5$"),
    "dot_local_k7": ("dot", "---", "$7\\times7$"),
    "dot_gate": ("dot", "+ compuerta", "---"),
    "dot_global": ("dot", "+ contexto global", "---"),
    "dot_wide_local_k5": ("dot (MLP 217)", "---", "$5\\times5$"),
    "ham_concat_local": ("dot + aditiva \\emph{pooling}", "concat", "$3\\times3$"),
    "ham_concat_local_k5": ("dot + aditiva \\emph{pooling}", "concat", "$5\\times5$"),
}

FUSIONS = ["additive", "additive_free", "ham_concat", "ham_pairwise", "ham_weighted", "ham_weighted_learn",
           "ham_sum", "ham_zscore", "ham_wout", "ham_gated", "ham_weighted_learn_out", "ham_heads", "dot_wide"]
MAIN = ["dot", "dot_wide", "ham_concat", "dot_local", "dot_local_k5", "ham_concat_local_k5"]
ABLATION = ["dot_local_k1", "dot_local", "dot_local_k5", "dot_local_k7", "dot_gate", "dot_global",
            "dot_local_gate", "dot_local_global", "dot_wide_local_k5", "ham_concat_local", "ham_concat_local_k5"]
FIRST5 = set(range(5))


def tab_variantes(out):
    groups = [["dot", "dot_wide", "additive"], FUSIONS[2:-1], ABLATION[:4] + ["dot_gate", "dot_global"] + ABLATION[-3:]]
    rows = []
    for gi, group in enumerate(groups):
        if gi:
            rows.append("mid")
        for name in group:
            v = Variant(name, seeds=FIRST5)
            branches, fusion, local = DESCRIPTION[name]
            rows.append([tt(name), branches, fusion, local, thousands(v.params), f"{v.s_epoch():.2f}"])
    table(out / "variantes.tex",
          "Variantes evaluadas en CIFAR-10. Parámetros totales del modelo y segundos por época de "
          "entrenamiento (media de las semillas 0--4, RTX 5080, sin otros procesos en la GPU).",
          "tab:variantes", "llllrr",
          ["Variante", "Ramas por cabeza", "Fusión", "Término local", "Parámetros", "s/época"], rows,
          note="concat: \\texttt{concat\\_linear}. scores: mezcla antes del \\emph{softmax}; salidas: "
               "mezcla convexa de $A_b v$ después del \\emph{softmax}. Aditiva libre: "
               "$e_{ij}=\\sum_d \\tanh(q_i+k_j)_d$. Nombres: \\texttt{\\_wide}, MLP de 217 (control de igual capacidad); "
               "\\texttt{\\_local}, término local $3\\times3$, y \\texttt{\\_local\\_k}$k$, $k\\times k$. "
               "\\texttt{ham\\_concat\\_local\\_k5} es la tríada y \\texttt{dot\\_wide\\_local\\_k5}, su control de "
               "igual capacidad.", size=r"\footnotesize")


def tab_fusiones(out):
    dot = Variant("dot", seeds=FIRST5)
    rows = with_holm([{"name": n, "v": Variant(n, seeds=FIRST5)} | paired(Variant(n, seeds=FIRST5), dot) for n in FUSIONS])
    body = [[tt(r["name"]), pm(r["v"].accs()), ci(r), wins(r), pfmt(r["p"]), pfmt(r["p_holm"])]
            for r in sorted(rows, key=lambda r: -r["delta"])]
    table(out / "fusiones.tex",
          f"Fusiones de dos mecanismos de atención en CIFAR-10 (400 épocas, semillas 0--4). "
          f"$\\Delta$: diferencia pareada por semilla contra \\texttt{{dot}} ({pm(dot.accs())}), "
          f"con IC 95~\\% de la t de Student; Holm sobre las {len(rows)} comparaciones.",
          "tab:fusiones", "lccccc",
          ["Variante", "Exactitud (\\%)", "$\\Delta$ [IC 95~\\%]", "$\\Delta>0$", "$p$", "$p$ Holm"], body)


def layer_range(x, nd=2):
    x = np.asarray(x)
    lo, hi = x.min(), x.max()
    return f"{lo:.{nd}f}" if f"{lo:.{nd}f}" == f"{hi:.{nd}f}" else f"{lo:.{nd}f}--{hi:.{nd}f}"


def tab_colapso(out):
    """Uso de las ramas por capa: JS entre la atención aplicada y cada rama, entropía, contribución."""
    names = ["ham_weighted", "ham_weighted_learn", "ham_sum", "ham_weighted_scaled", "ham_zscore",
             "ham_weighted_learn_out", "ham_wout", "ham_gated", "ham_concat"]
    rows = []
    for name in names:
        v = Variant(name, seeds=FIRST5)
        a = v.analysis(v.seeds[0])["attention"]
        other = next(b for b in a["branches"] if b not in ("dot", "fused"))
        js = a["js_between"]
        jf_dot = layer_range(v.mean_analysis("attention", "js_between", "dot|fused"), 3) if "dot|fused" in js else "---"
        jf_oth = layer_range(v.mean_analysis("attention", "js_between", f"{other}|fused"), 3) if f"{other}|fused" in js else "---"
        h_dot = layer_range(v.mean_analysis("attention", "branches", "dot", "entropy_all"))
        h_oth = layer_range(v.mean_analysis("attention", "branches", other, "entropy_all"))
        contrib = (layer_range(v.mean_analysis("attention", "branch_contribution", "dot"))
                   if a.get("branch_contribution") else "---")
        rows.append([tt(name), jf_dot, jf_oth, h_dot, h_oth, contrib])
    table(out / "colapso.tex",
          "Uso de las dos ramas en las fusiones de CIFAR-10 (media de las semillas 0--4; rango sobre las 4 capas). "
          "JS: divergencia de Jensen--Shannon (base 2) entre la atención que se aplica a $v$ y cada rama. "
          "$H$: entropía normalizada de cada rama (1 = uniforme). Contribución: fracción de la norma de la "
          "salida que aporta \\texttt{dot}, en fusiones de salidas.",
          "tab:colapso", "lccccc",
          ["Variante", "JS(fus., dot)", "JS(fus., aditiva)", "$H$ dot", "$H$ aditiva", "Contrib. dot"], rows,
          size=r"\footnotesize")


def tab_principal(out):
    dot = Variant("dot")
    variants = {n: Variant(n) for n in MAIN}
    comps = with_holm([{"name": n} | paired(variants[n], dot) for n in MAIN[1:]])
    rows = [[tt("dot"), thousands(dot.params), pm(dot.accs()), "---", "---", "---", epochs_cell(dot), f"{dot.s_epoch():.2f}"]]
    for r in comps:
        v = variants[r["name"]]
        rows.append([tt(r["name"]), thousands(v.params), pm(v.accs()), ci(r), wins(r), pfmt(r["p_holm"]),
                     epochs_cell(v), f"{v.s_epoch():.2f}"])
    table(out / "principal.tex",
          f"Resultados principales en CIFAR-10 con 16 semillas (0--15), 400 épocas. $\\Delta$: diferencia "
          f"pareada contra \\texttt{{dot}}; $p$ con Holm sobre {len(comps)} comparaciones. Época $\\geq${THRESHOLD:.0f}: "
          f"primera época con exactitud de validación $\\geq${THRESHOLD:.0f}~\\%, media de las semillas que la alcanzan.",
          "tab:principal", "lrcccccr",
          ["Variante", "Parám.", "Exactitud (\\%)", "$\\Delta$ [IC 95~\\%]", "$\\Delta>0$", "$p$ Holm",
           f"Época $\\geq${THRESHOLD:.0f}", "s/época"], rows, size=r"\footnotesize")


CONTRASTS = [  # (a, b, qué mide)
    ("ham_concat", "dot_wide", "rama global sin localidad, a igual capacidad"),
    ("dot_local", "dot_wide", "término local $3\\times3$ contra capacidad"),
    ("dot_local_gate", "dot_local", "compuerta sobre el término local"),
    ("dot_local_k5", "dot_local", "vecindad $5\\times5$ contra $3\\times3$"),
    ("ham_concat_local_k5", "dot_local_k5", "rama global con localidad $5\\times5$"),
]


def tab_contrastes(out):
    rows = with_holm([{"a": a, "b": b, "what": w} | paired(Variant(a), Variant(b)) for a, b, w in CONTRASTS])
    body = [[f"{tt(r['a'])} $-$ {tt(r['b'])}", r["what"], ci(r), wins(r), pfmt(r["p"]), pfmt(r["p_holm"]), pfmt(r["p_w"])]
            for r in rows]
    table(out / "contrastes.tex",
          f"Contrastes planeados en CIFAR-10 con 16 semillas, pareados por semilla. $p$: t pareada; "
          f"Holm sobre los {len(rows)} contrastes; Wilcoxon de rangos con signo.",
          "tab:contrastes", r">{\raggedright\arraybackslash}l>{\raggedright\arraybackslash}p{3.3cm}ccccc",
          ["Contraste", "Qué mide", "$\\Delta$ [IC 95~\\%]", "$\\Delta>0$", "$p$", "$p$ Holm", "$p$ Wilcoxon"], body,
          size=r"\footnotesize")


def tab_ablacion(out):
    dot, local = Variant("dot", seeds=FIRST5), Variant("dot_local", seeds=FIRST5)
    rows = []
    for name in ABLATION:
        v = Variant(name, seeds=FIRST5)
        vs_dot = paired(v, dot)
        vs_local = "---" if name == "dot_local" else f"{ci(paired(v, local))} ({wins(paired(v, local))})"
        rows.append([tt(name), thousands(v.params), pm(v.accs()), ci(vs_dot), vs_local, epochs_cell(v), f"{v.s_epoch():.2f}"])
    table(out / "ablacion.tex",
          f"Ablación de los términos de salida en CIFAR-10 (semillas 0--4, 400 épocas). Diferencias pareadas "
          f"contra \\texttt{{dot}} ({pm(dot.accs())}) y contra \\texttt{{dot\\_local}} ($3\\times3$), con el número "
          f"de semillas a favor entre paréntesis.",
          "tab:ablacion", "lrcccrr",
          ["Variante", "Parám.", "Exactitud (\\%)", "$\\Delta$ vs \\texttt{dot}", "$\\Delta$ vs \\texttt{dot\\_local}",
           f"Época $\\geq${THRESHOLD:.0f}", "s/época"], rows, size=r"\footnotesize")


def tab_eficiencia(out):
    rows = []
    for name in MAIN + ["dot_local_gate"]:
        v = Variant(name)
        reached, n = v.epochs_to()
        time_to = [epoch_reaching(v.metrics(s), THRESHOLD) * statistics.mean(r["time_s"] for r in v.metrics(s)[1:])
                   for s in v.seeds if epoch_reaching(v.metrics(s), THRESHOLD) is not None]
        try:
            short = Variant(name, roots=("runs_corta",))
            short_cell = pm(short.accs())
        except SystemExit:
            short_cell = "---"
        rows.append([tt(name), epochs_cell(v), f"{statistics.mean(time_to):.0f}" if time_to else "---",
                     short_cell, pm(v.accs())])
    table(out / "eficiencia.tex",
          f"Eficiencia de entrenamiento en CIFAR-10. Época y tiempo de GPU hasta una exactitud de validación de "
          f"{THRESHOLD:.0f}~\\% (16 semillas; entre paréntesis, las que la alcanzan si no son todas). Exactitud de "
          f"test entrenando 100 épocas (\\emph{{warmup}} de 3; semillas 0--4) y 400 épocas (16 semillas).",
          "tab:eficiencia", "lcccc",
          ["Variante", f"Época $\\geq${THRESHOLD:.0f}", "Tiempo (s)", "100 épocas", "400 épocas"], rows)


PV_NAMES = ["additive_b128", "additive_free_b128", "ham_concat_b128", "ham_sum_b128", "dot_local_b128", "dot_local_k5_b128"]


def tab_plantvillage(out):
    names = PV_NAMES
    dot = Variant("dot_b128", "plantvillage")
    comps = with_holm([{"name": n, "v": Variant(n, "plantvillage")} | paired(Variant(n, "plantvillage"), dot) for n in names])

    def extra(v):
        r = lambda k: v.mean_analysis("robustness", k)  # noqa: E731
        return [f"{v.s_epoch():.1f}", f"{v.mean_analysis('classification', 'ece'):.2f}", f"{r('noise_0.1'):.1f}",
                f"{r('blur_1.0'):.1f}", f"{r('occlusion_0.25'):.1f}"]

    rows = [[tt("dot"), pm(dot.accs()), "---", "---", "---"] + extra(dot)]
    for r in comps:
        rows.append([tt(r["name"].removesuffix("_b128")), pm(r["v"].accs()), ci(r), wins(r), pfmt(r["p_holm"])] + extra(r["v"]))
    table(out / "plantvillage.tex",
          f"PlantVillage (38 clases, $64\\times64$, 50 épocas, lote 128, semillas 0--4). $\\Delta$ pareada contra "
          f"\\texttt{{dot}}, Holm sobre {len(comps)} comparaciones. ECE en \\%. Robustez: exactitud con ruido gaussiano "
          f"$\\sigma=0.1$, desenfoque $\\sigma=1$~px y oclusión del 25~\\% de los parches.",
          "tab:plantvillage", "lccccrrrrr",
          ["Variante", "Exactitud (\\%)", "$\\Delta$ [IC 95~\\%]", "$\\Delta>0$", "$p$ Holm", "s/época", "ECE",
           "Ruido", "Blur", "Oclusión"], rows, size=r"\footnotesize")


def tab_calibracion(out):
    names = ["dot", "dot_wide", "ham_concat", "dot_local_k1", "dot_local", "dot_local_k5", "dot_local_k7",
             "dot_wide_local_k5", "ham_concat_local_k5"]
    rows = []
    for name in names:
        v = Variant(name)
        c = lambda k: v.mean_analysis("classification", k)  # noqa: E731
        r = lambda k: v.mean_analysis("robustness", k)  # noqa: E731
        rows.append([tt(name), str(len(v.seeds)), f"{c('ece'):.2f}", f"{c('nll'):.3f}", f"{c('f1_macro'):.2f}",
                     f"{r('noise_0.05'):.1f}", f"{r('noise_0.1'):.1f}", f"{r('blur_1.0'):.1f}", f"{r('occlusion_0.25'):.1f}"])
    table(out / "calibracion.tex",
          "Calibración y robustez en CIFAR-10 (media de las semillas disponibles, $n$). ECE con 15 intervalos, en \\%. "
          "Robustez: exactitud (\\%) sobre el test completo con ruido gaussiano ($\\sigma$ sobre la imagen en $[0,1]$), "
          "desenfoque gaussiano ($\\sigma=1$~px) y oclusión aleatoria del 25~\\% de los parches.",
          "tab:calibracion", "lcccccccc",
          ["Variante", "$n$", "ECE", "NLL", "F1 macro", "Ruido 0.05", "Ruido 0.1", "Blur 1.0", "Oclusión 25~\\%"], rows,
          size=r"\footnotesize")


def tab_otros(out):
    blocks = [("CIFAR-10 sin aumento", "cifar10", ("runs_noaug",), ["additive", "ham_concat", "ham_weighted", "ham_sum"]),
              ("Food-101", "food101", ("runs",), ["additive", "ham_concat"])]
    rows = []
    for bi, (title, ds, roots, names) in enumerate(blocks):
        if bi:
            rows.append("mid")
        dot = Variant("dot", ds, roots)
        rows.append([title, tt("dot"), pm(dot.accs()), "---", "---", "---"])
        comps = with_holm([{"name": n, "v": Variant(n, ds, roots)} | paired(Variant(n, ds, roots), dot) for n in names])
        rows += [["", tt(r["name"]), pm(r["v"].accs()), ci(r), wins(r), pfmt(r["p_holm"])] for r in comps]
    table(out / "otros.tex",
          "Fusiones en otros regímenes (semillas 0--4). CIFAR-10 sin aumento de datos (400 épocas) y Food-101 "
          "($192\\times192$, parches de 8, 50 épocas, lote 128). Holm dentro de cada bloque.",
          "tab:otros", "llcccc",
          ["Régimen", "Variante", "Exactitud (\\%)", "$\\Delta$ [IC 95~\\%]", "$\\Delta>0$", "$p$ Holm"], rows)


def cifras():
    """Cifras del texto que no están en las tablas."""
    print("\n== modo alto (test >= %.1f), 16 semillas" % HIGH_MODE)
    for name in ["dot", "dot_wide", "ham_concat", "dot_local", "dot_local_gate", "dot_local_k5", "ham_concat_local_k5"]:
        a = Variant(name).accs()
        print(f"  {name:22} {int((a >= HIGH_MODE).sum())}/{len(a)}  mín {a.min():.2f} máx {a.max():.2f}")
    for name in ["dot_local_k1", "dot_local_k7", "dot_wide_local_k5", "ham_concat_local"]:
        a = Variant(name).accs()
        print(f"  {name:22} {int((a >= HIGH_MODE).sum())}/{len(a)} ({len(a)} semillas)")
    k3, k5, h = Variant("dot_local").accs(), Variant("dot_local_k5").accs(), Variant("ham_concat_local_k5").accs()
    f = stats.fisher_exact([[(k5 >= HIGH_MODE).sum(), (k5 < HIGH_MODE).sum()], [(k3 >= HIGH_MODE).sum(), (k3 < HIGH_MODE).sum()]])
    print(f"  Fisher k5 vs k3 (modo alto): p = {f.pvalue:.3f}")
    lo5, lo3 = k5[k5 < HIGH_MODE], k3[k3 < HIGH_MODE]
    print(f"  modo bajo: k5 {lo5.mean():.2f} (n={len(lo5)}) vs k3 {lo3.mean():.2f} (n={len(lo3)}), "
          f"Mann-Whitney p = {stats.mannwhitneyu(lo5, lo3).pvalue:.3f}")
    hi_h, hi_5 = h[h >= HIGH_MODE], k5[k5 >= HIGH_MODE]
    print(f"  modo alto: ham_concat_local_k5 {hi_h.mean():.2f} (n={len(hi_h)}) vs k5 {hi_5.mean():.2f} (n={len(hi_5)}), "
          f"Mann-Whitney p = {stats.mannwhitneyu(hi_h, hi_5).pvalue:.3f}")
    w = Variant("dot_wide_local_k5").accs()
    print(f"  ham_concat_local_k5 (16) vs dot_wide_local_k5 ({len(w)}) sin parear: {h.mean() - w.mean():+.2f}, "
          f"Mann-Whitney p = {stats.mannwhitneyu(h, w).pvalue:.3f}")
    print(f"  dot_wide_local_k5 ({len(w)}) vs dot_local_k5 (16) sin parear: {w.mean() - k5.mean():+.2f}, "
          f"Mann-Whitney p = {stats.mannwhitneyu(w, k5).pvalue:.3f}")
    print("\n== train - val al final (16 semillas), pérdida de entrenamiento final")
    for name in MAIN:
        v = Variant(name)
        gap = statistics.mean(v.metrics(s)[-1]["train_acc"] - v.metrics(s)[-1]["val_acc"] for s in v.seeds)
        loss = statistics.mean(v.metrics(s)[-1]["train_loss"] for s in v.seeds)
        print(f"  {name:22} brecha {gap:+.2f}  pérdida {loss:.3f}")
    print("\n== 100 épocas de dot_local contra dot de 400 (semillas 0-4)")
    for name in ["dot_local", "dot_local_gate", "dot_local_global"]:
        print(f"  {name:22} {ci(paired(Variant(name, roots=('runs_corta',)), Variant('dot', seeds=FIRST5)))}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("paper/tablas"))
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for build in (tab_variantes, tab_fusiones, tab_colapso, tab_principal, tab_contrastes, tab_ablacion,
                  tab_eficiencia, tab_plantvillage, tab_calibracion, tab_otros):
        build(args.out)
    cifras()


if __name__ == "__main__":
    main()
