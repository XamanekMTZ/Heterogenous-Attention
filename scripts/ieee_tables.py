"""Tablas del artículo en formato IEEE (``paper/ieee/tablas/``) a partir de los runs.

    uv run python scripts/ieee_tables.py [--out paper/ieee/tablas]

Mismas tablas que ``paper_tables.py`` (y el presupuesto de parámetros y los criterios de
``propuesta_tables.py``), con dos cambios:

- a dos columnas: ``table*`` (ancho de página) salvo las tablas angostas de ``NARROW``;
- los resultados de las fases 14_extra, 15 y 16 que el suplemento (``paper/suplemento``, sección S5)
  propone incorporar: el control de capacidad ``dot_wide_local_k5`` con 16 semillas en la tabla
  principal y en los contrastes, y la atención aditiva con término local en PlantVillage.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import paper_tables as pt  # noqa: E402
import propuesta_tables  # noqa: E402

NARROW: set[str] = set()  # tablas a una columna; todas las actuales quedan ilegibles a ese ancho
ALL = range(1000)  # todas las semillas disponibles (fase17 lleva la tríada y su control a 48)

# (componente de la tríada, a, b, métrica, conjunto, semillas[, "primario"]): qué aporta cada sesgo y con qué evidencia.
# Las filas "primario" son el contraste declarado antes de correr fase17 (una prueba única, fuera de Holm).
EVIDENCIA = [
    ("Vecindad $5\\times5$, a igual capacidad", "dot_local_k5", "dot_wide", "test_acc", "cifar10", None),
    ("Proyección por cabeza de la fusión (calibración)", "ham_dotdot", "dot", "ece", "cifar10", None),
    ("Rama \\emph{pooling} sin vecindad, contra la proyección", "ham_concat", "ham_dotdot", "test_acc", "cifar10", None),
    ("Rama \\emph{pooling} sin vecindad, a igual capacidad", "ham_concat", "dot_wide", "test_acc", "cifar10", None),
    ("Rama \\emph{pooling} sobre la vecindad", "ham_concat_local_k5", "dot_local_k5", "test_acc", "cifar10", None),
    ("Tríada, a igual capacidad (contraste primario)", "ham_concat_local_k5", "dot_wide_local_k5", "test_acc",
     "cifar10", ALL, "primario"),
    ("Tríada, a igual capacidad (réplica: sólo semillas 16--47)", "ham_concat_local_k5", "dot_wide_local_k5",
     "test_acc", "cifar10", range(16, 48), "primario"),
    ("Tríada, a igual capacidad (calibración)", "ham_concat_local_k5", "dot_wide_local_k5", "ece", "cifar10", ALL),
    ("Segunda similitud por pares sobre la vecindad", "ham_zscore_local_k5", "dot_local_k5", "test_acc", "cifar10", None),
    ("Similitud aditiva por pares sobre la vecindad", "additive_free_local_k5_b128", "dot_local_k5_b128", "test_acc",
     "plantvillage", None),
    ("Tríada, a igual capacidad", "ham_concat_local_k5_b128", "dot_wide_local_k5_b128", "test_acc", "plantvillage", None),
    ("Rama \\emph{pooling} sin vecindad, a igual capacidad", "ham_concat_b128", "dot_wide_b128", "test_acc",
     "plantvillage", None),
]


def paired_metric(a: pt.Variant, b: pt.Variant, metric: str) -> dict:
    """Como ``pt.paired``, para la exactitud o una métrica de ``analysis.json`` (ECE: mejor si baja)."""
    if metric == "test_acc":
        return pt.paired(a, b)
    seeds = sorted(set(a.seeds) & set(b.seeds))
    d = np.array([a.analysis(s)["classification"][metric] - b.analysis(s)["classification"][metric] for s in seeds])
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return {"n": len(d), "delta": d.mean(), "lo": d.mean() - half, "hi": d.mean() + half, "wins": int((d < 0).sum()),
            "p": stats.ttest_1samp(d, 0).pvalue}


def tab_evidencia(out):
    rows = []
    for what, a, b, metric, dataset, seeds, *flags in EVIDENCIA:
        try:
            r = paired_metric(pt.Variant(a, dataset, seeds=seeds), pt.Variant(b, dataset, seeds=seeds), metric)
        except SystemExit:  # sin runs todavía
            continue
        rows.append({"what": what, "a": a, "b": b, "metric": metric, "dataset": dataset,
                     "primary": "primario" in flags} | r)
    pt.with_holm([r for r in rows if not r["primary"]])
    body, last = [], None
    for r in rows:
        if last and r["dataset"] != last:
            body.append("mid")
        last = r["dataset"]
        better = r["delta"] < 0 if r["metric"] == "ece" else r["delta"] > 0
        p_dec = r["p"] if r["primary"] else r["p_holm"]
        level = ("confirmado" if p_dec < 0.05 else "indicio" if r["p"] < 0.05
                 else "no distinguible") if better else "no distinguible" if r["p"] >= 0.05 else "en contra"
        body.append([r["what"], f"{pt.tt(r['a'].removesuffix('_b128'))} $-$ {pt.tt(r['b'].removesuffix('_b128'))}",
                     "ECE" if r["metric"] == "ece" else "Exactitud", pt.ci(r), pt.wins(r), pt.pfmt(r["p"]),
                     "---" if r["primary"] else pt.pfmt(r["p_holm"]), level])
    pt.table(out / "evidencia.tex",
             "Qué aporta cada componente de la tríada \\texttt{ham\\_concat\\_local\\_k5}. Diferencias pareadas por "
             "semilla en CIFAR-10 (arriba) y PlantVillage (abajo, lote 128, semillas 0--4); en ECE (\\%), "
             "$\\Delta>0$ cuenta las semillas en que baja. $n$: semillas pareadas. Holm sobre las filas de la tabla "
             "salvo el contraste primario (la tríada contra su control de capacidad en exactitud), que se declaró "
             "como prueba única antes de correr las semillas 16--47; la réplica usa sólo esas semillas. "
             "Indicio: $p<0.05$ sin corregir.",
             "tab:evidencia", ">{\\raggedright\\arraybackslash}p{4.4cm}lccccc>{\\raggedright\\arraybackslash}p{1.9cm}",
             ["Componente", "Contraste", "Métrica", "$\\Delta$ [IC 95~\\%]", "$\\Delta>0$ / $n$", "$p$", "$p$ Holm",
              "Evidencia"], body, size=r"\footnotesize")


# --------------------------------------------------------------------------- fase18 (revisión)

def variant(name, dataset="cifar10", seeds=None):
    """``pt.Variant`` o ``None`` si todavía no hay runs."""
    try:
        return pt.Variant(name, dataset, seeds=seeds)
    except SystemExit:
        return None


def paired_cells(a, b):
    """Celdas ``Δ [IC]`` y ``Δ>0/n`` de ``a − b`` pareadas por semilla, o guiones."""
    if a is None or b is None or len(set(a.seeds) & set(b.seeds)) < 2:
        return ["---", "---"]
    r = pt.paired(a, b)
    return [pt.ci(r), pt.wins(r)]


def acc_cell(v):
    return "---" if v is None else pt.pm(v.accs())


def min_head_distance(v, branch="dot", layer=0):
    """Distancia media de atención (px) de la cabeza más local en ``layer``, media sobre semillas."""
    vals = []
    for s in v.seeds:
        heads = v.analysis(s)["attention"]["branches"].get(branch, {}).get("distance_px_heads")
        if heads is None:
            return None
        vals.append(min(heads[layer]))
    return float(np.mean(vals))


ARMS = [("Estándar", ""), ("CutMix", "_cutmix"), ("Aumento fuerte", "_fuerte")]


def tab_aumento(out):
    """¿CutMix o un aumento fuerte reducen la ganancia del término local? (semillas 0--4)."""
    body = []
    for label, suf in ARMS:
        v = {n: variant(n + suf, seeds=range(5)) for n in pt_clave()}
        if v["dot"] is None:
            continue
        dist = [min_head_distance(v[n]) if v[n] is not None else None for n in ("dot", "dot_local_k5")]
        body.append([label, acc_cell(v["dot"]), acc_cell(v["dot_local_k5"]), *paired_cells(v["dot_local_k5"], v["dot_wide"]),
                     acc_cell(v["ham_concat_local_k5"]), *paired_cells(v["ham_concat_local_k5"], v["dot_wide_local_k5"]),
                     " / ".join("---" if d is None else f"{d:.1f}" for d in dist)])
    if not body:
        return
    pt.table(out / "aumento.tex",
             "Efecto del protocolo de aumento en CIFAR-10 (semillas 0--4, 400 épocas). CutMix: $\\alpha=1$ con "
             "probabilidad 0.6, como en \\cite{santiago2026inducing}. Aumento fuerte: RandAugment, CutMix o Mixup en "
             "cada lote y suavizado de etiquetas 0.1. Vecindad: \\texttt{dot\\_local\\_k5} $-$ \\texttt{dot\\_wide}; "
             "tríada: \\texttt{ham\\_concat\\_local\\_k5} $-$ \\texttt{dot\\_wide\\_local\\_k5}. Distancia: distancia "
             "media de atención (px) de la cabeza más local de la capa 1; con atención uniforme sería de 16.5~px.",
             "tab:aumento", "lcccccccc",
             ["Protocolo", "\\texttt{dot}", "\\texttt{dot\\_local\\_k5}", "$\\Delta$ vecindad [IC 95~\\%]", "$\\Delta>0$",
              "Tríada", "$\\Delta$ tríada [IC 95~\\%]", "$\\Delta>0$", "Distancia (\\texttt{dot} / local)"], body)


def pt_clave():
    return ["dot", "dot_wide", "dot_local_k5", "dot_wide_local_k5", "ham_concat_local_k5"]


BASELINES = [  # (variante, referencia)
    ("localvit", "LocalViT \\cite{li2021localvit}"),
    ("spt_lsa", "SPT + LSA \\cite{lee2021vision}"),
    ("convit", "ConViT \\cite{dascoli2021convit}"),
    ("conv_stem", "\\emph{Stem} conv. \\cite{xiao2021early}"),
    ("resnet", "ResNet \\cite{he2016deep}"),
]


def tab_baselines(out):
    """Baselines publicados con el mismo presupuesto de parámetros y protocolo (semillas 0--4)."""
    ref = variant("dot_local_k5", seeds=range(5))
    dot = variant("dot", seeds=range(5))
    ours = [("dot", "ViT base"), ("dot_wide", "ViT base, MLP 217"), ("dot_local_k5", "Este trabajo"),
            ("ham_concat_local_k5", "Este trabajo (tríada)")]
    rows = []
    for name, cite in ours + BASELINES:
        v = variant(name, seeds=range(5))
        if v is None:
            continue
        r = {"name": name, "cite": cite, "v": v}
        if name != "dot_local_k5":
            r |= pt.paired(v, ref)
        rows.append(r)
    if not any(r["name"] in dict(BASELINES) for r in rows):
        return
    pt.with_holm([r for r in rows if "p" in r])
    body = []
    for r in rows:
        v = r["v"]
        if r["name"] == "localvit":
            body.append("mid")
        cmp = [pt.ci(r), pt.wins(r), pt.pfmt(r["p_holm"])] if "p" in r else ["---"] * 3
        ece = np.mean([v.analysis(s)["classification"]["ece"] for s in v.seeds]) if all(
            (v.dirs[s] / "analysis.json").exists() for s in v.seeds) else None
        body.append([pt.tt(r["name"]), r["cite"], pt.thousands(v.params), pt.pm(v.accs()), *cmp,
                     f"{v.s_epoch() / dot.s_epoch():.2f}$\\times$", "---" if ece is None else f"{ece:.2f}"])
    pt.table(out / "baselines.tex",
             "Comparación con mecanismos de localidad publicados, con el mismo presupuesto de parámetros (119k--134k) y "
             "el mismo protocolo de entrenamiento (CIFAR-10, 400 épocas, semillas 0--4). $\\Delta$ pareada contra "
             "\\texttt{dot\\_local\\_k5}; Holm sobre las comparaciones de la tabla. Costo: tiempo por época relativo a "
             "\\texttt{dot}. ECE en \\%.",
             "tab:baselines", "llrcccccc",
             ["Variante", "Mecanismo", "Parám.", "Exactitud (\\%)", "$\\Delta$ [IC 95~\\%]", "$\\Delta>0$", "$p$ Holm",
              "Costo", "ECE"], body)


DATASETS = [("CIFAR-10", "cifar10", ""), ("CIFAR-100", "cifar100", ""), ("SVHN", "svhn", ""),
            ("PlantVillage", "plantvillage", "_b128")]


def tab_datos(out):
    """La vecindad y la tríada contra sus controles de capacidad en cuatro conjuntos (semillas 0--4)."""
    body = []
    for label, ds, suf in DATASETS:
        v = {n: variant(n + suf, ds, seeds=range(5)) for n in pt_clave()}
        if v["dot"] is None or v["dot_local_k5"] is None:
            continue
        body.append([label] + [acc_cell(v[n]) for n in pt_clave()]
                    + paired_cells(v["dot_local_k5"], v["dot_wide"])
                    + paired_cells(v["ham_concat_local_k5"], v["dot_wide_local_k5"]))
    if len(body) < 2:
        return
    pt.table(out / "datos.tex",
             "Exactitud de test (\\%) en cuatro conjuntos de datos, semillas 0--4. CIFAR-100: 400 épocas; SVHN: 100 "
             "épocas, sin volteo; PlantVillage: 50 épocas, lote 128. Vecindad: \\texttt{dot\\_local\\_k5} $-$ "
             "\\texttt{dot\\_wide}; tríada: \\texttt{ham\\_concat\\_local\\_k5} $-$ \\texttt{dot\\_wide\\_local\\_k5}.",
             "tab:datos", "lccccccccc",
             ["Conjunto", "\\texttt{dot}", "\\texttt{dot\\_wide}", "\\texttt{dot\\_local\\_k5}",
              "\\texttt{dot\\_wide\\_local\\_k5}", "Tríada", "$\\Delta$ vecindad", "$\\Delta>0$", "$\\Delta$ tríada",
              "$\\Delta>0$"], body)


def tab_escala(out):
    """Ganancia del término local según la fracción de datos de entrenamiento y el tamaño del modelo."""
    body = []
    for f in (10, 25, 50, 100):
        suf = "" if f == 100 else f"_frac{f}"
        dot, loc = variant("dot" + suf, seeds=range(5)), variant("dot_local_k5" + suf, seeds=range(5))
        tri, ctrl = variant("ham_concat_local_k5" + suf, seeds=range(5)), variant("dot_wide_local_k5" + suf, seeds=range(5))
        if dot is None or loc is None:
            continue
        body.append([f"{f}~\\% de los datos", pt.thousands(dot.params), acc_cell(dot), acc_cell(loc),
                     *paired_cells(loc, dot), *paired_cells(tri, ctrl)])
    body.append("mid")
    for d, l in ((48, 4), (96, 4), (192, 4), (48, 8), (96, 8), (192, 8)):
        suf = "" if (d, l) == (48, 4) else f"_d{d}_l{l}"
        dot, loc = variant("dot" + suf, seeds=range(3)), variant("dot_local_k5" + suf, seeds=range(3))
        if dot is None or loc is None:
            continue
        body.append([f"$D={d}$, $L={l}$", pt.thousands(dot.params), acc_cell(dot), acc_cell(loc),
                     *paired_cells(loc, dot), "---", "---"])
    if sum(r != "mid" for r in body) < 3:
        return
    pt.table(out / "escala.tex",
             "Ganancia del término local $5\\times5$ según la fracción del conjunto de entrenamiento de CIFAR-10 "
             "(arriba, semillas 0--4, 400 épocas) y el tamaño del modelo (abajo, 4 cabezas y MLP de $4D$, semillas "
             "0--2). Parámetros de \\texttt{dot}. Tríada: \\texttt{ham\\_concat\\_local\\_k5} $-$ "
             "\\texttt{dot\\_wide\\_local\\_k5}.",
             "tab:escala", "lrcccccc",
             ["Configuración", "Parám.", "\\texttt{dot}", "\\texttt{dot\\_local\\_k5}", "$\\Delta$ local [IC 95~\\%]",
              "$\\Delta>0$", "$\\Delta$ tríada", "$\\Delta>0$"], body)


COSTOS = Path(__file__).resolve().parents[1] / "results" / "costos.json"  # scripts/costos.py


def tab_costos(out):
    """Costo independiente del hardware (MACs, FLOPs, memoria) y de inferencia en CPU, de ``results/costos.json``."""
    if not COSTOS.exists():
        return
    import json
    data = json.loads(COSTOS.read_text())
    c = data["variants"]
    if "dot" not in c:
        return
    dot = variant("dot", seeds=range(5))
    names = ["dot", "dot_wide", "ham_concat", "ham_sum", "dot_local", "dot_local_k5", "dot_wide_local_k5",
             "ham_concat_local_k5", "mid"] + [b for b, _ in BASELINES]
    body = []
    for name in names:
        if name == "mid":
            body.append("mid")
            continue
        if name not in c:
            continue
        r, v = c[name], variant(name, seeds=range(5))
        rel = "---" if v is None or dot is None else f"{v.s_epoch() / dot.s_epoch():.2f}$\\times$"
        dmacs = "---" if name == "dot" else f"{100 * (r['macs_img'] / c['dot']['macs_img'] - 1):+.1f}~\\%"
        body.append([pt.tt(name), pt.thousands(r["params"]), f"{r['macs_img'] / 1e6:.2f}", dmacs,
                     f"{r['train_flops_epoch'] / 1e12:.2f}", f"{r['train_mem_mb']:.0f}", rel,
                     f"{r['cpu_ms_1thread']['median']:.2f}", f"{r['act_peak_kb']:.0f}", f"{r['disk_kb']:.0f}"])
    m = data["machine"]
    pt.table(out / "costos.tex",
             "Costo de cada variante en CIFAR-10. Independiente del hardware: MACs por imagen en inferencia y FLOPs "
             "de entrenamiento por época (\\emph{forward} y \\emph{backward} de 45\\,000 imágenes), contando productos "
             "de matrices y convoluciones, no operaciones elemento a elemento. Memoria pico de un paso de "
             "entrenamiento (lote 256, fp32, AdamW) y tiempo por época relativo a \\texttt{dot} (semillas 0--4), "
             f"en una {m['gpu'].replace('NVIDIA GeForce ', '')}. Inferencia con lote 1: latencia mediana en CPU con un "
             f"hilo ({m['cpu'].replace(' 8-Core Processor', '')}, {data['reps']} repeticiones), pico de activaciones "
             "y tamaño del modelo en disco (fp32).",
             "tab:costos", "lrrrrrrrrr",
             ["Variante", "Parám.", "MMACs", "$\\Delta$ MACs", "TFLOPs/época", "Mem. (MB)", "s/época",
              "CPU (ms)", "Activ. (KB)", "Disco (KB)"], body)


UMBRALES = [85.0, 85.25, 85.5, 85.75, 86.0]


def tab_umbral(out):
    """Sensibilidad de la frecuencia del modo alto al umbral (fijado en 85.5 después de ver la distribución)."""
    s16 = {n: variant(n) for n in ["dot_local", "dot_local_k5", "dot_wide_local_k5", "ham_concat_local_k5"]}
    s48 = {n: variant(n, seeds=ALL) for n in ["dot_wide_local_k5", "ham_concat_local_k5"]}

    def high(v, thr):
        return int((v.accs() >= thr).sum()), len(v.seeds)

    def fisher(a, b, thr):
        (ka, na), (kb, nb) = high(a, thr), high(b, thr)
        return stats.fisher_exact([[ka, na - ka], [kb, nb - kb]])[1]

    body = []
    for thr in UMBRALES:
        cells = [f"{thr:.2f}" + (" (usado)" if thr == pt.HIGH_MODE else "")]
        cells += ["{}/{}".format(*high(s16[n], thr)) for n in s16]
        cells.append(pt.pfmt(fisher(s16["dot_local_k5"], s16["dot_local"], thr)))
        cells += ["{}/{}".format(*high(s48[n], thr)) for n in ("ham_concat_local_k5", "dot_wide_local_k5")]
        cells.append(pt.pfmt(fisher(s48["ham_concat_local_k5"], s48["dot_wide_local_k5"], thr)))
        body.append(cells)
    pt.table(out / "umbral.tex",
             "Sensibilidad de la frecuencia del modo alto al umbral de exactitud de test en CIFAR-10. Semillas en el "
             "modo alto / semillas; $p$: prueba exacta de Fisher bilateral. Izquierda: 16 semillas (0--15); "
             "derecha: la tríada y su control de igual capacidad con 48 semillas (0--47).",
             "tab:umbral", "lcccccccc",
             ["Umbral (\\%)", "\\texttt{dot\\_local}", "\\texttt{dot\\_local\\_k5}", "\\texttt{dot\\_wide\\_local\\_k5}",
              "Tríada", "$p$ ($5\\times5$ vs.\\ $3\\times3$)", "Tríada", "Control", "$p$"], body, size=r"\footnotesize")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("paper/ieee/tablas"))
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    pt.HEADER = "% Generado por scripts/ieee_tables.py a partir de los runs; no editar a mano.\n"
    pt.ENV = "table*"
    pt.ENV_BY_LABEL = {label: "table" for label in NARROW}
    pt.MAIN = pt.MAIN[:-1] + ["dot_wide_local_k5", pt.MAIN[-1]]
    pt.CONTRASTS = pt.CONTRASTS + [("ham_concat", "ham_dotdot", "rama global sin localidad, contra la proyección de la fusión"),
                                   ("ham_concat_local_k5", "dot_wide_local_k5", "rama global con localidad $5\\times5$, a igual capacidad")]
    pt.PV_NAMES = (["dot_wide_b128"] + pt.PV_NAMES + ["dot_wide_local_k5_b128", "ham_concat_local_k5_b128",
                                                      "additive_free_local_k5_b128", "ham_sum_local_k5_b128"])

    for build in (pt.tab_variantes, pt.tab_fusiones, pt.tab_colapso, pt.tab_principal, pt.tab_contrastes, pt.tab_ablacion,
                  pt.tab_eficiencia, pt.tab_plantvillage, pt.tab_calibracion, pt.tab_otros,
                  propuesta_tables.t_parametros, lambda out: propuesta_tables.t_criterios(out, borrador=True), tab_evidencia,
                  tab_aumento, tab_baselines, tab_datos, tab_escala, tab_costos, tab_umbral):
        build(args.out)
    pt.cifras()


if __name__ == "__main__":
    main()
