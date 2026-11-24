"""Tablas de la versión corregida de la primera propuesta (``paper/propuesta1/tablas/``) a partir de los runs.

    uv run python scripts/propuesta_tables.py [--out paper/propuesta1/tablas]

``paper/propuesta1/corregido.tex`` conserva la estructura de la propuesta exportada de Google Docs,
pero toma todas sus cifras de los runs. Mismas convenciones que ``paper_tables.py``: comparaciones
pareadas por semilla, IC 95 % de la t, Wilcoxon con n >= 6 y Holm dentro de cada tabla. Además
imprime las cifras del texto que no están en las tablas.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import paper_tables  # noqa: E402
from paired import epoch_reaching  # noqa: E402
from paper_tables import (FIRST5, HIGH_MODE, THRESHOLD, Variant, ci, epochs_cell, layer_range, paired, pfmt, pm,  # noqa: E402
                          signed, table, thousands, tt, with_holm)

from ham.config import load_config  # noqa: E402
from ham.models import build_model  # noqa: E402

paper_tables.HEADER = "% Generado por scripts/propuesta_tables.py a partir de los runs; no editar a mano.\n"

# (variante, semillas que tiene): las fusiones de scores sólo tienen 0-4
MAIN = ["dot", "ham_weighted", "ham_sum", "ham_concat", "dot_wide", "dot_local", "dot_local_k5",
        "dot_wide_local_k5", "ham_concat_local_k5"]
LABEL = {"ham_weighted": "ham_weighted (scores)", "ham_sum": "ham_sum (scores)", "ham_concat": "ham_concat (concat)",
         "dot_local": "dot_local (3×3)"}
CAPACITY = {"ham_concat": "dot_wide", "dot_local_k5": "dot_wide", "ham_concat_local_k5": "dot_wide_local_k5"}


def name_cell(n):
    label = LABEL.get(n, n)
    base, _, extra = label.partition(" ")
    return tt(base) + (f" {extra}".replace("×", r"$\times$") if extra else "")


def t_principal(out):
    dot = Variant("dot")
    rows = with_holm([{"v": Variant(n)} | paired(Variant(n), dot) for n in MAIN[1:]])
    t_dot = dot.s_epoch()
    body = [[name_cell("dot"), "16", thousands(dot.params), pm(dot.accs()), "---", "", "", epochs_cell(dot),
             f"{t_dot:.2f}", "1.00"]]
    for r in rows:
        v = r["v"]
        body.append([name_cell(v.name), str(r["n"]), thousands(v.params), pm(v.accs()), ci(r), f"{r['wins']}/{r['n']}",
                     pfmt(r["p_holm"]), epochs_cell(v), f"{v.s_epoch():.2f}", f"{v.s_epoch() / t_dot:.2f}"])
    table(out / "principal.tex",
          r"Evaluación comparativa en CIFAR-10 (400 épocas). $\Delta$: diferencia pareada por semilla contra "
          rf"\texttt{{dot}}, con IC 95\,\% de la t de Student; Holm sobre las {len(rows)} comparaciones. Las fusiones de puntajes tienen 5 semillas (0--4) y las demás 16 (0--15). Época $\geq79$: primera "
          r"época con exactitud de validación $\geq79$\,\% (media de las semillas que la alcanzan). Tiempo por época en "
          r"una RTX 5080, una corrida a la vez.",
          "tab:principal", "lcrccccccc",
          ["Variante", "$n$", "Parám.", "Exactitud (\\%)", r"$\Delta$ [IC 95\,\%]", r"$\Delta>0$", "Holm",
           r"Época $\geq79$", "s/época", r"$\times$\,\texttt{dot}"], body, size=r"\footnotesize")


CATEGORIES = [  # (categoría, función sobre el nombre del parámetro)
    ("Embeddings (parches, posición, CLS)", lambda n: n.startswith("patch_embed")),
    (r"Proyección $\mathbf{Q},\mathbf{K},\mathbf{V}$", lambda n: ".qkv." in n),
    (r"Rama aditiva \emph{pooling}", lambda n: ".attn2." in n),
    (r"Fusión $\mathbf{W}_f$ (\texttt{concat\_linear})", lambda n: ".fusion." in n),
    (r"Proyección de salida $\mathbf{W}_o$", lambda n: ".out_proj." in n),
    (r"Término local ($\dwconv_{5\times5}$)", lambda n: ".enrich." in n),
    ("MLP", lambda n: ".mlp." in n),
    ("Normalizaciones de capa", lambda n: ".norm" in n or "final_norm" in n or n.startswith("mlp_head.0")),
    ("Clasificador", lambda n: n.startswith("mlp_head.1")),
]


def t_parametros(out):
    names = ["dot", "ham_concat", "dot_local_k5", "ham_concat_local_k5"]
    counts, dead = {}, {}
    for n in names:
        cfg = load_config(f"configs/cifar10_{n}.yaml")
        cfg.model.num_classes = 10
        c = Counter()
        last = f"encoder.layers.{cfg.model.num_layers - 1}."
        dead[n] = 0
        for pname, p in build_model(cfg.model).named_parameters():
            cat = [label for label, match in CATEGORIES if match(pname)]
            if len(cat) != 1:
                raise ValueError(f"{pname}: {cat}")
            c[cat[0]] += p.numel()
            # En la última capa sólo el CLS llega a la pérdida y el término local no actúa sobre el CLS:
            # esos parámetros no reciben gradiente (se quedan en cero).
            if pname.startswith(last) and ".enrich." in pname:
                dead[n] += p.numel()
        total = sum(c.values())
        if total != Variant(n).params:
            raise ValueError(f"{n}: {total} contra {Variant(n).params} de test.json")
        counts[n] = c
    body = [[label, *(thousands(counts[n][label]) if counts[n][label] else "---" for n in names)] for label, _ in CATEGORIES]
    body += ["mid", ["Total", *(thousands(sum(counts[n].values())) for n in names)],
             ["Total efectivo", *(thousands(sum(counts[n].values()) - dead[n]) for n in names)]]
    table(out / "parametros.tex",
          r"Presupuesto de parámetros por componente, contado sobre el modelo construido (4 capas, $D=48$, $H=4$, "
          r"MLP 192, CIFAR-10). La rama \emph{pooling} tiene $\mathbf{W}_q,\mathbf{W}_k\in\mathbb{R}^{d\times d}$ y "
          r"$\mathbf{w},b$ por cabeza (301 por cabeza); $\mathbf{W}_f\in\mathbb{R}^{2d\times d}$ con sesgo (300 por "
          r"cabeza); el término local, $Hd(k^2+1)=1\,248$ por capa. Total efectivo: sin el término local de la "
          r"capa 4, que no recibe gradiente porque sólo el CLS llega a la pérdida (sección~\ref{sec:terminos}).",
          "tab:parametros", "lrrrr", ["Componente", *(tt(n) for n in names)], body, size=r"\footnotesize")


def rob(v, k):
    return v.mean_analysis("robustness", k)


def t_calibracion(out):
    body = []
    for n in ["dot", "ham_concat", "dot_local_k5", "ham_concat_local_k5"]:
        v = Variant(n)
        body.append([tt(n), str(len(v.seeds)), f"{v.mean_analysis('classification', 'ece'):.2f}",
                     f"{v.mean_analysis('classification', 'nll'):.3f}", f"{rob(v, 'noise_0.1'):.1f}",
                     f"{rob(v, 'blur_1.0'):.1f}", f"{rob(v, 'occlusion_0.25'):.1f}"])
    table(out / "calibracion.tex",
          r"Calibración y exactitud bajo perturbaciones en CIFAR-10 (media de 16 semillas). ECE en \% (15 intervalos). "
          r"Ruido gaussiano con $\sigma=0.1$ sobre la imagen en $[0,1]$; desenfoque gaussiano con $\sigma=1$ píxel; "
          r"oclusión del 25\,\% de los parches, elegidos al azar.",
          "tab:calibracion", "lcccccc",
          ["Modelo", "Semillas", r"ECE (\%) $\downarrow$", r"NLL $\downarrow$", "Ruido", "Desenfoque", "Oclusión"], body)


def t_criterios(out, borrador=False):
    """Cumplimiento de los cuatro criterios, variante por variante.

    Con ``borrador=False``, los de la propuesta: C1 con Holm < 0.01 contra ``dot`` y contra el control de
    capacidad, C3 significativo en PlantVillage y en Food-101. Con ``borrador=True``, los fijados de antemano
    en el borrador (``paper/secciones/metodologia.tex``): C1 con Holm < 0.05 contra el control de igual
    capacidad y al menos 10 semillas, C3 el mismo signo en otro conjunto de datos.
    """
    dot, t_dot = Variant("dot"), Variant("dot").s_epoch()
    names = ["ham_weighted", "ham_sum", "ham_concat", "dot_local_k5", "ham_concat_local_k5"]
    c1 = with_holm([{"name": n, "kind": "dot"} | paired(Variant(n), dot) for n in names]
                   + [{"name": n, "kind": "cap"} | paired(Variant(n, seeds=range(1000)), Variant(CAPACITY[n], seeds=range(1000)))
                      for n in names if n in CAPACITY])  # contra la capacidad, todas las semillas (fase17: 48 en la tríada)
    by = {(r["name"], r["kind"]): r for r in c1}
    body = []
    for n in names:
        v = Variant(n)
        r = by[(n, "dot")]
        cell1 = f"{signed(r['delta'])}, Holm {pfmt(r['p_holm'])}"
        if (n, "cap") in by:
            rc = by[(n, "cap")]
            cell1 += rf"\newline vs.\ {tt(CAPACITY[n])}: {signed(rc['delta'])}, Holm {pfmt(rc['p_holm'])} ($n={rc['n']}$)"
        if borrador:  # sin control propio (mismos parámetros que dot), el control es dot
            control = by.get((n, "cap"), r)
            ok1 = control["p_holm"] < 0.05 and control["n"] >= 10
        else:
            ok1 = r["p_holm"] < 0.01 and ((n, "cap") not in by or by[(n, "cap")]["p_holm"] < 0.01)
        a = v.analysis(v.seeds[0])["attention"]
        if a.get("branch_contribution"):
            other = next(b for b in a["branch_contribution"] if b != "dot")
            lo = min(v.mean_analysis("attention", "branch_contribution", b).min() for b in ("dot", other))
            cell2, ok2 = f"rama menor: {lo:.2f}", 0.10 <= lo
        elif "dot|fused" in a["js_between"]:
            other = next(b for b in a["branches"] if b not in ("dot", "fused"))
            js = {b: v.mean_analysis("attention", "js_between", f"{b}|fused").max() for b in ("dot", other)}
            near = min(js, key=js.get)
            cell2, ok2 = f"fusión = {tt(near)} (JS $\\leq${js[near]:.3f})", False
        else:
            w = v.mean_analysis("attention", "enrich_contribution", "terms", "local")
            cell2, ok2 = f"una sola rama de atención; término local {w[:3].min():.2f}--{w[:3].max():.2f} (capas 1--3)", None
        cell3, ok3 = cross_dataset(n, borrador)
        ratio = v.s_epoch() / t_dot
        body.append([tt(n), cell1, yes(ok1), cell2, yes(ok2), cell3, yes(ok3), f"{ratio:.2f}$\\times$", yes(ratio < 1.5)])
    caption = CRITERIOS_BORRADOR if borrador else CRITERIOS_PROPUESTA
    table(out / "criterios.tex", caption,
          "tab:criterios", r">{\raggedright\arraybackslash}p{3.25cm}>{\raggedright\arraybackslash}p{3.5cm}c"
          r">{\raggedright\arraybackslash}p{2.85cm}c>{\raggedright\arraybackslash}p{3.0cm}ccc",
          ["Variante", "C1: ganancia", "", "C2: aporte por rama", "", "C3: otros conjuntos", "", "C4: costo", ""],
          body, size=r"\footnotesize")


CRITERIOS_PROPUESTA = (
          r"Cumplimiento de los cuatro criterios de validación de la sección~2. C1: $\Delta$ pareada en CIFAR-10 con Holm "
          r"$<0.01$ contra \texttt{dot} y contra el control de capacidad con el mismo número de parámetros (Holm sobre "
          r"las 8 comparaciones). C2: ninguna rama aporta menos del 10\,\% de la norma de la salida; en las fusiones "
          r"de puntajes no hay salidas separadas y se usa la divergencia JS entre la atención aplicada y cada rama "
          r"(JS $\approx0$: la fusión es esa rama). C3: $\Delta>0$ con $p<0.05$ (t pareada, semillas 0--4) en "
          r"PlantVillage (lote 128) y en Food-101 (parcial: sólo en el conjunto evaluado; n/a: no aplica). C4: tiempo por época $<1.5$ veces el de \texttt{dot}.")
CRITERIOS_BORRADOR = (
    r"Cumplimiento de los cuatro criterios fijados de antemano (sección~\ref{sec:analisis}). C1: ganancia pareada en "
    r"CIFAR-10 sobre el control de igual capacidad (\texttt{dot} para las fusiones de puntajes, que no agregan "
    r"parámetros) con Holm $<0.05$ sobre las 8 comparaciones y al menos 10 semillas. C2: ninguna rama aporta menos "
    r"del 10\,\% de la norma de la salida; en las fusiones de puntajes, que no tienen salidas separadas, se usa la "
    r"divergencia JS entre la atención aplicada y cada rama (JS $\approx0$: la fusión es esa rama); n/a: una sola rama "
    r"de atención. C3: el contraste de C1, contra el mismo control, tiene $\Delta>0$ en otro conjunto de datos (PV: "
    r"PlantVillage, lote 128; Food: Food-101; semillas 0--4, $p$ de la t pareada entre paréntesis). C4: tiempo por "
    r"época $<1.5$ veces el de \texttt{dot}.")


def yes(ok):
    return {True: "sí", False: "no", None: "n/a"}.get(ok, ok)


def cross_dataset(n, borrador=False):
    """Δ en PlantVillage (lote 128) y Food-101, semillas 0-4.

    Con ``borrador=False``, contra ``dot``. Con ``borrador=True``, el mismo contraste de C1: contra el control
    de igual capacidad cuando la variante lo tiene (``dot`` para las fusiones de puntajes).
    """
    cells, oks, signs = [], [], []
    ref_name = CAPACITY.get(n, "dot") if borrador else "dot"
    for ds, suffix, label in [("plantvillage", "_b128", "PV"), ("food101", "", "Food")]:
        try:
            v = Variant(n + suffix, ds)
        except SystemExit:
            cells.append(f"{label}: no evaluado")
            continue
        try:
            ref = Variant(ref_name + suffix, ds)
        except SystemExit:
            cells.append(f"{label}: control no evaluado")
            continue
        r = paired(v, ref)
        cells.append(f"{label}: {signed(r['delta'])} ({pfmt(r['p'])})")
        oks.append(r["delta"] > 0 and r["p"] < 0.05)
        signs.append(r["delta"] > 0)
    if borrador:
        return r"\newline ".join(cells), (any(signs) if signs else None)
    ok = None if not oks else (False if not all(oks) else (True if len(oks) == 2 else "parcial"))
    return r"\newline ".join(cells), ok


def cifras():
    """Cifras del texto que no están en las tablas."""
    for n in ["dot", "dot_local_k5", "ham_concat_local_k5"]:
        v = Variant(n)
        reached, total = v.epochs_to(THRESHOLD)
        # como paper_tables.tab_eficiencia: época alcanzada por el tiempo medio por época (sin la primera)
        secs = [epoch_reaching(v.metrics(s), THRESHOLD) * np.mean([r["time_s"] for r in v.metrics(s)[1:]])
                for s in v.seeds if epoch_reaching(v.metrics(s), THRESHOLD) is not None]
        print(f"{n}: época ≥{THRESHOLD:.0f} = {np.mean(reached):.0f} ({len(reached)}/{total}), {np.mean(secs):.0f} s de GPU")
    for n in ["dot_local", "dot_local_k5", "ham_concat_local_k5"]:
        a = Variant(n).accs()
        lo = a[a < HIGH_MODE]
        print(f"{n}: modo alto {(a >= HIGH_MODE).sum()}/{len(a)}; modo bajo media {lo.mean():.2f} [{lo.min():.2f}, {lo.max():.2f}]")
    v = Variant("ham_concat")
    print("ham_concat (16): contribución dot por capa", np.round(v.mean_analysis("attention", "branch_contribution", "dot"), 3),
          "JS dot|additive por capa", np.round(v.mean_analysis("attention", "js_between", "dot|additive"), 3))
    for a, b in [("ham_concat", "dot_wide"), ("ham_concat_local_k5", "dot_local_k5"), ("ham_concat_local_k5", "dot_wide_local_k5"),
                 ("dot_local_k5", "dot_local"), ("dot_local_k5", "dot_wide")]:
        print(f"{a} - {b}: {ci(paired(Variant(a), Variant(b)))}")
    for n in ["ham_weighted", "ham_sum"]:
        v = Variant(n, seeds=FIRST5)
        print(n, "H_all dot", layer_range(v.mean_analysis("attention", "branches", "dot", "entropy_all")),
              "H_all aditiva", layer_range(v.mean_analysis("attention", "branches", "additive_free", "entropy_all")))
    pv = {n: Variant(n, "plantvillage") for n in ["dot_b128", "dot_local_k5_b128", "additive_free_local_k5_b128", "ham_sum_local_k5_b128"]}
    for n in list(pv)[1:]:
        print(f"PV {n}: {ci(paired(pv[n], pv['dot_b128']))}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("paper/propuesta1/tablas"))
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for f in (t_principal, t_parametros, t_calibracion, t_criterios):
        f(args.out)
    cifras()


if __name__ == "__main__":
    main()
