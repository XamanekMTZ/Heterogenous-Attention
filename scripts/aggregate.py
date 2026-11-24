"""Agrega ``runs/**/test.json`` en una tabla de media ± desviación por experimento.

    uv run python scripts/aggregate.py [--runs runs] [--csv results.csv] [--latex results.tex]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def collect(runs: Path):
    groups = defaultdict(list)
    for test_file in sorted(runs.glob("*/*/seed*/test.json")):
        dataset, name = test_file.parts[-4], test_file.parts[-3]
        res = json.loads(test_file.read_text())
        groups[(dataset, name)].append(res)
    rows = []
    for (dataset, name), results in sorted(groups.items()):
        accs = [r["test_acc"] for r in results]
        rows.append({
            "dataset": dataset,
            "experiment": name,
            "n_seeds": len(accs),
            "test_acc_mean": statistics.mean(accs),
            "test_acc_std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
            "best_val_acc_mean": statistics.mean(r["best_val_acc"] for r in results),
            "num_params": results[0]["num_params"],
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--csv", type=Path)
    p.add_argument("--latex", type=Path)
    args = p.parse_args()

    rows = collect(args.runs)
    if not rows:
        print(f"No hay test.json bajo {args.runs}")
        return
    print(f"{'dataset':<14}{'experiment':<32}{'n':>3}  {'test acc':>16}  {'params':>9}")
    for r in rows:
        print(f"{r['dataset']:<14}{r['experiment']:<32}{r['n_seeds']:>3}  "
              f"{r['test_acc_mean']:>7.2f} ± {r['test_acc_std']:<5.2f}  {r['num_params']:>9,}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if args.latex:
        lines = [r"\begin{tabular}{llrrr}", r"\toprule",
                 r"Dataset & Experimento & Semillas & Test acc (\%) & Parámetros \\", r"\midrule"]
        for r in rows:
            exp = r["experiment"].replace("_", r"\_")
            lines.append(f"{r['dataset']} & {exp} & {r['n_seeds']} & "
                         f"${r['test_acc_mean']:.2f} \\pm {r['test_acc_std']:.2f}$ & {r['num_params']:,} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        args.latex.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
