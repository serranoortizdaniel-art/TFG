"""Tablas resumen en LaTeX y Markdown para la memoria.

A partir de final_summary.csv (generado por aggregate_results) produce:
  - tabla principal por ataque: accuracy media +- std de cada agregador
    para cada fraccion de maliciosos y particion
  - tabla de coste de sobreproteccion: rendimiento de cada agregador sin
    ataque frente a FedAvg (escenario E1b)

Uso:  python -m src.analysis.generate_tables
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.common import AGGREGATOR_LABELS, ATTACK_LABELS

AGG_ORDER = ["fedavg", "median", "trimmed_mean", "krum"]


def fmt_cell(mean: float, std: float) -> str:
    """Formato compacto de accuracy porcentual para Markdown y LaTeX."""
    if pd.isna(mean):
        return "--"
    if pd.isna(std):
        return f"{100 * mean:.2f}"
    return f"{100 * mean:.2f} $\\pm$ {100 * std:.2f}"


def attack_table(summary: pd.DataFrame, dataset: str, attack: str) -> pd.DataFrame:
    """Tabla pivote: filas = (particion, % maliciosos), columnas = agregador."""
    subset = summary[(summary["dataset"] == dataset)
                     & (summary["attack"] == attack)]
    rows = []
    for (partition, pct), group in subset.groupby(["partition", "malicious_pct"]):
        row = {"Particion": partition, "% maliciosos": pct}
        for agg in AGG_ORDER:
            match = group[group["aggregator"] == agg]
            if match.empty:
                row[AGGREGATOR_LABELS[agg]] = "--"
            else:
                row[AGGREGATOR_LABELS[agg]] = fmt_cell(
                    match["accuracy_mean"].iloc[0],
                    match["accuracy_std"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="results/processed/final_summary.csv")
    parser.add_argument("--output-dir", default="results/processed/tables")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"No existe {summary_path}; ejecuta antes "
              "python -m src.analysis.aggregate_results")
        return
    summary = pd.read_csv(summary_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for dataset in summary["dataset"].unique():
        for attack in summary["attack"].unique():
            table = attack_table(summary, dataset, attack)
            if table.empty:
                continue
            attack_label = ATTACK_LABELS.get(attack, attack)
            dataset_tex = dataset.replace("_", "\\_")
            caption = (f"Accuracy final (\\%, media $\\pm$ desviacion tipica "
                       f"sobre semillas) con {attack_label.lower()} "
                       f"en {dataset_tex}.")
            stem = output_dir / f"tabla_{dataset}_{attack}"
            # El % de la cabecera debe escaparse porque to_latex usa escape=False
            table_tex = table.rename(
                columns={"% maliciosos": "\\% maliciosos"})
            latex = table_tex.to_latex(index=False, escape=False,
                                       caption=caption,
                                       label=f"tab:{dataset}_{attack}",
                                       column_format="ll" + "r" * len(AGG_ORDER))
            Path(f"{stem}.tex").write_text(latex, encoding="utf-8")
            Path(f"{stem}.md").write_text(
                table.to_markdown(index=False), encoding="utf-8")
            count += 1
    print(f"{count} tablas en {output_dir} (.tex y .md)")


if __name__ == "__main__":
    main()
