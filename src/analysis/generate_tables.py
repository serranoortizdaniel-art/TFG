"""Tablas resumen en LaTeX y Markdown para la memoria.

A partir de final_summary.csv (generado por aggregate_results) produce, para
cada dataset y ataque:
  - tabla con la politica fija: accuracy media +- std de los cuatro agregadores
    para cada fraccion de maliciosos y particion (la tabla del ataque sin
    adversarios recoge el coste de sobreproteccion, escenario E1b)
  - tabla con la politica oracle (sufijo _oracle): misma estructura restringida
    a Krum y Trimmed Mean, las unicas defensas que la politica afecta, para
    cuantificar cuanto rinde conocer la fraccion real de maliciosos

Uso:  python -m src.analysis.generate_tables
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.common import AGGREGATOR_LABELS, ATTACK_LABELS

AGG_ORDER = ["fedavg", "median", "trimmed_mean", "krum"]
# Defensas parametrizadas cuyo resultado depende de la politica (fija u oracle);
# la mediana coordinada no tiene parametros, por lo que no figura en la oracle.
ORACLE_AGG_ORDER = ["trimmed_mean", "krum"]


def fmt_cell(mean: float, std: float) -> str:
    """Formato compacto de accuracy porcentual para Markdown y LaTeX."""
    if pd.isna(mean):
        return "--"
    if pd.isna(std):
        return f"{100 * mean:.2f}"
    return f"{100 * mean:.2f} $\\pm$ {100 * std:.2f}"


def attack_table(summary: pd.DataFrame, dataset: str, attack: str,
                 policy: str = "fixed") -> pd.DataFrame:
    """Tabla pivote: filas = (particion, % maliciosos), columnas = agregador.

    Con ``policy="fixed"`` (configuracion principal de la metodologia) la tabla
    incluye los cuatro agregadores; con ``policy="oracle"`` se restringe a las
    defensas parametrizadas (Krum y Trimmed Mean), las unicas que la politica
    afecta.
    """
    aggs = AGG_ORDER if policy == "fixed" else ORACLE_AGG_ORDER
    subset = summary[(summary["dataset"] == dataset)
                     & (summary["attack"] == attack)
                     & (summary["policy"] == policy)]
    rows = []
    for (partition, pct), group in subset.groupby(["partition", "malicious_pct"]):
        row = {"Particion": partition, "% maliciosos": pct}
        for agg in aggs:
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
            # Tabla con la politica fija (principal) y, cuando existan datos, su
            # equivalente con la politica oracle para las defensas afectadas.
            for policy in ("fixed", "oracle"):
                table = attack_table(summary, dataset, attack, policy)
                if table.empty:
                    continue
                aggs = AGG_ORDER if policy == "fixed" else ORACLE_AGG_ORDER
                suffix = "" if policy == "fixed" else "_oracle"
                policy_txt = "" if policy == "fixed" else " (politica oracle)"
                attack_label = ATTACK_LABELS.get(attack, attack)
                dataset_tex = dataset.replace("_", "\\_")
                caption = (f"Accuracy final (\\%, media $\\pm$ desviacion tipica "
                           f"sobre semillas) con {attack_label.lower()} "
                           f"en {dataset_tex}{policy_txt}.")
                stem = output_dir / f"tabla_{dataset}_{attack}{suffix}"
                # El % de la cabecera debe escaparse porque to_latex usa escape=False
                table_tex = table.rename(
                    columns={"% maliciosos": "\\% maliciosos"})
                latex = table_tex.to_latex(index=False, escape=False,
                                           caption=caption,
                                           label=f"tab:{dataset}_{attack}{suffix}",
                                           column_format="ll" + "r" * len(aggs))
                Path(f"{stem}.tex").write_text(latex, encoding="utf-8")
                Path(f"{stem}.md").write_text(
                    table.to_markdown(index=False), encoding="utf-8")
                count += 1
    print(f"{count} tablas en {output_dir} (.tex y .md)")


if __name__ == "__main__":
    main()
