"""Consolida los resultados brutos en CSV procesados (fase 9).

Genera en results/processed/:
  - all_rounds.csv     una fila por ejecucion y ronda (formato largo)
  - final_summary.csv  media y desviacion tipica sobre semillas de las
                       metricas finales de cada configuracion, junto con la
                       degradacion respecto a las lineas base:
                         * drop_vs_clean_fedavg: respecto a FedAvg sin ataque
                           (misma particion y dataset) -> impacto del ataque
                         * drop_vs_clean_same_agg: respecto al MISMO agregador
                           sin ataque -> separa el coste de sobreproteccion
                           de la defensa del dano causado por el ataque

Uso:  python -m src.analysis.aggregate_results [--results-dir results/raw]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.common import load_rounds, load_runs

GROUP_COLS = ["dataset", "partition", "attack", "malicious_pct", "aggregator"]


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Media y desviacion sobre semillas + degradaciones vs lineas base.

    La salida conserva una fila por configuracion experimental para que tablas
    y texto de resultados lean siempre la misma fuente consolidada.
    """
    summary = (runs
               .groupby(GROUP_COLS)
               .agg(num_seeds=("seed", "nunique"),
                    accuracy_mean=("final_accuracy", "mean"),
                    accuracy_std=("final_accuracy", "std"),
                    loss_mean=("final_loss", "mean"),
                    loss_std=("final_loss", "std"),
                    macro_f1_mean=("final_macro_f1", "mean"),
                    macro_f1_std=("final_macro_f1", "std"),
                    duration_s_mean=("duration_s", "mean"))
               .reset_index())

    # Linea base principal: FedAvg sin ataque, misma particion y dataset
    clean_fedavg = (summary[(summary["attack"] == "none")
                            & (summary["aggregator"] == "fedavg")]
                    .set_index(["dataset", "partition"])["accuracy_mean"])
    # Linea base por agregador: el mismo agregador sin ataque
    clean_same = (summary[summary["attack"] == "none"]
                  .set_index(["dataset", "partition", "aggregator"])
                  ["accuracy_mean"])

    def lookup(index, series):
        """Consulta tolerante: devuelve NaN si falta una linea base."""
        try:
            return float(series.loc[index])
        except KeyError:
            return float("nan")

    summary["drop_vs_clean_fedavg"] = summary.apply(
        lambda r: lookup((r["dataset"], r["partition"]), clean_fedavg)
        - r["accuracy_mean"], axis=1)
    summary["drop_vs_clean_same_agg"] = summary.apply(
        lambda r: lookup((r["dataset"], r["partition"], r["aggregator"]),
                         clean_same) - r["accuracy_mean"], axis=1)
    return summary.sort_values(GROUP_COLS).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--output-dir", default="results/processed")
    args = parser.parse_args()

    runs = load_runs(args.results_dir)
    if runs.empty:
        print("No hay ejecuciones completadas en", args.results_dir)
        return
    rounds = load_rounds(args.results_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rounds.to_csv(output_dir / "all_rounds.csv", index=False)
    summary = summarize(runs)
    summary.to_csv(output_dir / "final_summary.csv", index=False)
    print(f"{len(runs)} ejecuciones consolidadas "
          f"({summary.shape[0]} configuraciones)")
    print(f"  -> {output_dir / 'all_rounds.csv'}")
    print(f"  -> {output_dir / 'final_summary.csv'}")


if __name__ == "__main__":
    main()
