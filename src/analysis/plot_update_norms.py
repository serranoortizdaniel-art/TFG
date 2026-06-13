"""Normas de las actualizaciones locales: clientes benignos vs maliciosos.

Para cada configuracion con ataque dibuja la evolucion por ronda de la
norma L2 de las actualizaciones, separando benignos (media + banda
min-max) y maliciosos (media + banda min-max). Permite analizar si el
ataque produce actualizaciones anomalas en magnitud y discutir por que
defensas basadas en distancias las filtran o no (seccion 3.13).

Uso:  python -m src.analysis.plot_update_norms [--results-dir results/raw]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.common import (ATTACK_LABELS, iter_runs, partition_label,
                                 save_figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--output-dir", default="plots/norms")
    parser.add_argument("--seed", type=int, default=None,
                        help="Semilla concreta a usar (por defecto la primera)")
    args = parser.parse_args()

    seen = set()
    count = 0
    for metadata, run_dir in iter_runs(args.results_dir):
        if metadata["attack_type"] == "none":
            continue
        name = metadata["experiment_name"]
        if args.seed is not None and metadata["seed"] != args.seed:
            continue
        if name in seen:  # una figura por configuracion (una semilla)
            continue
        norms_path = Path(run_dir) / "update_norms.csv"
        if not norms_path.exists():
            continue
        seen.add(name)

        df = pd.read_csv(norms_path)
        fig, ax = plt.subplots(figsize=(6, 4))
        for malicious, color, label in [(0, "#1f77b4", "Benignos"),
                                        (1, "#d62728", "Maliciosos")]:
            group = df[df["is_malicious"] == malicious]
            if group.empty:
                continue
            # Banda min-max entre clientes del mismo tipo: muestra dispersion
            # intra-ronda sin ocultar outliers individuales bajo la media.
            stats = group.groupby("round")["update_norm"].agg(
                ["mean", "min", "max"])
            ax.plot(stats.index, stats["mean"], color=color, label=label)
            ax.fill_between(stats.index, stats["min"], stats["max"],
                            color=color, alpha=0.15)
        attack_label = ATTACK_LABELS.get(metadata["attack_type"],
                                         metadata["attack_type"])
        pct = int(round(metadata["malicious_fraction"] * 100))
        ax.set_xlabel("Ronda federada")
        ax.set_ylabel("Norma L2 de la actualizacion")
        ax.set_title(f"{attack_label} ({pct}%), {metadata['aggregator']} - "
                     f"{metadata['dataset']} {partition_label(metadata)} "
                     f"(seed {metadata['seed']})", fontsize=9)
        ax.legend()
        save_figure(fig, args.output_dir, f"norms_{name}")
        count += 1
    print(f"{count} figuras de normas en {args.output_dir}")


if __name__ == "__main__":
    main()
