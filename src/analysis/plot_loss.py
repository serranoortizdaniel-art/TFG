"""Curvas de loss por ronda (media +- desviacion sobre semillas).

Reutiliza el mismo trazador que accuracy para mantener identico criterio de
agrupacion, leyendas y linea base entre figuras.

Uso:  python -m src.analysis.plot_loss [--results-dir results/raw]
"""

from __future__ import annotations

import argparse

from src.analysis.common import load_rounds
from src.analysis.plot_accuracy import plot_metric_curves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--output-dir", default="plots/loss")
    args = parser.parse_args()
    rounds = load_rounds(args.results_dir)
    if rounds.empty:
        print("No hay resultados que graficar")
        return
    n = plot_metric_curves(rounds, "test_loss", "Loss (test)",
                           args.output_dir, "loss")
    print(f"{n} figuras de loss en {args.output_dir}")


if __name__ == "__main__":
    main()
