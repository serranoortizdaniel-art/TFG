"""Genera de una vez todos los productos de analisis (fase 9).

Equivale a ejecutar en orden:
    python -m src.analysis.aggregate_results
    python -m src.analysis.plot_accuracy
    python -m src.analysis.plot_loss
    python -m src.analysis.plot_confusion_matrix
    python -m src.analysis.plot_update_norms
    python -m src.analysis.plot_client_distribution
    python -m src.analysis.generate_tables

Uso:  python run_analysis.py [--results-dir results/raw]
"""

from __future__ import annotations

import argparse
import sys

from src.analysis import (aggregate_results, generate_tables, plot_accuracy,
                          plot_client_distribution, plot_confusion_matrix,
                          plot_loss, plot_update_norms)


def main() -> None:
    """Ejecuta la tuberia completa de analisis sobre un results-dir."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/raw")
    args = parser.parse_args()

    # Cada modulo conserva su CLI propia y lee sys.argv. Este lanzador fija
    # argumentos comunes para reutilizar esos entrypoints sin duplicar codigo.
    sys.argv = [sys.argv[0], "--results-dir", args.results_dir]
    for module in (aggregate_results, plot_accuracy, plot_loss,
                   plot_confusion_matrix, plot_update_norms,
                   plot_client_distribution):
        module.main()
    sys.argv = [sys.argv[0]]
    generate_tables.main()


if __name__ == "__main__":
    main()
