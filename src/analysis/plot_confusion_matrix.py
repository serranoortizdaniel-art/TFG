"""Matrices de confusion del modelo final (heatmap).

Promedia las matrices de confusion sobre las semillas disponibles de cada
configuracion. Es la figura clave para el analisis del label flipping
(confusiones clase origen -> destino y patron de superdiagonal con la
politica all_to_next, seccion 3.13).

Uso:
    python -m src.analysis.plot_confusion_matrix                # todas las atacadas con label_flipping
    python -m src.analysis.plot_confusion_matrix --all          # todas las configuraciones
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.common import (ATTACK_LABELS, iter_runs, load_runs,
                                 partition_label, save_figure)
from src.datasets.loader import DATASET_INFO


def plot_cm(cm: np.ndarray, title: str, class_names: list[str]):
    """Dibuja una matriz de confusion ya agregada para una configuracion."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    num_classes = cm.shape[0]
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    short = [n[:10] for n in class_names]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    # Solo se anotan celdas relevantes (>1% de la clase real) para evitar que
    # los heatmaps de 10 clases queden ilegibles en la memoria.
    row_sums = cm.sum(axis=1, keepdims=True)
    for i in range(num_classes):
        for j in range(num_classes):
            if cm[i, j] > 0.01 * max(row_sums[i, 0], 1):
                color = "white" if cm[i, j] > cm.max() * 0.6 else "black"
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        fontsize=6, color=color)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(title, fontsize=9)
    ax.grid(False)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--output-dir", default="plots/confusion")
    parser.add_argument("--all", action="store_true",
                        help="Generar tambien las configuraciones sin ataque")
    args = parser.parse_args()

    # agrupa matrices por configuracion (promediando semillas)
    grouped: dict[str, list] = {}
    meta_by_name: dict[str, dict] = {}
    for metadata, run_dir in iter_runs(args.results_dir):
        cm_path = Path(run_dir) / "confusion_matrix.csv"
        if not cm_path.exists():
            continue
        if not args.all and metadata["attack_type"] == "none":
            continue
        name = metadata["experiment_name"]
        grouped.setdefault(name, []).append(
            np.loadtxt(cm_path, delimiter=",", dtype=float))
        meta_by_name[name] = metadata

    count = 0
    for name, matrices in grouped.items():
        metadata = meta_by_name[name]
        cm = np.mean(matrices, axis=0)
        class_names = DATASET_INFO[metadata["dataset"]]["class_names"]
        attack_label = ATTACK_LABELS.get(metadata["attack_type"],
                                         metadata["attack_type"])
        pct = int(round(metadata["malicious_fraction"] * 100))
        seeds_label = ("1 semilla" if len(matrices) == 1
                       else f"media de {len(matrices)} semillas")
        title = (f"{attack_label} ({pct}% maliciosos), "
                 f"{metadata['aggregator']} - {metadata['dataset']} "
                 f"{partition_label(metadata)} ({seeds_label})")
        fig = plot_cm(cm, title, class_names)
        save_figure(fig, args.output_dir, f"cm_{name}")
        count += 1
    print(f"{count} matrices de confusion en {args.output_dir}")


if __name__ == "__main__":
    main()
