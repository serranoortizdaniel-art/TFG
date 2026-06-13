"""Distribucion de clases por cliente (barras apiladas).

Documenta la heterogeneidad del particionado de cada experimento
(seccion 3.10): en IID las barras son uniformes; con Dirichlet se aprecia
el desequilibrio creciente al reducir alpha. Marca con * los clientes
maliciosos.

Uso:
    python -m src.analysis.plot_client_distribution --run results/raw/<exp>/seed_42
    python -m src.analysis.plot_client_distribution            # una por configuracion
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.common import iter_runs, partition_label, save_figure


def plot_distribution(run_dir: Path, title: str):
    """Dibuja la distribucion de clases guardada para una ejecucion."""
    df = pd.read_csv(run_dir / "class_distribution.csv")
    class_cols = [c for c in df.columns if c.startswith("class_")]
    counts = df[class_cols].to_numpy()
    num_clients, num_classes = counts.shape

    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(num_clients)
    cmap = plt.get_cmap("tab10")
    for cls in range(num_classes):
        ax.bar(range(num_clients), counts[:, cls], bottom=bottom,
               color=cmap(cls % 10), label=f"{cls}", width=0.8)
        bottom += counts[:, cls]
    # El asterisco marca maliciosos solo como ayuda de lectura de la figura; no
    # implica que el agregador haya recibido esa informacion.
    labels = [f"{cid}*" if mal else str(cid)
              for cid, mal in zip(df["client_id"], df["is_malicious"])]
    ax.set_xticks(range(num_clients))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Cliente (* = malicioso)")
    ax.set_ylabel("Numero de muestras")
    ax.set_title(title, fontsize=9)
    ax.legend(title="Clase", fontsize=7, ncols=2)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--output-dir", default="plots/distribution")
    parser.add_argument("--run", default=None,
                        help="Directorio de una ejecucion concreta")
    args = parser.parse_args()

    if args.run:
        run_dir = Path(args.run)
        fig = plot_distribution(run_dir, run_dir.parent.name)
        paths = save_figure(fig, args.output_dir,
                            f"dist_{run_dir.parent.name}_{run_dir.name}")
        print(f"figura guardada: {paths[0]}")
        return

    # Una figura por combinacion (dataset, particion): la distribucion es la
    # misma para todos los escenarios con la misma semilla y particion.
    seen = set()
    count = 0
    for metadata, run_dir in iter_runs(args.results_dir):
        key = (metadata["dataset"], partition_label(metadata),
               metadata["seed"], metadata["malicious_fraction"])
        if key in seen or not (Path(run_dir) / "class_distribution.csv").exists():
            continue
        seen.add(key)
        pct = int(round(metadata["malicious_fraction"] * 100))
        title = (f"{metadata['dataset']} {partition_label(metadata)} "
                 f"(seed {metadata['seed']}, {pct}% maliciosos)")
        fig = plot_distribution(Path(run_dir), title)
        name = (f"dist_{metadata['dataset']}_{partition_label(metadata)}"
                f"_m{pct}_seed{metadata['seed']}")
        save_figure(fig, args.output_dir, name)
        count += 1
    print(f"{count} figuras de distribucion en {args.output_dir}")


if __name__ == "__main__":
    main()
