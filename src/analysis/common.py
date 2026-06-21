"""Utilidades comunes de analisis: carga de resultados y estilo grafico.

Todas las figuras y tablas de la memoria se generan por script a partir de
los ficheros de resultados, sin introducir datos a mano (seccion 3.14).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd

# Backend no interactivo: los scripts de analisis deben funcionar en WSL,
# terminales remotas y CI, sin depender de una ventana grafica.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Estilo homogeneo de todas las figuras de la memoria. Centralizarlo evita que
# cada script produzca una estetica distinta al regenerar resultados.
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.constrained_layout.use": True,
})

AGGREGATOR_COLORS = {
    "fedavg": "#1f77b4",
    "median": "#2ca02c",
    "trimmed_mean": "#ff7f0e",
    "krum": "#d62728",
}
AGGREGATOR_LABELS = {
    "fedavg": "FedAvg",
    "median": "Mediana coordinada",
    "trimmed_mean": "Trimmed Mean",
    "krum": "Krum",
}
ATTACK_LABELS = {
    "none": "Sin ataque",
    "label_flipping": "Label flipping",
    "sign_flipping": "Sign flipping",
    "scaling": "Scaling",
}


def partition_label(metadata: dict) -> str:
    """Etiqueta corta de la particion: 'iid' o 'dir<alpha>'."""
    if metadata.get("partition_type") == "dirichlet":
        return f"dir{metadata.get('dirichlet_alpha')}"
    return "iid"


def iter_runs(results_dir: str | Path = "results/raw"):
    """Itera sobre las ejecuciones federadas completadas: (metadata, run_dir)."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return
    for meta_path in sorted(results_dir.glob("*/seed_*/metadata.json")):
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                metadata = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if metadata.get("status") != "completed":
            continue
        if metadata.get("type") == "centralized":
            continue
        yield metadata, meta_path.parent


def load_runs(results_dir: str | Path = "results/raw") -> pd.DataFrame:
    """DataFrame con una fila por ejecucion completada (metricas finales)."""
    rows = []
    for metadata, run_dir in iter_runs(results_dir):
        rows.append({
            "experiment": metadata["experiment_name"],
            "dataset": metadata["dataset"],
            "partition": partition_label(metadata),
            "attack": metadata["attack_type"],
            "malicious_pct": int(round(metadata["malicious_fraction"] * 100)),
            "aggregator": metadata["aggregator"],
            # Politica de defensa (fija/oracle): distingue configuraciones que
            # comparten agregador pero ajustan sus parametros de forma distinta.
            "policy": metadata.get("defense_policy", {}).get("policy", "fixed"),
            "seed": metadata["seed"],
            "num_rounds": metadata["num_rounds"],
            "final_accuracy": metadata["final_test_accuracy"],
            "final_loss": metadata["final_test_loss"],
            "final_macro_f1": metadata["final_macro_f1"],
            "duration_s": metadata.get("duration_s"),
            "run_dir": str(run_dir),
        })
    return pd.DataFrame(rows)


def load_rounds(results_dir: str | Path = "results/raw") -> pd.DataFrame:
    """DataFrame largo con una fila por ejecucion y ronda.

    El formato largo facilita calcular medias sobre semillas y dibujar curvas
    por agregador sin reestructurar datos en cada script de figura.
    """
    frames = []
    for metadata, run_dir in iter_runs(results_dir):
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        df = pd.read_csv(metrics_path)
        df["experiment"] = metadata["experiment_name"]
        df["dataset"] = metadata["dataset"]
        df["partition"] = partition_label(metadata)
        df["attack"] = metadata["attack_type"]
        df["malicious_pct"] = int(round(metadata["malicious_fraction"] * 100))
        df["aggregator"] = metadata["aggregator"]
        df["policy"] = metadata.get("defense_policy", {}).get("policy", "fixed")
        df["seed"] = metadata["seed"]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_figure(fig, output_dir: str | Path, name: str,
                formats: tuple[str, ...] = ("png", "pdf")) -> list[Path]:
    """Guarda la figura en PNG para revision y PDF para la memoria."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    return paths
