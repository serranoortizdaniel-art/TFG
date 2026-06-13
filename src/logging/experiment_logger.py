"""Gestion del directorio de resultados de una ejecucion.

Estructura por ejecucion (seccion 3.14):

results/raw/<experiment_name>/seed_<seed>/
    config.yaml             configuracion efectiva completa
    metrics.csv             metricas por ronda
    update_norms.csv        norma de la actualizacion de cada cliente y ronda
    class_distribution.csv  muestras por clase y cliente de la particion
    confusion_matrix.csv    matriz de confusion del modelo final
    metadata.json           estado, duracion, hardware, resumen final

`metadata.json` contiene `status` ("running" -> "completed"), que es el
marcador que usa run_all.py para reanudar la matriz sin repetir
ejecuciones ya completadas.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.logging.metrics_logger import MetricsLogger, UpdateNormsLogger
from src.utils.config import experiment_name, save_config
from src.utils.hardware import hardware_info


class ExperimentLogger:
    """Propietario de todos los artefactos de una ejecucion concreta."""

    def __init__(self, cfg: dict, num_classes: int):
        self.cfg = cfg
        self.name = experiment_name(cfg)
        self.run_dir = Path(cfg["output_dir"]) / self.name / f"seed_{cfg['seed']}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._start = time.perf_counter()

        save_config(cfg, self.run_dir / "config.yaml")
        self.metrics = MetricsLogger(self.run_dir / "metrics.csv", num_classes)
        self.norms = UpdateNormsLogger(self.run_dir / "update_norms.csv")

        self.metadata = {
            "experiment_name": self.name,
            "seed": cfg["seed"],
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            # Campos clave duplicados de la config para que el analisis pueda
            # filtrar y agrupar leyendo solo metadata.json, sin parsear YAML.
            "dataset": cfg["dataset"],
            "partition_type": cfg["partition_type"],
            "dirichlet_alpha": cfg["dirichlet_alpha"] if cfg["partition_type"] == "dirichlet" else None,
            "attack_type": cfg["attack_type"],
            "malicious_fraction": cfg["malicious_fraction"],
            "aggregator": cfg["aggregator"],
            "num_clients": cfg["num_clients"],
            "num_rounds": cfg["num_rounds"],
            "hardware": hardware_info(),
        }
        self._write_metadata()

    def _write_metadata(self) -> None:
        """Escribe metadata.json de forma inmediata tras cada cambio relevante."""
        with open(self.run_dir / "metadata.json", "w", encoding="utf-8") as fh:
            json.dump(self.metadata, fh, indent=2, ensure_ascii=False)

    def set_info(self, **kwargs) -> None:
        """Anade campos informativos a los metadatos (maliciosos, defensa...)."""
        self.metadata.update(kwargs)
        self._write_metadata()

    def save_class_distribution(self, dist: np.ndarray,
                                malicious_ids: list[int]) -> None:
        """Guarda la particion real usada por los clientes de la ejecucion."""
        num_classes = dist.shape[1]
        header = "client_id,is_malicious," + ",".join(
            f"class_{c}" for c in range(num_classes))
        lines = [header]
        for client_id, row in enumerate(dist):
            lines.append(f"{client_id},{int(client_id in malicious_ids)},"
                         + ",".join(str(v) for v in row))
        path = self.run_dir / "class_distribution.csv"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def save_confusion_matrix(self, cm: np.ndarray) -> None:
        """Guarda la matriz de confusion final sin normalizar."""
        np.savetxt(self.run_dir / "confusion_matrix.csv", cm,
                   fmt="%d", delimiter=",")

    def finish(self, final_metrics: dict) -> None:
        """Marca la ejecucion como completada y guarda el resumen final."""
        self.metrics.close()
        self.norms.close()
        self.metadata.update({
            "status": "completed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(time.perf_counter() - self._start, 2),
            "final_test_accuracy": final_metrics["accuracy"],
            "final_test_loss": final_metrics["loss"],
            "final_macro_f1": final_metrics["macro_f1"],
        })
        self._write_metadata()

    @staticmethod
    def is_completed(run_dir: str | Path,
                     required_engine: str | None = None) -> bool:
        """True si el directorio contiene una ejecucion terminada compatible.

        Es tolerante a metadatos corruptos o incompletos: en caso de duda se
        considera pendiente para que run_all.py pueda rehacer la ejecucion.
        """
        meta_path = Path(run_dir) / "metadata.json"
        if not meta_path.exists():
            return False
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                metadata = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return False
        if metadata.get("status") != "completed":
            return False
        if required_engine is not None and metadata.get("engine") != required_engine:
            return False
        return True
