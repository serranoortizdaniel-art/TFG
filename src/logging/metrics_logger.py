"""Loggers de metricas en CSV.

Cada experimento produce ficheros CSV estructurados (seccion 3.14):
  - metrics.csv: una fila por ronda con todas las metricas de evaluacion.
  - update_norms.csv: una fila por cliente y ronda con la norma L2 de su
    actualizacion (instrumentacion para detectar actualizaciones anomalas).

Las filas se escriben y vuelcan a disco al final de cada ronda, de modo que
una ejecucion interrumpida conserva todas las rondas completadas.
"""

from __future__ import annotations

import csv
from pathlib import Path


class CSVLogger:
    """Escritor CSV incremental con cabecera fija.

    La cabecera fija mantiene los artefactos consumibles por pandas y LaTeX
    aunque una ejecucion se interrumpa a mitad de la matriz.
    """

    def __init__(self, path: str | Path, fieldnames: list[str]):
        self.path = Path(path)
        self.fieldnames = fieldnames
        self._fh = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames)
        self._writer.writeheader()

    def log(self, row: dict) -> None:
        self._writer.writerow(row)
        # Flush por fila: se sacrifica un poco de rendimiento para conservar el
        # progreso si la simulacion se interrumpe durante una tirada larga.
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


class MetricsLogger(CSVLogger):
    """Metricas del modelo global por ronda o por epoca en centralizado."""

    def __init__(self, path: str | Path, num_classes: int,
                 round_field: str = "round"):
        self.num_classes = num_classes
        self.round_field = round_field
        fields = [round_field, "test_loss", "test_accuracy", "macro_f1",
                  "train_loss", "round_time_s", "agg_time_s"]
        fields += [f"acc_class_{c}" for c in range(num_classes)]
        super().__init__(path, fields)

    def log_round(self, round_num: int, eval_metrics: dict,
                  train_loss: float | None = None,
                  round_time: float | None = None,
                  agg_time: float | None = None) -> None:
        """Normaliza el dict de metricas al esquema CSV estable."""
        row = {
            self.round_field: round_num,
            "test_loss": f"{eval_metrics['loss']:.6f}",
            "test_accuracy": f"{eval_metrics['accuracy']:.6f}",
            "macro_f1": f"{eval_metrics['macro_f1']:.6f}",
            "train_loss": "" if train_loss is None else f"{train_loss:.6f}",
            "round_time_s": "" if round_time is None else f"{round_time:.3f}",
            "agg_time_s": "" if agg_time is None else f"{agg_time:.4f}",
        }
        for c, acc in enumerate(eval_metrics["per_class_accuracy"]):
            row[f"acc_class_{c}"] = f"{acc:.6f}"
        self.log(row)


class UpdateNormsLogger(CSVLogger):
    """Norma L2 de la actualizacion de cada cliente en cada ronda."""

    def __init__(self, path: str | Path):
        super().__init__(path, ["round", "client_id", "is_malicious",
                                "num_examples", "update_norm"])

    def log_update(self, round_num: int, client_id: int, is_malicious: bool,
                   num_examples: int, norm: float) -> None:
        """Registra una fila de instrumentacion por cliente y ronda."""
        self.log({
            "round": round_num,
            "client_id": client_id,
            "is_malicious": int(is_malicious),
            "num_examples": num_examples,
            "update_norm": f"{norm:.6f}",
        })
