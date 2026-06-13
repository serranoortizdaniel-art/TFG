"""Entrenamiento centralizado de referencia (fase 1 del protocolo, 3.11).

Entrena la misma arquitectura, con el mismo preprocesamiento y las mismas
metricas que el escenario federado, pero sobre el dataset completo en una
sola maquina. Su accuracy final actua como referencia superior aproximada
para interpretar los resultados federados.

Uso:
    python train_centralized.py --dataset fashion_mnist --epochs 10
    python train_centralized.py --dataset mnist --epochs 5 --seed 123
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.datasets.loader import DATASET_INFO, TensorBatchLoader, load_datasets
from src.logging.metrics_logger import MetricsLogger
from src.models.cnn import build_model
from src.models.training import train_epochs
from src.utils.hardware import hardware_info
from src.utils.metrics import evaluate
from src.utils.seed import derive_seed, seeded_generator, set_global_seed


def main() -> None:
    """Entrena y registra la referencia centralizada de un dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="fashion_mnist",
                        choices=list(DATASET_INFO))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results/raw")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = DATASET_INFO[args.dataset]["num_classes"]

    train_ds, test_ds = load_datasets(args.dataset, args.data_dir)
    train_loader = TensorBatchLoader(
        images=train_ds.data.to(device),
        labels=torch.as_tensor(train_ds.targets, dtype=torch.long, device=device),
        batch_size=args.batch_size, shuffle=True,
        generator=seeded_generator(derive_seed(args.seed, "central_shuffle")))
    test_loader = TensorBatchLoader(
        images=test_ds.data.to(device),
        labels=torch.as_tensor(test_ds.targets, dtype=torch.long, device=device),
        batch_size=512, shuffle=False)

    model = build_model(args.dataset).to(device)
    run_dir = Path(args.output_dir) / f"centralized_{args.dataset}" / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = MetricsLogger(run_dir / "metrics.csv", num_classes,
                           round_field="epoch")

    print(f"[centralizado {args.dataset} | seed {args.seed}] "
          f"{args.epochs} epocas, dispositivo={device}")
    start = time.perf_counter()
    # Epoca 0: rendimiento del modelo inicial. Sirve como control de que las
    # curvas empiezan desde la misma inicializacion semillada.
    metrics = evaluate(model, test_loader, device, num_classes)
    logger.log_round(0, metrics)
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_epochs(model, train_loader, epochs=1, lr=args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay, device=device)
        metrics = evaluate(model, test_loader, device, num_classes)
        epoch_time = time.perf_counter() - t0
        logger.log_round(epoch, metrics, train_loss=train_loss,
                         round_time=epoch_time)
        print(f"  epoca {epoch:2d}/{args.epochs}  "
              f"acc={metrics['accuracy']:.4f}  loss={metrics['loss']:.4f}  "
              f"macroF1={metrics['macro_f1']:.4f}  ({epoch_time:.1f}s)")
    logger.close()

    duration = time.perf_counter() - start
    metadata = {
        "experiment_name": f"centralized_{args.dataset}",
        "type": "centralized",
        "status": "completed",
        "dataset": args.dataset,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": round(duration, 2),
        "final_test_accuracy": metrics["accuracy"],
        "final_test_loss": metrics["loss"],
        "final_macro_f1": metrics["macro_f1"],
        "hardware": hardware_info(),
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    # Import local para no cargar numpy antes de necesitar escribir este unico
    # artefacto al final del baseline.
    import numpy as np
    np.savetxt(run_dir / "confusion_matrix.csv",
               metrics["confusion_matrix"], fmt="%d", delimiter=",")
    print(f"  completado en {duration:.1f}s: acc final={metrics['accuracy']:.4f} "
          f"-> {run_dir}")


if __name__ == "__main__":
    main()
