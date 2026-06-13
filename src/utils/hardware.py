"""Recogida de metadatos de hardware y software para la reproducibilidad."""

from __future__ import annotations

import os
import platform
import sys

import numpy
import torch


def hardware_info() -> dict:
    """Devuelve un diccionario con la informacion del entorno de ejecucion.

    Se guarda en los metadatos de cada experimento para poder documentar en
    la memoria el hardware real utilizado y detectar diferencias de entorno
    entre ejecuciones.
    """
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "numpy_version": numpy.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["gpu_memory_gb"] = round(props.total_memory / 1024 ** 3, 2)
        info["cuda_version"] = torch.version.cuda
    return info
