"""Carga y preprocesamiento de los datasets del estudio.

Datasets estandar de clasificacion de imagenes (seccion 3.10): MNIST como
dataset de depuracion, Fashion-MNIST como dataset principal y CIFAR-10 como
extension opcional. El preprocesamiento (normalizacion) es fijo para todos
los escenarios comparables.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets

DATASET_INFO = {
    "mnist": {
        "num_classes": 10,
        "in_channels": 1,
        "mean": (0.1307,),
        "std": (0.3081,),
        "class_names": [str(i) for i in range(10)],
    },
    "fashion_mnist": {
        "num_classes": 10,
        "in_channels": 1,
        "mean": (0.2860,),
        "std": (0.3530,),
        "class_names": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
    },
    "cifar10": {
        "num_classes": 10,
        "in_channels": 3,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "class_names": ["airplane", "automobile", "bird", "cat", "deer",
                        "dog", "frog", "horse", "ship", "truck"],
    },
}


class InMemoryDataset(Dataset):
    """Dataset de imagenes preprocesado integramente en memoria.

    El preprocesamiento (conversion a float y normalizacion) se aplica una
    unica vez al cargar, en lugar de muestra a muestra en cada epoca. Esto
    elimina el cuello de botella de decodificacion por lote y reduce varias
    veces el tiempo por ronda federada, lo cual es determinante para que la
    matriz experimental completa (seccion 3.12) sea viable en el hardware
    disponible. MNIST/Fashion-MNIST ocupan ~190 MB como float32.
    """

    def __init__(self, images: torch.Tensor, targets: np.ndarray):
        self.data = images          # (N, C, H, W) float32 ya normalizado
        self.targets = targets      # (N,) int64

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], int(self.targets[idx])


def _to_memory(ds, name: str) -> InMemoryDataset:
    """Convierte un dataset torchvision al formato tensorial usado internamente."""
    info = DATASET_INFO[name]
    if name in ("mnist", "fashion_mnist"):
        images = ds.data.unsqueeze(1).float().div_(255.0)   # (N,1,28,28)
    else:  # cifar10: ds.data es numpy (N,32,32,3) uint8
        images = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(info["mean"]).view(1, -1, 1, 1)
    std = torch.tensor(info["std"]).view(1, -1, 1, 1)
    images = (images - mean) / std
    return InMemoryDataset(images, get_targets(ds))


def load_datasets(name: str, data_dir: str | Path = "data"):
    """Devuelve (train_dataset, test_dataset) normalizados y en memoria.

    Descarga el dataset la primera vez que se usa. El conjunto de prueba es
    el oficial de cada dataset y se utiliza centralizado e identico en todos
    los escenarios (seccion 3.13).
    """
    if name not in DATASET_INFO:
        raise ValueError(f"Dataset no soportado: {name}")
    data_dir = str(data_dir)
    if name == "mnist":
        train = datasets.MNIST(data_dir, train=True, download=True)
        test = datasets.MNIST(data_dir, train=False, download=True)
    elif name == "fashion_mnist":
        train = datasets.FashionMNIST(data_dir, train=True, download=True)
        test = datasets.FashionMNIST(data_dir, train=False, download=True)
    else:
        train = datasets.CIFAR10(data_dir, train=True, download=True)
        test = datasets.CIFAR10(data_dir, train=False, download=True)
    return _to_memory(train, name), _to_memory(test, name)


def get_targets(dataset: Dataset) -> np.ndarray:
    """Extrae el vector de etiquetas de un dataset de torchvision."""
    targets = dataset.targets
    if isinstance(targets, torch.Tensor):
        return targets.numpy().copy()
    return np.asarray(targets).copy()


class TensorBatchLoader:
    """Iterador de lotes mediante slicing vectorizado de tensores.

    Sustituye al DataLoader estandar para datasets en memoria: en lugar de
    invocar __getitem__ muestra a muestra y ensamblar el lote en Python,
    genera una permutacion por epoca (torch.randperm con generador propio,
    la misma secuencia que produce el RandomSampler de un DataLoader con el
    mismo generador) y extrae cada lote con una unica operacion de indexado.
    Si los tensores residen en GPU, los lotes nunca pasan por la CPU.
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor,
                 batch_size: int, shuffle: bool = False,
                 generator: torch.Generator | None = None,
                 indices: torch.Tensor | None = None):
        self.images = images
        self.labels = labels
        # `indices` permite iterar una particion (vista de cliente) sin
        # copiar los datos: se indexa el tensor global lote a lote.
        self.indices = indices
        self.num_samples = len(indices) if indices is not None else len(images)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator

    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            order = torch.randperm(self.num_samples, generator=self.generator)
        else:
            order = torch.arange(self.num_samples)
        for chunk in order.split(self.batch_size):
            if self.indices is not None:
                # images se indexa sobre el tensor global; labels ya contiene
                # solo las etiquetas locales del cliente, posiblemente
                # envenenadas, y se indexa por posicion local.
                images = self.images[self.indices[chunk]]
                labels = self.labels[chunk]
            else:
                images = self.images[chunk]
                labels = self.labels[chunk]
            yield images, labels


class ClientDataset(Dataset):
    """Vista local de un cliente sobre el dataset de entrenamiento global.

    Mantiene su propia copia de las etiquetas, de modo que un ataque de
    envenenamiento de datos puede modificar las etiquetas de un cliente
    malicioso sin afectar a los datos de los clientes benignos (restriccion
    del modelo de amenaza, seccion 3.7).
    """

    def __init__(self, base: Dataset, indices: np.ndarray):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        base_targets = get_targets(base)
        # Copia defensiva: un ataque de datos solo puede alterar la vista local
        # del cliente malicioso, nunca las etiquetas globales compartidas.
        self.labels = base_targets[self.indices].copy()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, _ = self.base[int(self.indices[idx])]
        return image, int(self.labels[idx])
