from __future__ import annotations

import argparse
import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from experiment_tracking import RunRecorder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "baseline.yml"
DATA_DIR = PROJECT_ROOT / "data" / "downloads"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


@dataclass
class Experiment:
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    model_config: dict[str, Any]
    dataloader: DataLoader
    device: torch.device
    output_dir: Path
    recorder: RunRecorder


def start_experiment(model_name: str) -> Experiment:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Configuração inválida em {CONFIG_PATH}")

    experiment_config = config.get("experiment", {})
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    models_config = config.get("models", {})

    if not all(
        isinstance(section, dict)
        for section in (experiment_config, data_config, training_config, models_config)
    ):
        raise ValueError(f"Seções da configuração inválidas em {CONFIG_PATH}")
    if data_config.get("dataset") != "fashion_mnist":
        raise ValueError("Este pipeline requer data.dataset: fashion_mnist")
    model_config = models_config.get(model_name)
    if not isinstance(model_config, dict):
        raise ValueError(f"Configuração ausente para o modelo {model_name!r}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--samples", type=int, help="Limita imagens para testes rápidos")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--generated-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deterministic", action="store_true", default=None)
    args = parser.parse_args()

    seed = int(args.seed if args.seed is not None else experiment_config.get("seed", 42))
    epochs = int(
        args.epochs if args.epochs is not None else training_config.get("epochs", 1)
    )
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else training_config.get("batch_size", 64)
    )
    sample_count = int(
        args.samples
        if args.samples is not None
        else data_config.get("samples", 5_000)
    )
    learning_rate = float(
        model_config.get(
            "learning_rate",
            training_config.get("learning_rate", 2e-4),
        )
    )
    if min(epochs, batch_size, sample_count) <= 0:
        raise ValueError("epochs, batch size e samples devem ser positivos")
    if args.generated_samples is not None:
        if args.generated_samples <= 0:
            raise ValueError("generated-samples deve ser positivo")
        model_config = {**model_config, "samples": args.generated_samples}
    if args.epochs is not None and "autoencoder_epochs" in model_config:
        model_config["autoencoder_epochs"] = epochs

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    deterministic = (
        args.deterministic
        if args.deterministic is not None
        else bool(training_config.get("deterministic", False))
    )
    if deterministic:
        # cuBLAS (camadas lineares na GPU) só é determinístico com esta variável,
        # definida antes da primeira operação CUDA; sem ela o PyTorch gera erro.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic

    dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        transform=transforms.ToTensor(),
        download=True,
    )
    dataset = Subset(dataset, range(min(sample_count, len(dataset))))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    effective_config = {
        "experiment": {**experiment_config, "seed": seed},
        "data": {
            **data_config,
            "samples": len(dataset),
            "image_size": 28,
            "channels": 1,
        },
        "training": {
            **training_config,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "deterministic": deterministic,
        },
        "models": {model_name: model_config},
    }
    fingerprint = hashlib.sha256()
    fingerprint.update(dataset.dataset.data[:len(dataset)].numpy().tobytes())
    fingerprint.update(dataset.dataset.targets[:len(dataset)].numpy().tobytes())
    recorder = RunRecorder(
        OUTPUT_ROOT, model_name, PROJECT_ROOT, CONFIG_PATH, effective_config,
        {
            "name": "FashionMNIST",
            "split": "train",
            "available_samples": len(dataset.dataset),
            "requested_samples": sample_count,
            "used_samples": len(dataset),
            "selection": "first N indices, shuffled at each epoch",
            "indices": {"start": 0, "stop_exclusive": len(dataset)},
            "selected_data_and_targets_sha256": fingerprint.hexdigest(),
            "transform": "ToTensor: uint8 [0,255] -> float32 [0,1]",
            "batch_size": batch_size,
            "batches_per_epoch": len(dataloader),
            "shuffle": True,
            "drop_last": False,
            "num_workers": 0,
            "dataloader_seed": seed,
        },
        device,
    )
    output_dir = recorder.output_dir

    print(
        f"Modelo: {model_name} | Dataset: Fashion-MNIST "
        f"({len(dataset)} imagens) | Épocas: {epochs} | "
        f"Dispositivo: {device}"
    )
    print(f"Saída: {output_dir}")

    return Experiment(
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        model_config=model_config,
        dataloader=dataloader,
        device=device,
        output_dir=output_dir,
        recorder=recorder,
    )


def save_samples(images: torch.Tensor, output_dir: Path, epoch: int) -> Path:
    output_path = output_dir / f"samples_epoch_{epoch:02d}.png"
    save_image(images.detach().cpu(), output_path, nrow=8)
    print(f"Amostras salvas: {output_path}")
    return output_path


def save_checkpoint(
    output_dir: Path,
    state: dict[str, Any],
) -> Path:
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(state, checkpoint_path)
    print(f"Checkpoint salvo: {checkpoint_path}")
    return checkpoint_path
