from __future__ import annotations

import argparse
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
    args = parser.parse_args()

    seed = int(experiment_config.get("seed", 42))
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
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = OUTPUT_ROOT / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

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
