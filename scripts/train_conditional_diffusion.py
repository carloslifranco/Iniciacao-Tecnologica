import math
import time

import torch

from conditional_diffusion_model import ConditionalDenoisingDiffusion
from train_diffusion import synchronize
from train_utils import save_checkpoint, save_samples, start_experiment


# Rótulos 0 a 9 do Fashion-MNIST, na ordem do dataset. Nomes oficiais:
# T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.
CLASS_NAMES = (
    "camiseta", "calca", "pulover", "vestido", "casaco",
    "sandalia", "camisa", "tenis", "bolsa", "bota",
)


def main() -> None:
    experiment = start_experiment("conditional_diffusion")
    with experiment.recorder.track():
        timesteps = int(experiment.model_config.get("timesteps", 100))
        samples_per_class = int(experiment.model_config.get("samples", 8))
        guidance_scale = float(experiment.model_config.get("guidance_scale", 3.0))
        # Ausente nas configurações antigas: False reproduz o amostrador original.
        clip_denoised = bool(experiment.model_config.get("clip_denoised", False))
        constructor = {
            "timesteps": timesteps,
            "num_classes": len(CLASS_NAMES),
            "input_channels": 1,
            "channels": int(experiment.model_config.get("channels", 32)),
            "label_dropout": float(experiment.model_config.get("label_dropout", 0.1)),
        }
        if guidance_scale < 0:
            raise ValueError("guidance_scale deve ser maior ou igual a 0")
        if guidance_scale != 1 and constructor["label_dropout"] == 0:
            raise ValueError(
                "guidance_scale diferente de 1 requer label_dropout > 0; "
                "sem isso a previsão sem classe nunca é treinada"
            )
        model = ConditionalDenoisingDiffusion(**constructor).to(experiment.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=experiment.learning_rate)
        recorder = experiment.recorder
        sampling_seed = experiment.seed + 1
        # Uma linha da grade por classe, na ordem dos rótulos.
        sample_labels = torch.arange(
            len(CLASS_NAMES), device=experiment.device
        ).repeat_interleave(samples_per_class)

        subset = experiment.dataloader.dataset
        used_targets = torch.as_tensor(subset.dataset.targets)[list(subset.indices)]
        recorder.metadata["dataset"]["model_input_transform"] = "x * 2 - 1: [-1,1]"
        recorder.metadata["dataset"]["class_counts"] = dict(zip(
            CLASS_NAMES,
            torch.bincount(used_targets, minlength=len(CLASS_NAMES)).tolist(),
        ))
        recorder.metadata["conditioning"] = {
            "type": "class label",
            "class_names": list(CLASS_NAMES),
            "null_label": len(CLASS_NAMES),
            "label_dropout": constructor["label_dropout"],
            "guidance": "eps_uncond + guidance_scale * (eps_cond - eps_uncond)",
        }
        recorder.metadata["sampling"] = {
            "seed": sampling_seed,
            "samples_per_class": samples_per_class,
            "samples_per_epoch": len(sample_labels),
            "grid": "one row per class, in label order",
            "guidance_scale": guidance_scale,
            "clip_denoised": clip_denoised,
            "spatial_size": 28,
            "same_random_sequence_each_epoch": True,
            "isolated_from_training_rng": True,
        }
        recorder.metadata["metrics"] = {
            "train_noise_mse": "Mean squared noise prediction error, weighted by batch size; not image quality.",
            "train_seconds": "Training loop, including data loading and optimizer updates.",
            "sampling_seconds": "Sampling and PNG saving.",
            "elapsed_seconds": "Time since recorder creation, including this epoch checkpoint saving.",
        }
        recorder.record_model(model, constructor, optimizer)

        for epoch in range(1, experiment.epochs + 1):
            synchronize(experiment.device)
            started = time.perf_counter()
            model.train()
            total_loss = 0.0
            seen = 0
            for images, labels in experiment.dataloader:
                clean_images = images.to(experiment.device).mul(2).sub(1)
                labels = labels.to(experiment.device)
                optimizer.zero_grad()
                loss = model.training_loss(clean_images, labels)
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    raise FloatingPointError(f"Perda não finita na época {epoch}: {loss_value}")
                loss.backward()
                optimizer.step()
                total_loss += loss_value * len(images)
                seen += len(images)

            synchronize(experiment.device)
            train_seconds = time.perf_counter() - started
            mean_loss = total_loss / seen
            print(
                f"Época {epoch:02d}/{experiment.epochs} | Ruído MSE: {mean_loss:.4f}"
            )
            started = time.perf_counter()
            model.eval()
            # Repeat the complete random sampling sequence without changing training RNG.
            cuda_devices = [experiment.device.index or 0] if experiment.device.type == "cuda" else []
            with torch.random.fork_rng(devices=cuda_devices):
                torch.random.default_generator.manual_seed(sampling_seed)
                if experiment.device.type == "cuda":
                    torch.cuda.manual_seed(sampling_seed)
                generated = model.sample(
                    sample_labels, guidance_scale, clip_denoised=clip_denoised
                )
                if not torch.isfinite(generated).all().item():
                    raise FloatingPointError(f"Amostras não finitas na época {epoch}")
                save_samples(
                    (generated.clamp(-1, 1) + 1) / 2,
                    experiment.output_dir,
                    epoch,
                    nrow=samples_per_class,
                )
            synchronize(experiment.device)
            sampling_seconds = time.perf_counter() - started
            save_checkpoint(
                experiment.output_dir,
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_constructor": constructor,
                    "class_names": list(CLASS_NAMES),
                    "guidance_scale": guidance_scale,
                    "clip_denoised": clip_denoised,
                    "timesteps": timesteps,
                    "epochs": epoch,
                    "seed": experiment.seed,
                    "sampling_seed": sampling_seed,
                },
            )
            recorder.log_epoch({
                "epoch": epoch,
                "train_noise_mse": mean_loss,
                "train_seconds": train_seconds,
                "sampling_seconds": sampling_seconds,
                "elapsed_seconds": time.perf_counter() - recorder.started,
            })


if __name__ == "__main__":
    main()
