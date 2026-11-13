import math
import time

import torch

from diffusion_model import DenoisingDiffusion
from train_utils import save_checkpoint, save_samples, start_experiment


def main() -> None:
    experiment = start_experiment("diffusion")
    with experiment.recorder.track():
        timesteps = int(experiment.model_config.get("timesteps", 100))
        sample_count = int(experiment.model_config.get("samples", 64))
        # Ausente nas configurações antigas: False reproduz o amostrador original.
        clip_denoised = bool(experiment.model_config.get("clip_denoised", False))
        constructor = {
            "timesteps": timesteps,
            "input_channels": 1,
            "channels": int(experiment.model_config.get("channels", 32)),
        }
        model = DenoisingDiffusion(**constructor).to(experiment.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=experiment.learning_rate)
        recorder = experiment.recorder
        sampling_seed = experiment.seed + 1
        recorder.metadata["sampling"] = {
            "seed": sampling_seed,
            "samples_per_epoch": sample_count,
            "clip_denoised": clip_denoised,
            "spatial_size": 28,
            "same_random_sequence_each_epoch": True,
            "isolated_from_training_rng": True,
        }
        recorder.metadata["dataset"]["model_input_transform"] = "x * 2 - 1: [-1,1]"
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
            for images, _ in experiment.dataloader:
                clean_images = images.to(experiment.device).mul(2).sub(1)
                optimizer.zero_grad()
                loss = model.training_loss(clean_images)
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
                    sample_count, experiment.device, clip_denoised=clip_denoised
                )
                if not torch.isfinite(generated).all().item():
                    raise FloatingPointError(f"Amostras não finitas na época {epoch}")
                save_samples(
                    (generated.clamp(-1, 1) + 1) / 2,
                    experiment.output_dir,
                    epoch,
                )
            synchronize(experiment.device)
            sampling_seconds = time.perf_counter() - started
            save_checkpoint(
                experiment.output_dir,
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_constructor": constructor,
                    "timesteps": timesteps,
                    "clip_denoised": clip_denoised,
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


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()
