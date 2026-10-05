import torch

from diffusion_model import DenoisingDiffusion
from train_utils import save_checkpoint, save_samples, start_experiment


def main() -> None:
    experiment = start_experiment("diffusion")
    timesteps = int(experiment.model_config.get("timesteps", 100))
    sample_count = int(experiment.model_config.get("samples", 64))
    model = DenoisingDiffusion(
        timesteps=timesteps,
        channels=int(experiment.model_config.get("channels", 32)),
    ).to(experiment.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=experiment.learning_rate
    )

    for epoch in range(1, experiment.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, _ in experiment.dataloader:
            clean_images = images.to(experiment.device).mul(2).sub(1)
            optimizer.zero_grad()
            loss = model.training_loss(clean_images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f"Época {epoch:02d}/{experiment.epochs} | "
            f"Ruído MSE: {total_loss / len(experiment.dataloader):.4f}"
        )
        model.eval()
        save_samples(
            (model.sample(sample_count, experiment.device).clamp(-1, 1) + 1) / 2,
            experiment.output_dir,
            epoch,
        )

    save_checkpoint(
        experiment.output_dir,
        {
            "model": model.state_dict(),
            "timesteps": timesteps,
            "epochs": experiment.epochs,
            "seed": experiment.seed,
        },
    )


if __name__ == "__main__":
    main()
