import torch
from torch import nn
from torch.nn import functional as F

from diffusion_model import DenoisingDiffusion
from train_utils import (
    Experiment,
    save_checkpoint,
    save_samples,
    start_experiment,
)


class LatentAutoencoder(nn.Module):
    def __init__(self, latent_channels: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.SiLU(),
        )
        self.mean = nn.Conv2d(64, latent_channels, 3, padding=1)
        self.log_variance = nn.Conv2d(64, latent_channels, 3, padding=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 64, 3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(images)
        return self.mean(features), self.log_variance(features).clamp(-20, 10)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_variance = self.encode(images)
        noise = torch.randn_like(mean)
        latents = mean + noise * torch.exp(0.5 * log_variance)
        return self.decode(latents), mean, log_variance


def train_autoencoder(
    autoencoder: LatentAutoencoder,
    experiment: Experiment,
    epochs: int,
    learning_rate: float,
) -> None:
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    for epoch in range(1, epochs + 1):
        autoencoder.train()
        total_loss = 0.0
        for images, _ in experiment.dataloader:
            images = images.to(experiment.device)
            optimizer.zero_grad()
            reconstruction, mean, log_variance = autoencoder(images)
            reconstruction_loss = F.binary_cross_entropy(
                reconstruction, images, reduction="sum"
            )
            kl_divergence = -0.5 * torch.sum(
                1 + log_variance - mean.square() - log_variance.exp()
            )
            loss = (reconstruction_loss + kl_divergence) / len(images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(
            f"Autoencoder {epoch:02d}/{epochs} | "
            f"Loss/imagem: {total_loss / len(experiment.dataloader):.4f}"
        )


@torch.no_grad()
def latent_scale(
    autoencoder: LatentAutoencoder,
    experiment: Experiment,
) -> torch.Tensor:
    autoencoder.eval()
    sum_of_squares = 0.0
    element_count = 0
    for images, _ in experiment.dataloader:
        mean, _ = autoencoder.encode(images.to(experiment.device))
        sum_of_squares += mean.square().sum().item()
        element_count += mean.numel()
    return torch.tensor(
        (sum_of_squares / element_count) ** 0.5,
        device=experiment.device,
    ).clamp_min(1e-6)


def main() -> None:
    experiment = start_experiment("latent_diffusion")
    config = experiment.model_config
    latent_channels = int(config.get("latent_channels", 4))
    autoencoder_epochs = int(config.get("autoencoder_epochs", experiment.epochs))
    autoencoder = LatentAutoencoder(latent_channels).to(experiment.device)

    train_autoencoder(
        autoencoder,
        experiment,
        autoencoder_epochs,
        float(config.get("autoencoder_learning_rate", 1e-3)),
    )
    autoencoder.eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)

    scale = latent_scale(autoencoder, experiment)
    diffusion = DenoisingDiffusion(
        timesteps=int(config.get("timesteps", 100)),
        input_channels=latent_channels,
        channels=int(config.get("channels", 32)),
    ).to(experiment.device)
    optimizer = torch.optim.AdamW(
        diffusion.parameters(), lr=experiment.learning_rate
    )
    sample_count = int(config.get("samples", 64))

    for epoch in range(1, experiment.epochs + 1):
        diffusion.train()
        total_loss = 0.0
        for images, _ in experiment.dataloader:
            with torch.no_grad():
                mean, _ = autoencoder.encode(images.to(experiment.device))
                clean_latents = mean / scale
            optimizer.zero_grad()
            loss = diffusion.training_loss(clean_latents)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(
            f"Difusão latente {epoch:02d}/{experiment.epochs} | "
            f"Ruído MSE: {total_loss / len(experiment.dataloader):.4f}"
        )
        diffusion.eval()
        with torch.no_grad():
            generated_latents = diffusion.sample(
                sample_count,
                experiment.device,
                spatial_size=7,
            )
            generated_images = autoencoder.decode(generated_latents * scale)
        save_samples(generated_images, experiment.output_dir, epoch)

    save_checkpoint(
        experiment.output_dir,
        {
            "autoencoder": autoencoder.state_dict(),
            "diffusion": diffusion.state_dict(),
            "latent_scale": scale.item(),
            "latent_channels": latent_channels,
            "epochs": experiment.epochs,
            "autoencoder_epochs": autoencoder_epochs,
            "seed": experiment.seed,
        },
    )


if __name__ == "__main__":
    main()
