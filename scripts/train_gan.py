import torch
from torch import nn
from torch.nn import functional as F

from train_utils import save_checkpoint, save_samples, start_experiment


class Generator(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.project = nn.Linear(latent_dim, 128 * 7 * 7)
        self.network = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        features = self.project(noise).view(-1, 128, 7, 7)
        return self.network(features)


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.output = nn.Linear(128 * 3 * 3, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(images).flatten(start_dim=1)).squeeze(1)


def main() -> None:
    experiment = start_experiment("gan")
    latent_dim = int(experiment.model_config.get("latent_dim", 64))
    sample_count = int(experiment.model_config.get("samples", 64))
    generator = Generator(latent_dim).to(experiment.device)
    discriminator = Discriminator().to(experiment.device)
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=experiment.learning_rate, betas=(0.5, 0.999)
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=experiment.learning_rate, betas=(0.5, 0.999)
    )

    for epoch in range(1, experiment.epochs + 1):
        generator.train()
        discriminator.train()
        total_generator_loss = 0.0
        total_discriminator_loss = 0.0

        for real_images, _ in experiment.dataloader:
            real_images = real_images.to(experiment.device)
            batch_size = len(real_images)
            real_targets = torch.ones(batch_size, device=experiment.device)
            fake_targets = torch.zeros(batch_size, device=experiment.device)

            discriminator_optimizer.zero_grad()
            real_logits = discriminator(real_images)
            noise = torch.randn(batch_size, latent_dim, device=experiment.device)
            fake_images = generator(noise)
            fake_logits = discriminator(fake_images.detach())
            discriminator_loss = (
                F.binary_cross_entropy_with_logits(real_logits, real_targets)
                + F.binary_cross_entropy_with_logits(fake_logits, fake_targets)
            )
            discriminator_loss.backward()
            discriminator_optimizer.step()

            generator_optimizer.zero_grad()
            generated_logits = discriminator(fake_images)
            generator_loss = F.binary_cross_entropy_with_logits(
                generated_logits, real_targets
            )
            generator_loss.backward()
            generator_optimizer.step()

            total_discriminator_loss += discriminator_loss.item()
            total_generator_loss += generator_loss.item()

        batch_count = len(experiment.dataloader)
        print(
            f"Época {epoch:02d}/{experiment.epochs} | "
            f"D: {total_discriminator_loss / batch_count:.4f} | "
            f"G: {total_generator_loss / batch_count:.4f}"
        )
        generator.eval()
        with torch.no_grad():
            noise = torch.randn(sample_count, latent_dim, device=experiment.device)
            save_samples(generator(noise), experiment.output_dir, epoch)

    save_checkpoint(
        experiment.output_dir,
        {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "latent_dim": latent_dim,
            "epochs": experiment.epochs,
            "seed": experiment.seed,
        },
    )


if __name__ == "__main__":
    main()
