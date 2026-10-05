from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image

from train_utils import save_checkpoint, start_experiment

# ============================================================
# Configuração
# ============================================================

# ============================================================
# Modelo
# ============================================================

class VAE(nn.Module):
    """Variational Autoencoder simples para imagens 28 x 28."""

    def __init__(self, latent_dim: int = 20) -> None:
        super().__init__()

        # Encoder: imagem achatada -> representação intermediária
        self.encoder = nn.Linear(28 * 28, 400)

        # Parâmetros da distribuição latente
        self.mu = nn.Linear(400, latent_dim)
        self.log_variance = nn.Linear(400, latent_dim)

        # Decoder: vetor latente -> imagem reconstruída
        self.decoder_hidden = nn.Linear(latent_dim, 400)
        self.decoder_output = nn.Linear(400, 28 * 28)

    def encode(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.relu(self.encoder(x))

        mu = self.mu(hidden)
        log_variance = self.log_variance(hidden)

        return mu, log_variance

    def reparameterize(
        self,
        mu: torch.Tensor,
        log_variance: torch.Tensor,
    ) -> torch.Tensor:
        standard_deviation = torch.exp(0.5 * log_variance)
        noise = torch.randn_like(standard_deviation)

        return mu + noise * standard_deviation

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.decoder_hidden(z))

        return torch.sigmoid(self.decoder_output(hidden))

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Transforma [batch, 1, 28, 28] em [batch, 784]
        x = x.flatten(start_dim=1)

        mu, log_variance = self.encode(x)
        z = self.reparameterize(mu, log_variance)
        reconstruction = self.decode(z)

        return reconstruction, mu, log_variance


# ============================================================
# Função de perda
# ============================================================

def vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    log_variance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = target.flatten(start_dim=1)

    reconstruction_loss = F.binary_cross_entropy(
        reconstruction,
        target,
        reduction="sum",
    )

    kl_divergence = -0.5 * torch.sum(
        1 + log_variance - mu.pow(2) - log_variance.exp()
    )

    total_loss = reconstruction_loss + kl_divergence

    return total_loss, reconstruction_loss, kl_divergence


# ============================================================
# Treinamento
# ============================================================

def train_one_epoch(
    model: VAE,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float, float]:
    model.train()

    total_loss = 0.0
    total_reconstruction = 0.0
    total_kl = 0.0

    for images, _ in dataloader:
        images = images.to(device)

        optimizer.zero_grad()

        reconstruction, mu, log_variance = model(images)

        loss, reconstruction_loss, kl_divergence = vae_loss(
            reconstruction,
            images,
            mu,
            log_variance,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_reconstruction += reconstruction_loss.item()
        total_kl += kl_divergence.item()

    dataset_size = len(dataloader.dataset)

    return (
        total_loss / dataset_size,
        total_reconstruction / dataset_size,
        total_kl / dataset_size,
    )


# ============================================================
# Geração
# ============================================================

@torch.no_grad()
def generate_images(
    model: VAE,
    device: torch.device,
    epoch: int,
    latent_dim: int,
    sample_count: int,
    output_dir: Path,
) -> None:
    model.eval()

    # Amostra pontos da distribuição normal latente
    z = torch.randn(sample_count, latent_dim, device=device)

    generated_images = model.decode(z)
    generated_images = generated_images.view(-1, 1, 28, 28)

    output_path = output_dir / f"samples_epoch_{epoch:02d}.png"

    save_image(
        generated_images,
        output_path,
        nrow=8,
    )


# ============================================================
# Execução
# ============================================================

def main() -> None:
    experiment = start_experiment("vae")
    latent_dim = int(experiment.model_config.get("latent_dim", 20))
    sample_count = int(experiment.model_config.get("samples", 64))
    model = VAE(latent_dim=latent_dim).to(experiment.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=experiment.learning_rate,
    )

    for epoch in range(1, experiment.epochs + 1):
        loss, reconstruction_loss, kl_divergence = train_one_epoch(
            model=model,
            dataloader=experiment.dataloader,
            optimizer=optimizer,
            device=experiment.device,
        )

        print(
            f"Época {epoch:02d}/{experiment.epochs} | "
            f"Loss: {loss:.4f} | "
            f"Reconstrução: {reconstruction_loss:.4f} | "
            f"KL: {kl_divergence:.4f}"
        )

        generate_images(
            model=model,
            device=experiment.device,
            epoch=epoch,
            latent_dim=latent_dim,
            sample_count=sample_count,
            output_dir=experiment.output_dir,
        )

    save_checkpoint(
        experiment.output_dir,
        {
            "model": model.state_dict(),
            "latent_dim": latent_dim,
            "seed": experiment.seed,
            "epochs": experiment.epochs,
        },
    )


if __name__ == "__main__":
    main()