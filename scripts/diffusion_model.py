import math

import torch
from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_width = self.width // 2
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(half_width, device=timesteps.device)
            / max(half_width - 1, 1)
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        return torch.cat((angles.sin(), angles.cos()), dim=1)


class TimeResidualBlock(nn.Module):
    def __init__(self, channels: int, time_width: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_projection = nn.Linear(time_width, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(
        self,
        features: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(features)))
        hidden = hidden + self.time_projection(time_embedding)[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return features + hidden


class NoisePredictor(nn.Module):
    def __init__(self, input_channels: int = 1, channels: int = 32) -> None:
        super().__init__()
        self.time_embedding = nn.Sequential(
            TimeEmbedding(channels),
            nn.Linear(channels, channels * 4),
            nn.SiLU(),
            nn.Linear(channels * 4, channels),
        )
        self.input = nn.Conv2d(input_channels, channels, 3, padding=1)
        self.block1 = TimeResidualBlock(channels, channels)
        self.block2 = TimeResidualBlock(channels, channels)
        self.output_norm = nn.GroupNorm(8, channels)
        self.output = nn.Conv2d(channels, input_channels, 3, padding=1)

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        time_embedding = self.time_embedding(timesteps)
        features = self.input(noisy_images)
        features = self.block1(features, time_embedding)
        features = self.block2(features, time_embedding)
        return self.output(F.silu(self.output_norm(features)))


class DenoisingDiffusion(nn.Module):
    def __init__(
        self,
        timesteps: int,
        input_channels: int = 1,
        channels: int = 32,
    ) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.denoiser = NoisePredictor(input_channels, channels)
        positions = torch.linspace(0, timesteps, timesteps + 1) / timesteps
        offset = 0.008
        alpha_bars = torch.cos(
            ((positions + offset) / (1 + offset)) * math.pi / 2
        ).square()
        alpha_bars = alpha_bars / alpha_bars[0]
        betas = (1 - alpha_bars[1:] / alpha_bars[:-1]).clamp(max=0.999)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        posterior_variance = betas * (1 - F.pad(alpha_bars[:-1], (1, 0), value=1)) / (
            1 - alpha_bars
        )
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(1e-20))

    def training_loss(self, clean_images: torch.Tensor) -> torch.Tensor:
        batch_size = len(clean_images)
        timestep_indices = torch.randint(
            self.timesteps, (batch_size,), device=clean_images.device
        )
        noise = torch.randn_like(clean_images)
        alpha_bar = self.alpha_bars[timestep_indices][:, None, None, None]
        noisy_images = (
            alpha_bar.sqrt() * clean_images
            + (1 - alpha_bar).sqrt() * noise
        )
        predicted_noise = self.denoiser(noisy_images, timestep_indices)
        return F.mse_loss(predicted_noise, noise)

    def denoise_step(
        self,
        images: torch.Tensor,
        index: int,
        predicted_noise: torch.Tensor,
        clip_denoised: bool = False,
    ) -> torch.Tensor:
        """Um passo da geração: remove parte do ruído de ``images`` no passo ``index``.

        Com ``clip_denoised``, a estimativa da imagem limpa (x0) é limitada a
        [-1, 1] antes do passo, como no DDPM original. Isso impede que erros da
        rede se acumulem ao longo dos passos e saturem a imagem. Só faz sentido
        quando os dados estão em [-1, 1] (não vale para latentes).
        """
        beta = self.betas[index]
        alpha = self.alphas[index]
        alpha_bar = self.alpha_bars[index]
        if not clip_denoised:
            images = (images - beta / torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha)
        else:
            clean = (images - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
            clean = clean.clamp(-1, 1)
            if index == 0:
                return clean
            # Média da distribuição do passo anterior, calculada a partir de x0.
            alpha_bar_prev = self.alpha_bars[index - 1]
            images = (
                beta * alpha_bar_prev.sqrt() * clean
                + (1 - alpha_bar_prev) * alpha.sqrt() * images
            ) / (1 - alpha_bar)
        if index > 0:
            images = images + self.posterior_variance[index].sqrt() * torch.randn_like(images)
        return images

    @torch.no_grad()
    def sample(
        self,
        sample_count: int,
        device: torch.device,
        spatial_size: int = 28,
        clip_denoised: bool = False,
    ) -> torch.Tensor:
        images = torch.randn(
            sample_count,
            self.denoiser.output.out_channels,
            spatial_size,
            spatial_size,
            device=device,
        )
        for index in reversed(range(self.timesteps)):
            timesteps = torch.full(
                (sample_count,), index, device=device, dtype=torch.long
            )
            predicted_noise = self.denoiser(images, timesteps)
            images = self.denoise_step(images, index, predicted_noise, clip_denoised)
        return images
