import math

import torch
from torch import nn

from train_utils import save_checkpoint, save_samples, start_experiment


class AffineCoupling(nn.Module):
    def __init__(self, dimensions: int, hidden_dim: int, parity: int) -> None:
        super().__init__()
        mask = (torch.arange(dimensions) % 2 == parity).float()
        self.register_buffer("mask", mask)
        self.network = nn.Sequential(
            nn.Linear(dimensions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dimensions * 2),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        values: torch.Tensor,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fixed_values = values * self.mask
        shift, raw_scale = self.network(fixed_values).chunk(2, dim=1)
        scale = 0.8 * torch.tanh(raw_scale) * (1 - self.mask)
        shift = shift * (1 - self.mask)

        if reverse:
            transformed = fixed_values + (1 - self.mask) * (
                values - shift
            ) * torch.exp(-scale)
            log_determinant = -scale.sum(dim=1)
        else:
            transformed = fixed_values + (1 - self.mask) * (
                values * torch.exp(scale) + shift
            )
            log_determinant = scale.sum(dim=1)
        return transformed, log_determinant


class RealNVP(nn.Module):
    def __init__(
        self,
        dimensions: int,
        hidden_dim: int,
        coupling_layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            AffineCoupling(dimensions, hidden_dim, index % 2)
            for index in range(coupling_layers)
        )
        self.dimensions = dimensions

    def log_probability(self, values: torch.Tensor) -> torch.Tensor:
        log_determinant = torch.zeros(len(values), device=values.device)
        transformed = values
        for layer in self.layers:
            transformed, layer_log_determinant = layer(transformed)
            log_determinant += layer_log_determinant
        base_log_probability = -0.5 * (
            transformed.square() + math.log(2 * math.pi)
        ).sum(dim=1)
        return base_log_probability + log_determinant

    @torch.no_grad()
    def sample(self, sample_count: int, device: torch.device) -> torch.Tensor:
        values = torch.randn(sample_count, self.dimensions, device=device)
        for layer in reversed(self.layers):
            values, _ = layer(values, reverse=True)
        return values.view(-1, 1, 28, 28).clamp(0, 1)


def main() -> None:
    experiment = start_experiment("normalizing_flow")
    model = RealNVP(
        dimensions=28 * 28,
        hidden_dim=int(experiment.model_config.get("hidden_dim", 512)),
        coupling_layers=int(
            experiment.model_config.get("coupling_layers", 6)
        ),
    ).to(experiment.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=experiment.learning_rate
    )
    sample_count = int(experiment.model_config.get("samples", 64))

    for epoch in range(1, experiment.epochs + 1):
        model.train()
        total_nll = 0.0
        for images, _ in experiment.dataloader:
            images = images.to(experiment.device).flatten(start_dim=1)
            dequantized = (images * 255 + torch.rand_like(images)) / 256
            optimizer.zero_grad()
            log_probability = model.log_probability(dequantized)
            loss = -log_probability.mean() / model.dimensions
            loss.backward()
            optimizer.step()
            total_nll += loss.item()

        print(
            f"Época {epoch:02d}/{experiment.epochs} | "
            f"NLL/dimensão: {total_nll / len(experiment.dataloader):.4f}"
        )
        save_samples(
            model.sample(sample_count, experiment.device),
            experiment.output_dir,
            epoch,
        )

    save_checkpoint(
        experiment.output_dir,
        {
            "model": model.state_dict(),
            "dimensions": model.dimensions,
            "epochs": experiment.epochs,
            "seed": experiment.seed,
        },
    )


if __name__ == "__main__":
    main()
