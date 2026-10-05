import torch
from torch import nn
from torch.nn import functional as F

from train_utils import save_checkpoint, save_samples, start_experiment


class MaskedConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        mask_type: str,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
        )
        if mask_type not in {"A", "B"}:
            raise ValueError("mask_type deve ser A ou B")
        mask = torch.ones_like(self.weight)
        center_y, center_x = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        mask[:, :, center_y + 1 :, :] = 0
        mask[:, :, center_y, center_x + (mask_type == "A") :] = 0
        self.register_buffer("mask", mask)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            images,
            self.weight * self.mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class PixelCNN(nn.Module):
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            MaskedConv2d(
                1, hidden_channels, 7, padding=3, mask_type="A"
            ),
            nn.ReLU(),
            MaskedConv2d(
                hidden_channels,
                hidden_channels,
                7,
                padding=3,
                mask_type="B",
            ),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 256, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


@torch.no_grad()
def sample_images(
    model: PixelCNN,
    sample_count: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    images = torch.zeros(sample_count, 1, 28, 28, device=device)
    for row in range(28):
        for column in range(28):
            logits = model(images)[:, :, row, column]
            pixel_values = torch.multinomial(
                torch.softmax(logits, dim=1),
                num_samples=1,
            )
            images[:, 0, row, column] = pixel_values[:, 0].float() / 255.0
    return images


def main() -> None:
    experiment = start_experiment("pixelcnn")
    model = PixelCNN(
        int(experiment.model_config.get("hidden_channels", 64))
    ).to(experiment.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=experiment.learning_rate
    )
    sample_count = int(experiment.model_config.get("samples", 8))

    for epoch in range(1, experiment.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, _ in experiment.dataloader:
            images = images.to(experiment.device)
            targets = (images[:, 0] * 255).round().long()
            optimizer.zero_grad()
            loss = F.cross_entropy(model(images), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f"Época {epoch:02d}/{experiment.epochs} | "
            f"Cross-entropy: {total_loss / len(experiment.dataloader):.4f}"
        )
        save_samples(
            sample_images(model, sample_count, experiment.device),
            experiment.output_dir,
            epoch,
        )

    save_checkpoint(
        experiment.output_dir,
        {
            "model": model.state_dict(),
            "epochs": experiment.epochs,
            "seed": experiment.seed,
        },
    )


if __name__ == "__main__":
    main()
