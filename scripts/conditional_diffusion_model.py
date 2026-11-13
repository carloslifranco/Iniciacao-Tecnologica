"""DDPM condicionado por classe, com classifier-free guidance.

Reaproveita o cronograma de ruído e a rede do DDPM incondicional
(``diffusion_model.py``). A única mudança na rede é que o preditor de ruído
também recebe a classe da imagem; assim, na geração, é possível escolher o que
será gerado (por exemplo, "tênis" no Fashion-MNIST ou, futuramente, um achado
radiológico).
"""
import torch
from torch import nn
from torch.nn import functional as F

from diffusion_model import DenoisingDiffusion, NoisePredictor


class ConditionalNoisePredictor(NoisePredictor):
    """Preditor de ruído que recebe a imagem ruidosa, o passo de tempo e a classe.

    O embedding da classe é somado ao embedding de tempo, então cada bloco
    residual recebe as duas informações pela mesma projeção. O índice extra
    ``num_classes`` significa "sem classe" e é usado no classifier-free guidance.
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 1,
        channels: int = 32,
    ) -> None:
        super().__init__(input_channels, channels)
        self.class_embedding = nn.Embedding(num_classes + 1, channels)

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.time_embedding(timesteps) + self.class_embedding(labels)
        features = self.input(noisy_images)
        features = self.block1(features, condition)
        features = self.block2(features, condition)
        return self.output(F.silu(self.output_norm(features)))


class ConditionalDenoisingDiffusion(DenoisingDiffusion):
    def __init__(
        self,
        timesteps: int,
        num_classes: int,
        input_channels: int = 1,
        channels: int = 32,
        label_dropout: float = 0.1,
    ) -> None:
        if num_classes < 1:
            raise ValueError("num_classes deve ser positivo")
        if not 0 <= label_dropout <= 1:
            raise ValueError("label_dropout deve estar entre 0 e 1")
        # A classe base cria o cronograma de ruído (betas, alphas...) e um
        # preditor incondicional, que é substituído logo abaixo.
        super().__init__(timesteps, input_channels, channels)
        self.num_classes = num_classes
        self.null_label = num_classes
        self.label_dropout = label_dropout
        self.denoiser = ConditionalNoisePredictor(num_classes, input_channels, channels)

    def training_loss(
        self,
        clean_images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.label_dropout > 0:
            # Classifier-free guidance: em parte dos exemplos a classe é
            # escondida, e a mesma rede aprende também a versão incondicional.
            hidden = torch.rand(len(labels), device=labels.device) < self.label_dropout
            labels = labels.masked_fill(hidden, self.null_label)
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
        predicted_noise = self.denoiser(noisy_images, timestep_indices, labels)
        return F.mse_loss(predicted_noise, noise)

    def guided_noise(
        self,
        images: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Combina as previsões de ruído com e sem classe.

        ``guidance_scale`` 0 gera sem considerar a classe; 1 usa só a previsão
        condicional (sem guidance); valores maiores reforçam a classe pedida,
        em troca de menos diversidade.
        """
        if guidance_scale == 1:
            return self.denoiser(images, timesteps, labels)
        null_labels = torch.full_like(labels, self.null_label)
        # As duas previsões são calculadas no mesmo lote; o GroupNorm atua
        # por imagem, então uma metade não interfere na outra.
        conditional, unconditional = self.denoiser(
            torch.cat((images, images)),
            torch.cat((timesteps, timesteps)),
            torch.cat((labels, null_labels)),
        ).chunk(2)
        return unconditional + guidance_scale * (conditional - unconditional)

    @torch.no_grad()
    def sample(
        self,
        labels: torch.Tensor,
        guidance_scale: float = 1.0,
        spatial_size: int = 28,
        clip_denoised: bool = False,
    ) -> torch.Tensor:
        """Gera uma imagem para cada classe em ``labels``, no dispositivo dos rótulos.

        ``clip_denoised`` limita a estimativa da imagem limpa a [-1, 1] em cada
        passo (ver ``DenoisingDiffusion.denoise_step``). É especialmente útil com
        guidance, que amplifica os desvios e tende a saturar a imagem.
        """
        if labels.ndim != 1 or len(labels) == 0:
            raise ValueError("labels deve ser um vetor não vazio de classes")
        if labels.min().item() < 0 or labels.max().item() >= self.num_classes:
            raise ValueError(
                f"Toda classe deve estar entre 0 e {self.num_classes - 1}"
            )
        device = labels.device
        images = torch.randn(
            len(labels),
            self.denoiser.output.out_channels,
            spatial_size,
            spatial_size,
            device=device,
        )
        for index in reversed(range(self.timesteps)):
            timesteps = torch.full(
                (len(labels),), index, device=device, dtype=torch.long
            )
            predicted_noise = self.guided_noise(images, timesteps, labels, guidance_scale)
            images = self.denoise_step(images, index, predicted_noise, clip_denoised)
        return images
