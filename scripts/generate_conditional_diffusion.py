"""Gera imagens das classes escolhidas a partir de um checkpoint condicional.

Exemplo (8 tênis e 8 botas, uma linha por classe), trocando o nome da pasta
pelo que o treino imprimiu no terminal:

    python scripts\\generate_conditional_diffusion.py outputs\\conditional_diffusion\\20261112T233335_375850Z_8d91b0ef\\checkpoint.pt --classes tenis bota
"""
from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path
from typing import Sequence

import torch
from torchvision.utils import save_image

from conditional_diffusion_model import ConditionalDenoisingDiffusion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "outputs" / "conditional_diffusion"


def missing_checkpoint_message(path: Path) -> str:
    message = (
        f"checkpoint não encontrado: {path}\n"
        "Use o caminho real da pasta do treino (sem < >), impresso no terminal ao fim do treino."
    )
    available = sorted(RUNS_DIR.glob("*/checkpoint.pt"))[-3:]
    if available:
        listed = "\n".join(f"  {item.relative_to(PROJECT_ROOT)}" for item in reversed(available))
        message += f"\nCheckpoints mais recentes:\n{listed}"
    return message


def normalize(name: str) -> str:
    """Ignora maiúsculas e acentos: "Tênis" e "tenis" são a mesma classe."""
    decomposed = unicodedata.normalize("NFKD", name.strip().lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def resolve_classes(requested: Sequence[str], class_names: Sequence[str]) -> list[int]:
    lookup = {normalize(name): index for index, name in enumerate(class_names)}
    indices = []
    for item in requested:
        key = normalize(item)
        if key.isdigit() and int(key) < len(class_names):
            indices.append(int(key))
        elif key in lookup:
            indices.append(lookup[key])
        else:
            options = ", ".join(f"{index}={name}" for index, name in enumerate(class_names))
            raise ValueError(f"Classe desconhecida: {item!r}. Opções: {options}")
    return indices


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description="Gera imagens de classes escolhidas com a difusão condicional."
    )
    parser.add_argument("checkpoint", type=Path, help="checkpoint.pt de train_conditional_diffusion.py")
    parser.add_argument(
        "--classes", nargs="+", required=True,
        help="Nomes ou índices das classes, por exemplo: tenis bota 5",
    )
    parser.add_argument("--per-class", type=int, default=8, help="Imagens por classe (padrão: 8)")
    parser.add_argument(
        "--guidance-scale", type=float,
        help="1 = sem guidance; maior = classe mais evidente (padrão: valor do treinamento)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--clip-denoised", action=argparse.BooleanOptionalAction, default=True,
        help="Limita a imagem limpa estimada a [-1, 1] em cada passo, evitando imagens saturadas",
    )
    parser.add_argument("--output", type=Path, help="PNG de saída (padrão: na pasta do checkpoint)")
    args = parser.parse_args(argv)
    if args.per_class <= 0:
        parser.error("--per-class deve ser positivo")
    if args.guidance_scale is not None and args.guidance_scale < 0:
        parser.error("--guidance-scale deve ser maior ou igual a 0")
    if not args.checkpoint.is_file():
        parser.error(missing_checkpoint_message(args.checkpoint))

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    class_names = checkpoint["class_names"]
    try:
        indices = resolve_classes(args.classes, class_names)
    except ValueError as error:
        parser.error(str(error))
    guidance_scale = (
        args.guidance_scale if args.guidance_scale is not None else checkpoint["guidance_scale"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalDenoisingDiffusion(**checkpoint["model_constructor"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    torch.manual_seed(args.seed)
    labels = torch.tensor(indices, device=device).repeat_interleave(args.per_class)
    images = model.sample(labels, guidance_scale, clip_denoised=args.clip_denoised)

    names = "_".join(class_names[index] for index in indices)
    suffix = "" if args.clip_denoised else "_semclip"
    output = args.output or (
        args.checkpoint.parent
        / f"generated_{names}_g{guidance_scale:g}_seed{args.seed}{suffix}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(((images.clamp(-1, 1) + 1) / 2).cpu(), output, nrow=args.per_class)
    print(
        f"Classes: {', '.join(class_names[index] for index in indices)} | "
        f"guidance: {guidance_scale:g} | limitação de x0: {'sim' if args.clip_denoised else 'não'}"
    )
    print(f"Imagens salvas: {output}")
    return output


if __name__ == "__main__":
    main()
