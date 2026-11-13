"""Conditional diffusion: model behaviour and small CPU training/generation runs."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_conditional_diffusion
import train_conditional_diffusion
import train_utils
from conditional_diffusion_model import ConditionalDenoisingDiffusion
from diffusion_model import DenoisingDiffusion
from test_experiment_tracking import TinyFashionDataset


class SamplerTests(unittest.TestCase):
    def test_unclipped_step_keeps_original_formula(self):
        torch.manual_seed(0)
        model = DenoisingDiffusion(timesteps=4, channels=8)
        images, noise = torch.randn(2, 1, 28, 28), torch.randn(2, 1, 28, 28)
        for index in (0, 3):
            with self.subTest(index=index):
                beta, alpha = model.betas[index], model.alphas[index]
                alpha_bar = model.alpha_bars[index]
                expected = (images - beta / torch.sqrt(1 - alpha_bar) * noise) / torch.sqrt(alpha)
                torch.manual_seed(1)
                if index > 0:
                    expected = expected + model.posterior_variance[index].sqrt() * torch.randn_like(images)
                torch.manual_seed(1)
                self.assertTrue(torch.equal(model.denoise_step(images, index, noise), expected))

    def test_clipped_sampling_stays_in_image_range(self):
        torch.manual_seed(0)
        unconditional = DenoisingDiffusion(timesteps=4, channels=8).eval()
        conditional = ConditionalDenoisingDiffusion(timesteps=4, num_classes=3, channels=8).eval()
        for name, images in (
            ("unconditional", unconditional.sample(3, torch.device("cpu"), clip_denoised=True)),
            ("conditional", conditional.sample(
                torch.tensor([0, 1, 2]), guidance_scale=5.0, clip_denoised=True
            )),
        ):
            with self.subTest(model=name):
                self.assertLessEqual(images.abs().max().item(), 1.0)


class ConditionalModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = ConditionalDenoisingDiffusion(timesteps=4, num_classes=3, channels=8).eval()
        self.images = torch.randn(2, 1, 28, 28)
        self.timesteps = torch.tensor([0, 3])
        self.labels = torch.tensor([0, 2])

    def test_class_changes_noise_prediction(self):
        with torch.no_grad():
            first = self.model.denoiser(self.images, self.timesteps, self.labels)
            second = self.model.denoiser(self.images, self.timesteps, self.labels.flip(0))
        self.assertFalse(torch.allclose(first, second))

    def test_guidance_scale_combines_conditional_and_unconditional(self):
        null_labels = torch.full_like(self.labels, self.model.null_label)
        with torch.no_grad():
            conditional = self.model.denoiser(self.images, self.timesteps, self.labels)
            unconditional = self.model.denoiser(self.images, self.timesteps, null_labels)
            for scale, expected in (
                (0.0, unconditional),
                (1.0, conditional),
                (3.0, unconditional + 3 * (conditional - unconditional)),
            ):
                with self.subTest(scale=scale):
                    torch.testing.assert_close(
                        self.model.guided_noise(self.images, self.timesteps, self.labels, scale),
                        expected,
                    )

    def test_label_dropout_hides_classes_only_when_enabled(self):
        labels = torch.tensor([0, 1, 2, 0, 1])
        for dropout, expected in ((1.0, torch.full((5,), 3)), (0.0, labels)):
            with self.subTest(dropout=dropout):
                model = ConditionalDenoisingDiffusion(4, 3, channels=8, label_dropout=dropout)
                seen = []
                forward = model.denoiser.forward

                def capture(images, timesteps, received):
                    seen.append(received.clone())
                    return forward(images, timesteps, received)

                with patch.object(model.denoiser, "forward", capture):
                    model.training_loss(torch.zeros(5, 1, 28, 28), labels)
                self.assertTrue(torch.equal(seen[0], expected))

    def test_sample_rejects_unknown_class(self):
        with self.assertRaisesRegex(ValueError, "classe"):
            self.model.sample(torch.tensor([3]))

    def test_class_names_ignore_case_and_accents(self):
        names = train_conditional_diffusion.CLASS_NAMES
        self.assertEqual(
            generate_conditional_diffusion.resolve_classes(["Tênis", "9", "sandália"], names),
            [7, 9, 5],
        )
        with self.assertRaisesRegex(ValueError, "sapato"):
            generate_conditional_diffusion.resolve_classes(["sapato"], names)


class ConditionalTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        cls.deterministic = torch.are_deterministic_algorithms_enabled()
        cls.cudnn_benchmark = torch.backends.cudnn.benchmark
        cls.cudnn_deterministic = torch.backends.cudnn.deterministic
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        torch.use_deterministic_algorithms(cls.deterministic)
        torch.backends.cudnn.benchmark = cls.cudnn_benchmark
        torch.backends.cudnn.deterministic = cls.cudnn_deterministic

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = self.root / "baseline.yml"
        self.write_config(label_dropout=0.2)
        for target, value in (
            ("CONFIG_PATH", self.config),
            ("OUTPUT_ROOT", self.root / "outputs"),
        ):
            patcher = patch.object(train_utils, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for patcher in (
            patch.object(train_utils.datasets, "FashionMNIST", TinyFashionDataset),
            patch.object(torch.cuda, "is_available", return_value=False),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_config(self, label_dropout):
        self.config.write_text(yaml.safe_dump({
            "experiment": {"name": "test", "seed": 42},
            "data": {"dataset": "fashion_mnist", "samples": 7},
            "training": {"epochs": 3, "batch_size": 4, "learning_rate": 0.0002},
            "models": {"conditional_diffusion": {
                "timesteps": 4, "channels": 8, "samples": 3,
                "label_dropout": label_dropout, "guidance_scale": 2.0, "clip_denoised": True,
            }},
        }), encoding="utf-8")

    def run_training(self, epochs=2):
        arguments = [
            "train_conditional_diffusion.py", "--epochs", str(epochs), "--samples", "5",
            "--batch-size", "3", "--generated-samples", "1", "--seed", "7", "--deterministic",
        ]
        with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()):
            train_conditional_diffusion.main()
        return sorted((self.root / "outputs" / "conditional_diffusion").iterdir())[-1]

    def generate(self, checkpoint, *arguments):
        with contextlib.redirect_stdout(io.StringIO()):
            return generate_conditional_diffusion.main([str(checkpoint), *arguments])

    def test_records_conditioning_and_reproduces_runs(self):
        first, second = self.run_training(), self.run_training()
        metadata = json.loads((first / "metadata.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(first / "checkpoint.pt", weights_only=True)
        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(metadata["model"]["class"], "ConditionalDenoisingDiffusion")
        self.assertEqual(metadata["model"]["constructor"]["num_classes"], 10)
        self.assertEqual(metadata["conditioning"]["label_dropout"], 0.2)
        self.assertEqual(metadata["sampling"]["samples_per_epoch"], 10)
        self.assertIs(metadata["sampling"]["clip_denoised"], True)
        self.assertEqual(
            list(metadata["dataset"]["class_counts"].values()), [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        )
        self.assertEqual(checkpoint["class_names"][7], "tenis")
        self.assertEqual(checkpoint["guidance_scale"], 2.0)
        self.assertIs(checkpoint["clip_denoised"], True)
        # --generated-samples 1: one 28x28 image per class, one row per class.
        with Image.open(first / "samples_epoch_02.png") as image:
            self.assertEqual(image.size, (28 + 2 * 2, 10 * 28 + 11 * 2))
        a = torch.load(second / "checkpoint.pt", weights_only=True)["model"]
        self.assertTrue(all(torch.equal(checkpoint["model"][key], a[key]) for key in a))
        self.assertEqual(
            (first / "samples_epoch_02.png").read_bytes(),
            (second / "samples_epoch_02.png").read_bytes(),
        )

    def test_generates_requested_classes_from_checkpoint(self):
        checkpoint = self.run_training(epochs=1) / "checkpoint.pt"
        arguments = ["--classes", "tênis", "9", "--per-class", "3", "--seed", "1"]
        first = self.generate(checkpoint, *arguments)
        second = self.generate(checkpoint, *arguments, "--output", str(self.root / "again.png"))
        self.assertEqual(first.name, "generated_tenis_bota_g2_seed1.png")
        with Image.open(first) as image:
            self.assertEqual(image.size, (3 * 28 + 4 * 2, 2 * 28 + 3 * 2))
        self.assertEqual(first.read_bytes(), second.read_bytes())
        unclipped = self.generate(checkpoint, *arguments, "--no-clip-denoised")
        self.assertEqual(unclipped.name, "generated_tenis_bota_g2_seed1_semclip.png")
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.generate(checkpoint, "--classes", "sapato")

    def test_missing_checkpoint_lists_available_ones(self):
        checkpoint = self.run_training(epochs=1) / "checkpoint.pt"
        errors = io.StringIO()
        runs = self.root / "outputs" / "conditional_diffusion"
        with patch.object(generate_conditional_diffusion, "RUNS_DIR", runs), \
                self.assertRaises(SystemExit), contextlib.redirect_stderr(errors):
            self.generate(runs / "<pasta>" / "checkpoint.pt", "--classes", "tenis")
        self.assertIn("checkpoint não encontrado", errors.getvalue())
        self.assertIn(str(checkpoint.relative_to(ROOT)), errors.getvalue())

    def test_guidance_without_label_dropout_is_rejected(self):
        self.write_config(label_dropout=0.0)
        with self.assertRaisesRegex(ValueError, "label_dropout"):
            self.run_training()
        path = next((self.root / "outputs" / "conditional_diffusion").iterdir())
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "failed")


if __name__ == "__main__":
    unittest.main()
