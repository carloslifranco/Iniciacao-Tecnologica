"""Small CPU integration runs: no network or medical data required."""
import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_diffusion
import train_utils


class TinyFashionDataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        self.data = torch.randint(
            0, 256, (7, 28, 28), dtype=torch.uint8,
            generator=torch.Generator().manual_seed(100),
        )
        self.targets = torch.arange(7)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index].unsqueeze(0).float() / 255, self.targets[index]


class ExperimentTrackingTests(unittest.TestCase):
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
        self.config.write_text(yaml.safe_dump({
            "experiment": {"name": "test", "seed": 42},
            "data": {"dataset": "fashion_mnist", "samples": 7},
            "training": {"epochs": 3, "batch_size": 4, "learning_rate": 0.0002},
            "models": {"diffusion": {"timesteps": 4, "channels": 8, "samples": 3}},
        }), encoding="utf-8")
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

    def run_training(self, generated=1, epochs=2):
        arguments = [
            "train_diffusion.py", "--epochs", str(epochs), "--samples", "5",
            "--batch-size", "3", "--generated-samples", str(generated),
            "--seed", "7", "--deterministic",
        ]
        with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()):
            train_diffusion.main()
        return sorted((self.root / "outputs" / "diffusion").iterdir())[-1]

    @staticmethod
    def metrics(path):
        with (path / "metrics.csv").open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_records_effective_config_architecture_and_checkpoint(self):
        path = self.run_training()
        effective = yaml.safe_load((path / "config_effective.yml").read_text(encoding="utf-8"))
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(effective["training"]["epochs"], 2)
        self.assertEqual(effective["training"]["batch_size"], 3)
        self.assertEqual(effective["data"]["samples"], 5)
        self.assertEqual(effective["experiment"]["seed"], 7)
        self.assertEqual(effective["models"]["diffusion"]["samples"], 1)
        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(metadata["completed_epochs"], 2)
        self.assertEqual(metadata["dataset"]["available_samples"], 7)
        self.assertEqual(metadata["model"]["constructor"]["channels"], 8)
        self.assertEqual(metadata["model"]["constructor"]["timesteps"], 4)
        self.assertGreater(metadata["elapsed_seconds"], 0)
        self.assertTrue((path / "source/scripts/diffusion_model.py").is_file())
        self.assertTrue((path / "architecture.txt").is_file())
        self.assertEqual((path / "config_original.yml").read_bytes(), self.config.read_bytes())
        self.assertEqual(len(self.metrics(path)), 2)
        checkpoint = torch.load(path / "checkpoint.pt", weights_only=True)
        self.assertEqual(checkpoint["epochs"], 2)
        self.assertEqual(checkpoint["timesteps"], 4)
        self.assertEqual(len(checkpoint["model"]["betas"]), 4)
        self.assertIn("optimizer", checkpoint)
        self.assertTrue((path / "samples_epoch_02.png").is_file())

    def test_repeated_runs_reproduce_weights_and_samples_without_overwriting(self):
        first, second = self.run_training(), self.run_training()
        self.assertNotEqual(first, second)
        self.assertEqual(
            [row["train_noise_mse"] for row in self.metrics(first)],
            [row["train_noise_mse"] for row in self.metrics(second)],
        )
        a = torch.load(first / "checkpoint.pt", weights_only=True)["model"]
        b = torch.load(second / "checkpoint.pt", weights_only=True)["model"]
        self.assertTrue(all(torch.equal(a[key], b[key]) for key in a))
        self.assertEqual(
            (first / "samples_epoch_02.png").read_bytes(),
            (second / "samples_epoch_02.png").read_bytes(),
        )

    def test_sampling_count_does_not_change_training(self):
        first, second = self.run_training(generated=1), self.run_training(generated=2)
        self.assertEqual(
            [row["train_noise_mse"] for row in self.metrics(first)],
            [row["train_noise_mse"] for row in self.metrics(second)],
        )

    def test_loss_is_weighted_by_actual_batch_size(self):
        values = iter((1.0, 3.0))

        def loss(model, images):
            return next(model.parameters()).sum() * 0 + next(values)

        with patch.object(train_diffusion.DenoisingDiffusion, "training_loss", loss):
            path = self.run_training(epochs=1)
        self.assertAlmostEqual(float(self.metrics(path)[0]["train_noise_mse"]), 1.8)

    def test_failure_is_recorded_and_raised(self):
        with patch.object(
            train_diffusion.DenoisingDiffusion, "training_loss", side_effect=RuntimeError("test failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                self.run_training()
        path = next((self.root / "outputs" / "diffusion").iterdir())
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "failed")
        self.assertEqual(metadata["error"]["type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
