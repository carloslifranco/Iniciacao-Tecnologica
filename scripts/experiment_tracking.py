"""Local experiment records; no external tracking service is required."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

import torch
import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunRecorder:
    def __init__(
        self,
        output_root: Path,
        model_name: str,
        project_root: Path,
        config_path: Path,
        effective_config: dict[str, Any],
        dataset_metadata: dict[str, Any],
        device: torch.device,
    ) -> None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.output_dir = output_root / model_name / f"{run_id}_{uuid4().hex[:8]}"
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.started = time.perf_counter()
        self.metadata: dict[str, Any] = {
            "status": "prepared",
            "created_at_utc": utc_now(),
            "model_name": model_name,
            "command": [sys.executable, *sys.argv],
            "dataset": dataset_metadata,
            "environment": self._environment(device),
            "reproducibility": {
                "seed": effective_config["experiment"]["seed"],
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "note": "Seeds do not guarantee identical results across hardware or library versions.",
            },
        }
        shutil.copy2(config_path, self.output_dir / "config_original.yml")
        (self.output_dir / "config_effective.yml").write_text(
            yaml.safe_dump(effective_config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        # Preserve the actual local implementation, including uncommitted changes.
        source_files = list((project_root / "scripts").glob("*.py"))
        source_files += list((project_root / "src").rglob("*.py"))
        source_files += [project_root / "pyproject.toml"]
        hashes = {}
        for source in source_files:
            if not source.is_file():
                continue
            relative = source.relative_to(project_root)
            target = self.output_dir / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
        self.metadata["source_sha256"] = hashes
        self.save_metadata()

    @staticmethod
    def _environment(device: torch.device) -> dict[str, Any]:
        packages = {}
        for package in ("torch", "torchvision", "numpy", "PyYAML", "Pillow"):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = "not installed"
        return {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "torch_num_threads": torch.get_num_threads(),
        }

    def save_metadata(self) -> None:
        target = self.output_dir / "metadata.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(target)

    def record_model(
        self,
        model: torch.nn.Module,
        constructor: dict[str, Any],
        optimizer: torch.optim.Optimizer,
    ) -> None:
        (self.output_dir / "architecture.txt").write_text(str(model), encoding="utf-8")
        self.metadata["model"] = {
            "class": type(model).__name__,
            "constructor": constructor,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        self.metadata["optimizer"] = {
            "class": type(optimizer).__name__, "defaults": optimizer.defaults
        }
        self.save_metadata()

    def log_epoch(self, metrics: dict[str, Any]) -> None:
        path = self.output_dir / "metrics.csv"
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics))
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)
        self.metadata["completed_epochs"] = metrics["epoch"]
        self.save_metadata()

    @contextmanager
    def track(self) -> Iterator[None]:
        self.metadata.update(status="running", training_started_at_utc=utc_now())
        self.save_metadata()
        try:
            yield
        except BaseException as error:
            self.metadata.update(
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error={"type": type(error).__name__, "message": str(error)},
            )
            raise
        else:
            self.metadata["status"] = "completed"
        finally:
            self.metadata.update(
                finished_at_utc=utc_now(),
                elapsed_seconds=time.perf_counter() - self.started,
            )
            self.save_metadata()
