"""Thin training orchestration for embodied policy runs."""

from __future__ import annotations

import asyncio
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

ProgressCallback = Callable[[str], Awaitable[None]]


@dataclass(frozen=True)
class TrainingConfig:
    dataset_path: str
    output_dir: str
    algorithm: str = "default"
    epochs: int = 100
    extra_args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingResult:
    ok: bool
    checkpoint_path: str | None
    epochs_completed: int
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def _command() -> list[str]:
    return ["roboclaw-train"] if shutil.which("roboclaw-train") else [sys.executable, "-m", "roboclaw.embodied.learning.train"]


def _epoch(line: str) -> int | None:
    match = re.search(r"epoch\D*(\d+)", line, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


async def run_training(config: TrainingConfig, on_progress: ProgressCallback | None = None) -> TrainingResult:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = _command() + [
        "--dataset",
        config.dataset_path,
        "--output",
        str(output_dir),
        "--epochs",
        str(config.epochs),
        "--algorithm",
        config.algorithm,
    ]
    for key, value in config.extra_args.items():
        flag = f"--{key.replace('_', '-')}"
        if value is True:
            command.append(flag)
        elif value is not False and value is not None:
            command.extend((flag, str(value)))
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError as exc:
        return TrainingResult(False, None, 0, "Training failed.", {"command": command, "error": str(exc)})
    epochs_completed = 0
    while process.stdout is not None:
        raw = await process.stdout.readline()
        if not raw:
            break
        epoch = _epoch(raw.decode(errors="replace").strip())
        if epoch is None:
            continue
        epochs_completed = epoch
        if on_progress is not None:
            await on_progress(f"Training: epoch {epoch}/{config.epochs}")
    return_code = await process.wait()
    checkpoint = next(
        (str(path) for pattern in ("*.pt", "*.ckpt", "*.safetensors") for path in sorted(output_dir.glob(pattern))),
        None,
    )
    return TrainingResult(
        ok=return_code == 0,
        checkpoint_path=checkpoint,
        epochs_completed=epochs_completed,
        message="Training complete." if return_code == 0 else "Training failed.",
        details={"command": command, "return_code": return_code},
    )
