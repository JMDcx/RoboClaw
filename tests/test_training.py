from __future__ import annotations

import sys
from pathlib import Path

import pytest

from roboclaw.embodied.execution.orchestration import training


class _Stdout:
    def __init__(self, lines: list[str]) -> None:
        self._lines = [f"{line}\n".encode() for line in lines]

    async def readline(self) -> bytes:
        return self._lines.pop(0) if self._lines else b""


class _Process:
    def __init__(self, lines: list[str], return_code: int) -> None:
        self.stdout = _Stdout(lines)
        self._return_code = return_code

    async def wait(self) -> int:
        return self._return_code


def test_training_config_defaults() -> None:
    config = training.TrainingConfig(dataset_path="data.jsonl", output_dir="out")
    assert config.algorithm == "default"
    assert config.epochs == 100
    assert config.extra_args == {}


@pytest.mark.asyncio
async def test_run_training_reports_progress_and_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training.shutil, "which", lambda name: "/usr/bin/roboclaw-train")

    async def fake_create(*command, **kwargs):
        (tmp_path / "out" / "model.ckpt").parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "out" / "model.ckpt").write_text("x", encoding="utf-8")
        assert command[0] == "roboclaw-train"
        return _Process(["epoch 1/3", "Epoch 3 done"], 0)

    monkeypatch.setattr(training.asyncio, "create_subprocess_exec", fake_create)
    updates: list[str] = []

    async def on_progress(message: str) -> None:
        updates.append(message)

    result = await training.run_training(
        training.TrainingConfig(dataset_path="data.jsonl", output_dir=str(tmp_path / "out"), epochs=3),
        on_progress=on_progress,
    )

    assert result.ok is True
    assert result.checkpoint_path == str(tmp_path / "out" / "model.ckpt")
    assert result.epochs_completed == 3
    assert updates == ["Training: epoch 1/3", "Training: epoch 3/3"]


@pytest.mark.asyncio
async def test_run_training_returns_failure_on_nonzero_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training.shutil, "which", lambda name: None)

    async def fake_create(*command, **kwargs):
        assert command[:3] == (sys.executable, "-m", "roboclaw.embodied.learning.train")
        return _Process(["epoch 2/5"], 1)

    monkeypatch.setattr(training.asyncio, "create_subprocess_exec", fake_create)
    result = await training.run_training(
        training.TrainingConfig(dataset_path="data.jsonl", output_dir=str(tmp_path / "out"), epochs=5)
    )

    assert result.ok is False
    assert result.checkpoint_path is None
    assert result.epochs_completed == 2
