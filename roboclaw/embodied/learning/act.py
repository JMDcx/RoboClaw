"""ACT policy command builder for LeRobot 0.5.0."""

from __future__ import annotations

from pathlib import Path


class ACTPipeline:
    """Builds LeRobot training commands for ACT policy."""

    def train(
        self,
        repo_id: str,
        dataset_root: str,
        output_dir: str,
        steps: int = 100000,
        device: str = "cuda",
    ) -> list[str]:
        """Build the ACT training command (LeRobot 0.5.0 format)."""
        return [
            "lerobot-train",
            f"--dataset.repo_id={repo_id}",
            f"--dataset.root={Path(dataset_root).expanduser()}",
            "--dataset.video_backend=pyav",
            "--policy.type=act",
            "--policy.push_to_hub=false",
            f"--policy.repo_id={repo_id}",
            f"--output_dir={Path(output_dir).expanduser()}",
            f"--steps={steps}",
            f"--policy.device={device}",
        ]

    def checkpoint_path(self, output_dir: str) -> str:
        """Return the last checkpoint path."""
        return str(Path(output_dir).expanduser() / "checkpoints" / "last" / "pretrained_model")
