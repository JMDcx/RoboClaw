"""Minimal embodied data collection utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from roboclaw.embodied.execution.orchestration.skills import SkillSpec, execute_skill

if TYPE_CHECKING:
    from roboclaw.embodied.execution.orchestration.runtime.executor import ExecutionContext, ProcedureExecutor

ProgressCallback = Callable[[str], Awaitable[None]]


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: int
    skill_name: str
    steps: tuple[dict[str, Any], ...]
    ok: bool


@dataclass(frozen=True)
class CollectionResult:
    ok: bool
    dataset_path: str
    episodes_requested: int
    episodes_completed: int
    episodes_failed: int
    message: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def collect_episodes(
    executor: ProcedureExecutor,
    context: ExecutionContext,
    skill: SkillSpec,
    num_episodes: int,
    output_dir: Path,
    on_progress: ProgressCallback | None = None,
) -> CollectionResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "dataset.jsonl"
    completed = 0
    failed = 0

    for episode_id in range(1, num_episodes + 1):
        steps: list[dict[str, Any]] = []
        reset_result = await executor.execute_reset(context)
        if reset_result.ok:
            class RecordingExecutor:
                async def execute_move(
                    self,
                    step_context: Any,
                    *,
                    primitive_name: str,
                    primitive_args: dict[str, Any] | None = None,
                    on_progress: ProgressCallback | None = None,
                ) -> Any:
                    result = await executor.execute_move(
                        step_context,
                        primitive_name=primitive_name,
                        primitive_args=primitive_args,
                        on_progress=on_progress,
                    )
                    steps.append(
                        {
                            "timestamp": _now(),
                            "state_before": dict(result.details.get("state_before") or {}),
                            "state_after": dict(result.details.get("state_after") or {}),
                            "primitive": {"name": primitive_name, "args": dict(primitive_args or {})},
                            "ok": result.ok,
                        }
                    )
                    return result

            skill_result = await execute_skill(RecordingExecutor(), context, skill, on_progress=on_progress)
            episode_ok = bool(skill_result.ok)
        else:
            episode_ok = False

        record = EpisodeRecord(episode_id=episode_id, skill_name=skill.name, steps=tuple(steps), ok=episode_ok)
        with dataset_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        completed += int(episode_ok)
        failed += int(not episode_ok)
        if on_progress is not None:
            await on_progress(f"Episode {episode_id}/{num_episodes} completed ({'ok' if episode_ok else 'failed'}).")

    message = f"Collected {completed} episodes of {skill.name}. Dataset saved."
    if failed:
        message = f"Collected {completed} of {num_episodes} episodes of {skill.name}. Dataset saved."
    return CollectionResult(
        ok=failed == 0,
        dataset_path=str(dataset_path),
        episodes_requested=num_episodes,
        episodes_completed=completed,
        episodes_failed=failed,
        message=message,
    )
