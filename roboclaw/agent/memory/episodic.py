"""Episodic memory: JSONL-backed per-session task records."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EpisodeSceneObject:
    object_id: str
    raw_class_name: str
    task_label: str
    pickable: bool
    container_candidate: bool
    stable: bool
    centroid_3d: list[float] | None = None
    extent_3d: list[float] | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeAttempt:
    action_type: str
    target_object_id: str
    target_raw_class: str
    status: str
    reason: str
    attempt_count: int
    grasp_backend: str = ""
    grasp_width_m: float | None = None
    next_recommendation: str = ""
    duration_ms: int = 0


@dataclass
class EpisodicRecord:
    episode_id: str
    user_id: str
    session_key: str
    task_id: str
    user_goal: str
    task_category: str
    started_at_iso: str
    ended_at_iso: str
    duration_ms: int
    outcome: str  # "success" | "partial" | "failed" | "aborted"
    total_failure_count: int
    scene_objects: list[EpisodeSceneObject] = field(default_factory=list)
    objects_moved: list[str] = field(default_factory=list)
    attempts: list[EpisodeAttempt] = field(default_factory=list)
    room_label: str | None = None
    camera_name: str = ""
    consolidation_notes: str = ""
    retrieved_memory_summary: str = ""
    planner_output: dict[str, Any] = field(default_factory=dict)
    skill_sequence: list[dict[str, Any]] = field(default_factory=list)
    personalization_decisions: list[dict[str, Any]] = field(default_factory=list)
    user_feedback: str = ""
    satisfaction: str | None = None

    @staticmethod
    def new_id() -> str:
        return str(uuid.uuid4())


def _to_dict(obj) -> dict:
    return asdict(obj)


class EpisodicStore:
    """JSONL-backed store for episodic records under a user directory."""

    _FILENAME = "episodes.jsonl"

    def __init__(self, user_dir: Path):
        self._path = user_dir / self._FILENAME

    def append(self, record: EpisodicRecord) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_to_dict(record), ensure_ascii=False) + "\n")

    def recent(self, n: int = 20) -> list[dict]:
        if not self._path.exists():
            return []
        lines = self._path.read_text(encoding="utf-8").strip().splitlines()
        tail = lines[-n:] if len(lines) > n else lines
        return [json.loads(line) for line in tail]

    def all_for_category(self, task_category: str) -> list[dict]:
        if not self._path.exists():
            return []
        results = []
        for line in self._path.read_text(encoding="utf-8").strip().splitlines():
            rec = json.loads(line)
            if rec.get("task_category") == task_category:
                results.append(rec)
        return results
