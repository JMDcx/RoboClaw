"""Shared schemas and storage for tidyup multi-agent coordination."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SceneObject:
    """Task-facing object state stored by the supervisor."""

    object_id: str
    raw_class_name: str
    task_label: str
    confidence: float
    bbox_xyxy: list[float]
    center_xy: list[float]
    stable: bool
    pickable: bool
    container_candidate: bool
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneObject":
        return cls(
            object_id=str(data.get("object_id") or data.get("track_id") or ""),
            raw_class_name=str(data.get("raw_class_name") or data.get("class_name") or "unknown"),
            task_label=str(data.get("task_label") or "unknown"),
            confidence=float(data.get("confidence") or 0.0),
            bbox_xyxy=[float(v) for v in data.get("bbox_xyxy", [])],
            center_xy=[float(v) for v in data.get("center_xy", [])],
            stable=bool(data.get("stable")),
            pickable=bool(data.get("pickable")),
            container_candidate=bool(data.get("container_candidate")),
            attributes=dict(data.get("attributes") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class HighLevelAction:
    """Structured action contract from supervisor to executor."""

    action_type: str
    target_object_id: str | None = None
    target_container_id: str | None = None
    constraints: dict[str, Any] = field(default_factory=dict)
    retry_budget: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExecutionReport:
    """Structured report emitted by the executor."""

    action_type: str
    status: str
    reason: str
    attempt_count: int
    observed_effect: dict[str, Any] = field(default_factory=dict)
    next_recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WorldState:
    """Shared supervisor-owned task state."""

    task_id: str
    user_goal: str = ""
    scene_frame_id: str = ""
    objects: list[SceneObject] = field(default_factory=list)
    target_object_id: str | None = None
    container_object_id: str | None = None
    candidate_grasps: list[dict[str, Any]] = field(default_factory=list)
    selected_grasp: dict[str, Any] | None = None
    grasp_backend: str | None = None
    grasp_frame_id: str = ""
    current_phase: str = "idle"
    last_executor_report: dict[str, Any] | None = None
    failure_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "user_goal": self.user_goal,
            "scene_frame_id": self.scene_frame_id,
            "objects": [obj.to_dict() for obj in self.objects],
            "target_object_id": self.target_object_id,
            "container_object_id": self.container_object_id,
            "candidate_grasps": self.candidate_grasps,
            "selected_grasp": self.selected_grasp,
            "grasp_backend": self.grasp_backend,
            "grasp_frame_id": self.grasp_frame_id,
            "current_phase": self.current_phase,
            "last_executor_report": self.last_executor_report,
            "failure_count": self.failure_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorldState":
        return cls(
            task_id=str(data.get("task_id") or "unknown"),
            user_goal=str(data.get("user_goal") or ""),
            scene_frame_id=str(data.get("scene_frame_id") or ""),
            objects=[SceneObject.from_dict(item) for item in data.get("objects", [])],
            target_object_id=data.get("target_object_id"),
            container_object_id=data.get("container_object_id"),
            candidate_grasps=[dict(item) for item in data.get("candidate_grasps", [])],
            selected_grasp=dict(data.get("selected_grasp")) if data.get("selected_grasp") else None,
            grasp_backend=data.get("grasp_backend"),
            grasp_frame_id=str(data.get("grasp_frame_id") or ""),
            current_phase=str(data.get("current_phase") or "idle"),
            last_executor_report=data.get("last_executor_report"),
            failure_count=int(data.get("failure_count") or 0),
        )


class TaskStateStore:
    """File-backed state store shared between supervisor and subagents."""

    def __init__(self, workspace: Path):
        self._root = workspace / ".roboclaw_tmp" / "task_state"

    def load(self, session_key: str) -> WorldState:
        path = self._path_for(session_key)
        if not path.exists():
            return WorldState(task_id=self._task_id_for(session_key))
        data = json.loads(path.read_text(encoding="utf-8"))
        return WorldState.from_dict(data)

    def save(self, session_key: str, state: WorldState) -> WorldState:
        path = self._path_for(session_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return state

    def reset(self, session_key: str) -> WorldState:
        state = WorldState(task_id=self._task_id_for(session_key))
        return self.save(session_key, state)

    def _path_for(self, session_key: str) -> Path:
        safe = hashlib.sha1(session_key.encode("utf-8")).hexdigest()[:16]
        return self._root / f"{safe}.json"

    @staticmethod
    def _task_id_for(session_key: str) -> str:
        return f"task-{hashlib.sha1(session_key.encode('utf-8')).hexdigest()[:10]}"
