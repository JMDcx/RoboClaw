"""Working memory: per-task ephemeral state extending WorldState."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from roboclaw.embodied.tasking import SceneObject, WorldState


@dataclass
class ObjectAttemptLog:
    """Records a single pick or place attempt within a task."""

    object_id: str
    raw_class_name: str
    task_label: str
    action_type: str  # "pick" | "place"
    attempt_number: int
    status: str  # "success" | "retryable_failure" | "terminal_failure"
    failure_reason: str = ""
    grasp_backend: str = ""
    grasp_width_m: float | None = None
    centroid_3d: list[float] | None = None
    duration_ms: int = 0


@dataclass
class TaskWorkingMemory:
    """Per-task state combining WorldState fields with personalization context."""

    # WorldState fields
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

    # Personalization fields
    task_start_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    user_id: str = ""
    session_key: str = ""
    attempt_log: list[ObjectAttemptLog] = field(default_factory=list)
    user_preferred_language: str = "en"
    user_confirm_before_pick: bool = True
    user_confirm_before_place: bool = False
    avoid_object_classes: list[str] = field(default_factory=list)
    known_container_locations: dict[str, list[float]] = field(default_factory=dict)
    current_instruction: str = ""
    perception_snapshot: dict[str, Any] = field(default_factory=dict)
    active_plan: dict[str, Any] = field(default_factory=dict)
    retrieved_memory: dict[str, Any] = field(default_factory=dict)
    planner_decision_trace: list[dict[str, Any]] = field(default_factory=list)
    skill_calls: list[dict[str, Any]] = field(default_factory=list)
    user_corrections: list[str] = field(default_factory=list)

    def to_world_state(self) -> WorldState:
        """Return a WorldState view of the current task."""
        return WorldState(
            task_id=self.task_id,
            user_goal=self.user_goal,
            scene_frame_id=self.scene_frame_id,
            objects=self.objects,
            target_object_id=self.target_object_id,
            container_object_id=self.container_object_id,
            candidate_grasps=self.candidate_grasps,
            selected_grasp=self.selected_grasp,
            grasp_backend=self.grasp_backend,
            grasp_frame_id=self.grasp_frame_id,
            current_phase=self.current_phase,
            last_executor_report=self.last_executor_report,
            failure_count=self.failure_count,
        )
