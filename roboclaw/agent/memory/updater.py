"""Semantic memory updater: writes episodic and semantic records after each task."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from roboclaw.agent.memory._utils import _safe_user_id
from roboclaw.agent.memory.episodic import (
    EpisodeAttempt,
    EpisodeSceneObject,
    EpisodicRecord,
    EpisodicStore,
)
from roboclaw.agent.memory.semantic import (
    FailureKnowledge,
    ObjectHandlingKnowledge,
    SemanticStore,
    SpatialAnchor,
)
from roboclaw.agent.memory.working import TaskWorkingMemory

_EMA_ALPHA = 0.3
_MAX_FAILURE_PATTERNS = 50


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _build_episode_objects(working: TaskWorkingMemory) -> list[EpisodeSceneObject]:
    result = []
    for obj in working.objects:
        result.append(EpisodeSceneObject(
            object_id=obj.object_id,
            raw_class_name=obj.raw_class_name,
            task_label=obj.task_label,
            pickable=obj.pickable,
            container_candidate=obj.container_candidate,
            stable=obj.stable,
            centroid_3d=None,
            extent_3d=None,
            attributes=dict(obj.attributes),
        ))
    return result


def _build_episode_attempts(working: TaskWorkingMemory) -> list[EpisodeAttempt]:
    result = []
    for log in working.attempt_log:
        result.append(EpisodeAttempt(
            action_type=log.action_type,
            target_object_id=log.object_id,
            target_raw_class=log.raw_class_name,
            status=log.status,
            reason=log.failure_reason,
            attempt_count=log.attempt_number,
            grasp_backend=log.grasp_backend,
            grasp_width_m=log.grasp_width_m,
            duration_ms=log.duration_ms,
        ))
    return result


def _build_episodic_record(
    working: TaskWorkingMemory,
    outcome: str,
    task_category: str,
    room_label: str | None,
    ended_at_iso: str,
    duration_ms: int,
) -> EpisodicRecord:
    objects_moved = [
        log.object_id
        for log in working.attempt_log
        if log.status == "success" and log.action_type == "pick"
    ]
    return EpisodicRecord(
        episode_id=EpisodicRecord.new_id(),
        user_id=working.user_id,
        session_key=working.session_key,
        task_id=working.task_id,
        user_goal=working.user_goal,
        task_category=task_category,
        started_at_iso=datetime.fromtimestamp(
            working.task_start_ms / 1000, tz=timezone.utc
        ).isoformat(),
        ended_at_iso=ended_at_iso,
        duration_ms=duration_ms,
        outcome=outcome,
        total_failure_count=working.failure_count,
        scene_objects=_build_episode_objects(working),
        objects_moved=objects_moved,
        attempts=_build_episode_attempts(working),
        room_label=room_label,
        camera_name="",
        retrieved_memory_summary=_summarize_retrieved_memory(working),
        planner_output=dict(working.active_plan),
        skill_sequence=list(working.skill_calls),
        personalization_decisions=list(working.planner_decision_trace),
        user_feedback="; ".join(working.user_corrections),
        satisfaction=(
            "corrected"
            if working.user_corrections
            else "satisfied"
            if outcome == "success"
            else "unknown"
        ),
    )


def _summarize_retrieved_memory(working: TaskWorkingMemory) -> str:
    if not working.retrieved_memory:
        return ""
    constraints = working.retrieved_memory.get("object_constraints", {})
    hands_off = constraints.get("hands_off", [])
    prefs = working.retrieved_memory.get("preferences", {})
    bits = []
    if hands_off:
        bits.append(f"hands_off={','.join(str(x) for x in hands_off)}")
    if prefs:
        bits.append(f"preferences={','.join(sorted(str(k) for k in prefs.keys()))}")
    return " | ".join(bits)


def _update_object_knowledge(mem, working: TaskWorkingMemory) -> None:
    for log in working.attempt_log:
        key = log.raw_class_name
        if key not in mem.object_knowledge:
            mem.object_knowledge[key] = ObjectHandlingKnowledge(raw_class_name=key)
        obj_k = mem.object_knowledge[key]

        if log.action_type == "pick":
            obj_k.total_pick_attempts += 1
            if log.status == "success":
                obj_k.successful_picks += 1
            obj_k.pick_success_rate = (
                obj_k.successful_picks / obj_k.total_pick_attempts
            )
            if log.status == "success" and log.grasp_width_m is not None:
                obj_k.best_grasp_width_m = log.grasp_width_m
            if log.status == "success" and log.grasp_backend:
                obj_k.best_grasp_backend = log.grasp_backend
            if log.failure_reason and log.failure_reason not in obj_k.known_failure_reasons:
                obj_k.known_failure_reasons.append(log.failure_reason)


def _ema_update(old: list[float], new: list[float], alpha: float) -> list[float]:
    if len(old) != len(new):
        return new
    return [alpha * n + (1 - alpha) * o for o, n in zip(old, new)]


def _update_spatial_anchors(mem, working: TaskWorkingMemory, now_iso: str) -> None:
    for obj in working.objects:
        if not obj.container_candidate:
            continue
        label = obj.task_label
        centroid = obj.attributes.get("centroid_3d")
        if not centroid or len(centroid) < 3:
            continue
        centroid = [float(v) for v in centroid[:3]]
        if label in mem.environment_map:
            anchor = mem.environment_map[label]
            anchor.centroid_3d = _ema_update(anchor.centroid_3d, centroid, _EMA_ALPHA)
            anchor.confirmed_count += 1
            anchor.last_seen_iso = now_iso
        else:
            mem.environment_map[label] = SpatialAnchor(
                label=label,
                centroid_3d=centroid,
                last_seen_iso=now_iso,
                confirmed_count=1,
                is_container=True,
            )


def _update_failure_patterns(mem, working: TaskWorkingMemory, now_iso: str) -> None:
    failure_counts: dict[tuple, int] = {}
    for log in working.attempt_log:
        if log.status in ("retryable_failure", "terminal_failure") and log.failure_reason:
            key = (log.raw_class_name, log.action_type, log.failure_reason)
            failure_counts[key] = failure_counts.get(key, 0) + 1

    for (cls, action, reason), count in failure_counts.items():
        if count < 2:
            continue
        matched = next(
            (f for f in mem.failure_patterns
             if f.raw_class_name == cls and f.action_type == action and f.recurring_reason == reason),
            None,
        )
        if matched:
            matched.occurrence_count += count
            matched.last_seen_iso = now_iso
        else:
            mem.failure_patterns.append(FailureKnowledge(
                raw_class_name=cls,
                action_type=action,
                recurring_reason=reason,
                occurrence_count=count,
                last_seen_iso=now_iso,
            ))

    if len(mem.failure_patterns) > _MAX_FAILURE_PATTERNS:
        mem.failure_patterns = sorted(
            mem.failure_patterns,
            key=lambda f: f.occurrence_count,
            reverse=True,
        )[:_MAX_FAILURE_PATTERNS]


class SemanticMemoryUpdater:
    def __init__(self, workspace: Path):
        self._workspace = workspace

    def _user_dir(self, user_id: str) -> Path:
        return self._workspace / "memory" / "users" / _safe_user_id(user_id)

    def record_task_completion(
        self,
        working: TaskWorkingMemory,
        outcome: str,
        task_category: str,
        room_label: str | None = None,
    ) -> None:
        now_ms = int(time.time() * 1000)
        duration_ms = now_ms - working.task_start_ms
        now_iso = _now_iso()

        user_dir = self._user_dir(working.user_id)
        ep_store = EpisodicStore(user_dir)
        record = _build_episodic_record(
            working, outcome, task_category, room_label, now_iso, duration_ms
        )
        ep_store.append(record)

        sem_store = SemanticStore(user_dir)
        mem = sem_store.load(working.user_id)

        mem.profile.total_tasks_attempted += 1
        if outcome == "success":
            mem.profile.total_tasks_completed += 1
        mem.profile.last_seen_iso = now_iso

        _update_object_knowledge(mem, working)
        _update_spatial_anchors(mem, working, now_iso)
        _update_failure_patterns(mem, working, now_iso)

        sem_store.save(mem)
