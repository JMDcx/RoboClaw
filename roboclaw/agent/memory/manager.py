"""Facade for the personalized memory system."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

from roboclaw.agent.memory._utils import _safe_user_id
from roboclaw.agent.memory.intake import (
    MemoryIntakeClassifier,
    MemoryIntakeInput,
    MemoryIntakeResult,
    SemanticMemoryUpdate,
)
from roboclaw.agent.memory.retrieval import MemoryContext, PersonalizedMemoryRetriever
from roboclaw.agent.memory.semantic import (
    InteractionPreferences,
    ObjectHandlingKnowledge,
    OperationalRule,
    PreferenceMemory,
    SemanticStore,
)
from roboclaw.agent.memory.updater import SemanticMemoryUpdater
from roboclaw.agent.memory.working import TaskWorkingMemory


class PersonalizedMemoryManager:
    def __init__(self, workspace: Path):
        self._workspace = workspace
        self._retriever = PersonalizedMemoryRetriever(workspace)
        self._updater = SemanticMemoryUpdater(workspace)
        self._intake = MemoryIntakeClassifier()
        self._sem_store_cache: dict[str, SemanticStore] = {}

    def _user_dir(self, user_id: str) -> Path:
        return self._workspace / "memory" / "users" / _safe_user_id(user_id)

    def _get_sem_store(self, user_id: str) -> SemanticStore:
        if user_id not in self._sem_store_cache:
            self._sem_store_cache[user_id] = SemanticStore(self._user_dir(user_id))
        return self._sem_store_cache[user_id]

    def get_context(self, user_id: str, task_category: str | None = None) -> MemoryContext:
        return self._retriever.get_context(user_id, task_category)

    def get_planning_context(
        self,
        user_id: str,
        task_category: str | None = None,
        scene_objects: list[str] | None = None,
    ):
        return self._retriever.get_planning_context(user_id, task_category, scene_objects)

    def ingest_user_message(
        self,
        user_id: str,
        session_key: str,
        user_message: str,
        task_category: str | None = None,
        current_scene: dict | None = None,
    ) -> MemoryIntakeResult:
        result = self._intake.classify(
            MemoryIntakeInput(
                user_id=user_id,
                session_key=session_key,
                user_message=user_message,
                task_category=task_category,
                current_scene=current_scene,
            )
        )
        self._apply_semantic_updates(user_id, result.semantic_updates)
        return result

    def begin_task(
        self,
        user_id: str,
        session_key: str,
        task_id: str,
        user_goal: str,
        task_category: str | None = None,
    ) -> TaskWorkingMemory:
        sem_store = self._get_sem_store(user_id)
        mem = sem_store.load(user_id)
        planning_context = self.get_planning_context(user_id, task_category, None)

        return TaskWorkingMemory(
            task_id=task_id,
            user_goal=user_goal,
            current_instruction=user_goal,
            user_id=user_id,
            session_key=session_key,
            user_preferred_language=mem.profile.preferred_language,
            user_confirm_before_pick=mem.interaction.confirm_before_pick,
            user_confirm_before_place=mem.interaction.confirm_before_place,
            avoid_object_classes=list(mem.kitchen.hands_off_object_classes),
            retrieved_memory=planning_context.to_dict(),
            known_container_locations={
                label: list(anchor.centroid_3d)
                for label, anchor in mem.environment_map.items()
                if anchor.is_container
            },
        )

    def end_task(
        self,
        working: TaskWorkingMemory,
        outcome: str,
        task_category: str,
        room_label: str | None = None,
    ) -> None:
        self._updater.record_task_completion(working, outcome, task_category, room_label)

    def update_interaction_preference(self, user_id: str, field_name: str, value) -> None:
        if field_name not in InteractionPreferences.__dataclass_fields__:
            return
        sem_store = self._get_sem_store(user_id)
        mem = sem_store.load(user_id)
        setattr(mem.interaction, field_name, value)
        sem_store.save(mem)

    def prohibit_object_class(self, user_id: str, raw_class_name: str) -> None:
        sem_store = self._get_sem_store(user_id)
        mem = sem_store.load(user_id)

        if raw_class_name not in mem.object_knowledge:
            mem.object_knowledge[raw_class_name] = ObjectHandlingKnowledge(
                raw_class_name=raw_class_name
            )
        mem.object_knowledge[raw_class_name].user_prohibited = True

        if raw_class_name not in mem.kitchen.hands_off_object_classes:
            mem.kitchen.hands_off_object_classes.append(raw_class_name)

        sem_store.save(mem)

    def _apply_semantic_updates(
        self,
        user_id: str,
        updates: list[SemanticMemoryUpdate],
    ) -> None:
        if not updates:
            return
        sem_store = self._get_sem_store(user_id)
        mem = sem_store.load(user_id)
        now = datetime.now(tz=timezone.utc).isoformat()

        for update in updates:
            if update.confidence < 0.85:
                continue
            if update.update_type == "operational_rule":
                _upsert_operational_rule(mem, update)
                if update.key == "hands_off" and update.target:
                    if update.target not in mem.kitchen.hands_off_object_classes:
                        mem.kitchen.hands_off_object_classes.append(update.target)
                    if update.target not in mem.object_knowledge:
                        mem.object_knowledge[update.target] = ObjectHandlingKnowledge(
                            raw_class_name=update.target
                        )
                    mem.object_knowledge[update.target].user_prohibited = True
                if update.key == "placement" and update.target:
                    mem.kitchen.preferred_placement[update.target] = update.value
            elif update.update_type == "preference":
                _upsert_preference(mem, update, now)
                if update.scope == "interaction" and hasattr(mem.interaction, update.key):
                    setattr(mem.interaction, update.key, update.value)
                if update.key.startswith("preferred_placement.") and update.target:
                    mem.kitchen.preferred_placement[update.target] = update.value

        sem_store.save(mem)


def _upsert_operational_rule(mem, update: SemanticMemoryUpdate) -> None:
    rule_type = update.key
    matched = next(
        (
            r for r in mem.operational_rules
            if r.rule_type == rule_type and r.target == update.target
        ),
        None,
    )
    if matched:
        matched.value = update.value
        matched.priority = max(matched.priority, update.priority)
        matched.confidence = max(matched.confidence, update.confidence)
        matched.source = update.source
        return
    mem.operational_rules.append(
        OperationalRule(
            rule_id=f"rule_{uuid4().hex[:12]}",
            rule_type=rule_type,
            target=update.target,
            value=update.value,
            priority=update.priority,
            source=update.source,
            confidence=update.confidence,
        )
    )


def _upsert_preference(mem, update: SemanticMemoryUpdate, now_iso: str) -> None:
    matched = next(
        (
            p for p in mem.preferences
            if p.scope == update.scope and p.key == update.key
        ),
        None,
    )
    if matched:
        matched.value = update.value
        matched.confidence = max(matched.confidence, update.confidence)
        matched.evidence_count += 1
        matched.source = update.source
        matched.last_updated_iso = now_iso
        return
    mem.preferences.append(
        PreferenceMemory(
            preference_id=f"pref_{uuid4().hex[:12]}",
            scope=update.scope,
            key=update.key,
            value=update.value,
            source=update.source,
            confidence=update.confidence,
            evidence_count=1,
            last_updated_iso=now_iso,
        )
    )
