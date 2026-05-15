"""Retrieval layer: builds personalized memory context for the system prompt."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboclaw.agent.memory._utils import _safe_user_id
from roboclaw.agent.memory.episodic import EpisodicStore
from roboclaw.agent.memory.semantic import SemanticStore, UserSemanticMemory

_DEFAULTS = {
    "confirm_before_pick": True,
    "confirm_before_place": False,
    "report_each_step": True,
    "report_only_on_completion": False,
    "use_terse_replies": False,
    "preferred_arm_side": "auto",
    "allow_multi_object_sessions": True,
}

_PREF_LABELS = {
    "confirm_before_pick": "Confirm before picking",
    "confirm_before_place": "Confirm before placing",
    "report_each_step": "Report each step",
    "report_only_on_completion": "Report only on completion",
    "use_terse_replies": "Terse replies",
    "preferred_arm_side": "Preferred arm",
    "allow_multi_object_sessions": "Multi-object sessions",
}


@dataclass
class MemoryContext:
    user_profile_block: str
    preferences_block: str
    object_knowledge_block: str
    spatial_block: str
    failure_warnings_block: str
    recent_tasks_block: str

    def render(self, max_tokens: int = 600) -> str:
        sections = [
            ("Profile", self.user_profile_block),
            ("Preferences", self.preferences_block),
            ("Object Knowledge", self.object_knowledge_block),
            ("Spatial Anchors", self.spatial_block),
            ("Failure Warnings", self.failure_warnings_block),
            ("Recent Tasks", self.recent_tasks_block),
        ]
        parts = ["## Robotic Memory"]
        for title, content in sections:
            if content:
                parts.append(f"### {title}\n{content}")
        result = "\n\n".join(parts)
        # Rough token budget: 1 token ~= 4 chars
        if len(result) > max_tokens * 4:
            result = result[: max_tokens * 4]
        return result


@dataclass
class PlanningMemoryContext:
    user_profile: dict[str, Any]
    preferences: dict[str, Any]
    operational_rules: dict[str, list[dict[str, Any]]]
    object_constraints: dict[str, Any]
    recent_relevant_episodes: list[dict[str, Any]]
    planner_hints: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_profile": self.user_profile,
            "preferences": self.preferences,
            "operational_rules": self.operational_rules,
            "object_constraints": self.object_constraints,
            "recent_relevant_episodes": self.recent_relevant_episodes,
            "planner_hints": self.planner_hints,
        }


class PersonalizedMemoryRetriever:
    def __init__(self, workspace: Path):
        self._workspace = workspace

    def _user_dir(self, user_id: str) -> Path:
        return self._workspace / "memory" / "users" / _safe_user_id(user_id)

    def get_context(self, user_id: str, task_category: str | None = None) -> MemoryContext:
        user_dir = self._user_dir(user_id)
        sem_store = SemanticStore(user_dir)
        mem = sem_store.load(user_id)
        ep_store = EpisodicStore(user_dir)
        recent = ep_store.recent(n=3)

        return MemoryContext(
            user_profile_block=_render_profile(mem),
            preferences_block=_render_prefs(mem),
            object_knowledge_block=_render_objects(mem),
            spatial_block=_render_spatial(mem),
            failure_warnings_block=_render_failures(mem),
            recent_tasks_block=_render_recent(recent),
        )

    def get_planning_context(
        self,
        user_id: str,
        task_category: str | None = None,
        scene_objects: list[str] | None = None,
    ) -> PlanningMemoryContext:
        user_dir = self._user_dir(user_id)
        mem = SemanticStore(user_dir).load(user_id)
        ep_store = EpisodicStore(user_dir)
        _keep = {"episode_id", "task_category", "outcome", "completed_subgoals", "failure_reasons"}
        raw_episodes = (
            ep_store.all_for_category(task_category)[-3:]
            if task_category
            else ep_store.recent(n=3)
        )
        recent = [
            {k: v for k, v in ep.items() if k in _keep}
            for ep in raw_episodes
        ]
        scene_set = {obj for obj in scene_objects or [] if obj}

        preferences = {
            pref.key: pref.value
            for pref in mem.preferences
            if pref.confidence >= 0.5
        }
        for obj, target in mem.kitchen.preferred_placement.items():
            preferences.setdefault(f"preferred_placement.{obj}", target)

        operational_rules: dict[str, list[dict[str, Any]]] = {}
        for rule in mem.operational_rules:
            if scene_set and rule.target and rule.target not in scene_set:
                continue
            operational_rules.setdefault(rule.rule_type, []).append({
                "rule_id": rule.rule_id,
                "target": rule.target,
                "value": rule.value,
                "priority": rule.priority,
                "source": rule.source,
                "confidence": rule.confidence,
            })

        hands_off = list(mem.kitchen.hands_off_object_classes)
        fragile = list(mem.kitchen.fragile_object_classes)
        for rule in mem.operational_rules:
            if rule.rule_type == "hands_off" and rule.target not in hands_off:
                hands_off.append(rule.target)
            if rule.rule_type == "fragile" and rule.target not in fragile:
                fragile.append(rule.target)

        object_constraints = {
            "hands_off": sorted(hands_off),
            "fragile": sorted(fragile),
            "preferred_placements": dict(mem.kitchen.preferred_placement),
            "object_knowledge": {
                key: {
                    "pick_success_rate": obj.pick_success_rate,
                    "user_prohibited": obj.user_prohibited,
                    "preferred_container_label": obj.preferred_container_label,
                    "known_failure_reasons": list(obj.known_failure_reasons),
                }
                for key, obj in mem.object_knowledge.items()
            },
        }

        planner_hints = _planning_hints(preferences, operational_rules, object_constraints)
        return PlanningMemoryContext(
            user_profile={
                "sender_id": mem.profile.sender_id,
                "display_name": mem.profile.display_name,
                "preferred_language": mem.profile.preferred_language,
                "tasks_completed": mem.profile.total_tasks_completed,
            },
            preferences=preferences,
            operational_rules=operational_rules,
            object_constraints=object_constraints,
            recent_relevant_episodes=recent,
            planner_hints=planner_hints,
        )


def _render_profile(mem: UserSemanticMemory) -> str:
    p = mem.profile
    name = p.display_name or p.sender_id or "unknown"
    return f"User: {name} | Language: {p.preferred_language} | Tasks completed: {p.total_tasks_completed}"


def _render_prefs(mem: UserSemanticMemory) -> str:
    ip = mem.interaction
    lines = []
    for attr, default in _DEFAULTS.items():
        val = getattr(ip, attr, default)
        if val != default:
            label = _PREF_LABELS.get(attr, attr)
            lines.append(f"- {label}: {val}")
    if mem.kitchen.preferred_placement:
        lines.append("- Preferred placement:")
        for obj, target in sorted(mem.kitchen.preferred_placement.items()):
            lines.append(f"  - {obj}: {target}")
    if mem.kitchen.tidy_dirty_dishes_to != "sink":
        lines.append(f"- Dirty dishes go to: {mem.kitchen.tidy_dirty_dishes_to}")
    if mem.kitchen.tidy_food_items_to != "counter":
        lines.append(f"- Food items go to: {mem.kitchen.tidy_food_items_to}")
    if mem.kitchen.fragile_object_classes:
        lines.append(f"- Fragile classes: {', '.join(mem.kitchen.fragile_object_classes)}")
    if mem.kitchen.hands_off_object_classes:
        lines.append(f"- Hands-off classes: {', '.join(mem.kitchen.hands_off_object_classes)}")
    if mem.preferences:
        lines.append("- Learned preferences:")
        for pref in mem.preferences[:8]:
            lines.append(f"  - {pref.scope}.{pref.key}: {pref.value} (confidence={pref.confidence:.2f})")
    if mem.operational_rules:
        lines.append("- Operational rules:")
        for rule in mem.operational_rules[:8]:
            lines.append(
                f"  - {rule.rule_type}:{rule.target} -> {rule.value} "
                f"(priority={rule.priority}, confidence={rule.confidence:.2f})"
            )
    return "\n".join(lines)


def _render_objects(mem: UserSemanticMemory) -> str:
    if not mem.object_knowledge:
        return ""
    top = sorted(
        mem.object_knowledge.values(),
        key=lambda o: o.total_pick_attempts,
        reverse=True,
    )[:8]
    rows = ["| Class | Success% | Notes |", "| --- | --- | --- |"]
    for obj in top:
        pct = f"{obj.pick_success_rate * 100:.0f}%"
        notes_parts = []
        if obj.user_prohibited:
            notes_parts.append("prohibited")
        if obj.fragile:
            notes_parts.append("fragile")
        if obj.best_grasp_backend:
            notes_parts.append(f"backend={obj.best_grasp_backend}")
        notes = ", ".join(notes_parts) or "-"
        rows.append(f"| {obj.raw_class_name} | {pct} | {notes} |")
    return "\n".join(rows)


def _render_spatial(mem: UserSemanticMemory) -> str:
    if not mem.environment_map:
        return ""
    anchors = list(mem.environment_map.values())[:10]
    lines = []
    for a in anchors:
        coords = ", ".join(f"{v:.2f}" for v in a.centroid_3d)
        tag = " [container]" if a.is_container else ""
        lines.append(f"- {a.label}: ({coords}){tag}")
    return "\n".join(lines)


def _render_failures(mem: UserSemanticMemory) -> str:
    if not mem.failure_patterns:
        return ""
    top = sorted(mem.failure_patterns, key=lambda f: f.occurrence_count, reverse=True)[:5]
    lines = []
    for f in top:
        mitigation = f" -> {f.mitigation}" if f.mitigation else ""
        lines.append(f"- {f.raw_class_name} {f.action_type}: {f.recurring_reason} (x{f.occurrence_count}){mitigation}")
    return "\n".join(lines)


def _render_recent(records: list[dict]) -> str:
    if not records:
        return ""
    lines = []
    for r in records:
        goal = r.get("user_goal", "")[:60]
        outcome = r.get("outcome", "unknown")
        cat = r.get("task_category", "")
        lines.append(f"- [{outcome}] {cat}: {goal}")
    return "\n".join(lines)


def _planning_hints(
    preferences: dict[str, Any],
    operational_rules: dict[str, list[dict[str, Any]]],
    object_constraints: dict[str, Any],
) -> list[str]:
    hints = []
    for obj in object_constraints.get("hands_off", []):
        hints.append(f"Do not plan skills that manipulate {obj}.")
    for key, value in sorted(preferences.items()):
        if key.startswith("preferred_placement."):
            obj = key.split(".", 1)[1]
            hints.append(f"Prefer placing {obj} at {value}.")
    for rule in operational_rules.get("verification", []):
        hints.append(f"Verification rule for {rule.get('target')}: {rule.get('value')}")
    return hints
