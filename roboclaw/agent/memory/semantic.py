"""Semantic memory: JSON-backed persistent user knowledge."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = 2


@dataclass
class UserProfile:
    sender_id: str
    display_name: str = ""
    preferred_language: str = "en"
    first_seen_iso: str = ""
    last_seen_iso: str = ""
    total_tasks_completed: int = 0
    total_tasks_attempted: int = 0


@dataclass
class InteractionPreferences:
    confirm_before_pick: bool = True
    confirm_before_place: bool = False
    report_each_step: bool = True
    report_only_on_completion: bool = False
    use_terse_replies: bool = False
    preferred_arm_side: str = "auto"
    allow_multi_object_sessions: bool = True


@dataclass
class KitchenPreferences:
    preferred_placement: dict = field(default_factory=dict)
    tidy_dirty_dishes_to: str = "sink"
    tidy_food_items_to: str = "counter"
    tidy_trash_to: str = "trash_bin"
    hands_off_object_classes: list = field(default_factory=list)
    fragile_object_classes: list = field(default_factory=list)
    known_container_labels: list = field(default_factory=list)


@dataclass
class ObjectHandlingKnowledge:
    raw_class_name: str
    total_pick_attempts: int = 0
    successful_picks: int = 0
    pick_success_rate: float = 0.0
    best_grasp_width_m: float | None = None
    best_grasp_backend: str = ""
    known_failure_reasons: list[str] = field(default_factory=list)
    preferred_container_label: str = ""
    fragile: bool = False
    user_prohibited: bool = False


@dataclass
class SpatialAnchor:
    label: str
    centroid_3d: list[float]
    extent_3d: list[float] | None = None
    last_seen_iso: str = ""
    confirmed_count: int = 0
    is_container: bool = False


@dataclass
class FailureKnowledge:
    raw_class_name: str
    action_type: str
    recurring_reason: str
    occurrence_count: int
    last_seen_iso: str
    mitigation: str = ""


@dataclass
class PreferenceMemory:
    preference_id: str
    scope: str
    key: str
    value: Any
    source: str
    confidence: float
    evidence_count: int
    last_updated_iso: str


@dataclass
class OperationalRule:
    rule_id: str
    rule_type: str
    target: str
    value: Any
    priority: int
    source: str
    confidence: float


@dataclass
class UserSemanticMemory:
    profile: UserProfile
    interaction: InteractionPreferences
    kitchen: KitchenPreferences
    object_knowledge: dict[str, ObjectHandlingKnowledge] = field(default_factory=dict)
    environment_map: dict[str, SpatialAnchor] = field(default_factory=dict)
    failure_patterns: list[FailureKnowledge] = field(default_factory=list)
    preferences: list[PreferenceMemory] = field(default_factory=list)
    operational_rules: list[OperationalRule] = field(default_factory=list)
    schema_version: int = _SCHEMA_VERSION


def _deserialize(data: dict) -> UserSemanticMemory:
    p = data.get("profile", {})
    profile = UserProfile(
        sender_id=p.get("sender_id", ""),
        display_name=p.get("display_name", ""),
        preferred_language=p.get("preferred_language", "en"),
        first_seen_iso=p.get("first_seen_iso", ""),
        last_seen_iso=p.get("last_seen_iso", ""),
        total_tasks_completed=p.get("total_tasks_completed", 0),
        total_tasks_attempted=p.get("total_tasks_attempted", 0),
    )

    ip = data.get("interaction", {})
    interaction = InteractionPreferences(
        confirm_before_pick=ip.get("confirm_before_pick", True),
        confirm_before_place=ip.get("confirm_before_place", False),
        report_each_step=ip.get("report_each_step", True),
        report_only_on_completion=ip.get("report_only_on_completion", False),
        use_terse_replies=ip.get("use_terse_replies", False),
        preferred_arm_side=ip.get("preferred_arm_side", "auto"),
        allow_multi_object_sessions=ip.get("allow_multi_object_sessions", True),
    )

    kp = data.get("kitchen", {})
    kitchen = KitchenPreferences(
        preferred_placement=kp.get("preferred_placement", {}),
        tidy_dirty_dishes_to=kp.get("tidy_dirty_dishes_to", "sink"),
        tidy_food_items_to=kp.get("tidy_food_items_to", "counter"),
        tidy_trash_to=kp.get("tidy_trash_to", "trash_bin"),
        hands_off_object_classes=kp.get("hands_off_object_classes", []),
        fragile_object_classes=kp.get("fragile_object_classes", []),
        known_container_labels=kp.get("known_container_labels", []),
    )

    object_knowledge = {
        k: _deserialize_obj_knowledge(v)
        for k, v in data.get("object_knowledge", {}).items()
    }

    environment_map = {
        k: _deserialize_anchor(v)
        for k, v in data.get("environment_map", {}).items()
    }

    failure_patterns = [
        _deserialize_failure(f) for f in data.get("failure_patterns", [])
    ]

    return UserSemanticMemory(
        profile=profile,
        interaction=interaction,
        kitchen=kitchen,
        object_knowledge=object_knowledge,
        environment_map=environment_map,
        failure_patterns=failure_patterns,
        preferences=[
            _deserialize_preference(p) for p in data.get("preferences", [])
        ],
        operational_rules=[
            _deserialize_rule(r) for r in data.get("operational_rules", [])
        ],
        schema_version=max(int(data.get("schema_version", 1)), _SCHEMA_VERSION),
    )


def _deserialize_obj_knowledge(d: dict) -> ObjectHandlingKnowledge:
    return ObjectHandlingKnowledge(
        raw_class_name=d.get("raw_class_name", ""),
        total_pick_attempts=d.get("total_pick_attempts", 0),
        successful_picks=d.get("successful_picks", 0),
        pick_success_rate=d.get("pick_success_rate", 0.0),
        best_grasp_width_m=d.get("best_grasp_width_m"),
        best_grasp_backend=d.get("best_grasp_backend", ""),
        known_failure_reasons=d.get("known_failure_reasons", []),
        preferred_container_label=d.get("preferred_container_label", ""),
        fragile=d.get("fragile", False),
        user_prohibited=d.get("user_prohibited", False),
    )


def _deserialize_anchor(d: dict) -> SpatialAnchor:
    return SpatialAnchor(
        label=d.get("label", ""),
        centroid_3d=d.get("centroid_3d", [0.0, 0.0, 0.0]),
        extent_3d=d.get("extent_3d"),
        last_seen_iso=d.get("last_seen_iso", ""),
        confirmed_count=d.get("confirmed_count", 0),
        is_container=d.get("is_container", False),
    )


def _deserialize_failure(d: dict) -> FailureKnowledge:
    return FailureKnowledge(
        raw_class_name=d.get("raw_class_name", ""),
        action_type=d.get("action_type", ""),
        recurring_reason=d.get("recurring_reason", ""),
        occurrence_count=d.get("occurrence_count", 0),
        last_seen_iso=d.get("last_seen_iso", ""),
        mitigation=d.get("mitigation", ""),
    )


def _deserialize_preference(d: dict) -> PreferenceMemory:
    return PreferenceMemory(
        preference_id=d.get("preference_id", ""),
        scope=d.get("scope", ""),
        key=d.get("key", ""),
        value=d.get("value"),
        source=d.get("source", ""),
        confidence=float(d.get("confidence", 0.0)),
        evidence_count=int(d.get("evidence_count", 1)),
        last_updated_iso=d.get("last_updated_iso", ""),
    )


def _deserialize_rule(d: dict) -> OperationalRule:
    return OperationalRule(
        rule_id=d.get("rule_id", ""),
        rule_type=d.get("rule_type", ""),
        target=d.get("target", ""),
        value=d.get("value"),
        priority=int(d.get("priority", 0)),
        source=d.get("source", ""),
        confidence=float(d.get("confidence", 0.0)),
    )


class SemanticStore:
    """JSON-backed store for user semantic memory."""

    _FILENAME = "semantic.json"

    def __init__(self, user_dir: Path):
        self._path = user_dir / self._FILENAME
        self._user_dir = user_dir

    def load(self, user_id: str) -> UserSemanticMemory:
        if not self._path.exists():
            return UserSemanticMemory(
                profile=UserProfile(sender_id=user_id),
                interaction=InteractionPreferences(),
                kitchen=KitchenPreferences(),
            )
        data = json.loads(self._path.read_text(encoding="utf-8"))
        return _deserialize(data)

    def save(self, memory: UserSemanticMemory) -> None:
        self._user_dir.mkdir(parents=True, exist_ok=True)
        memory.schema_version = _SCHEMA_VERSION

        def _convert(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return asdict(obj)
            return obj

        self._path.write_text(
            json.dumps(asdict(memory), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
