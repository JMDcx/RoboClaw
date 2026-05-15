"""Memory intake: classify user messages into robotic memory updates.

The first implementation is deliberately conservative and deterministic. It
captures explicit personalization facts for the demo/research prototype, and
leaves ambiguous or low-confidence statements as episodic notes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryIntakeInput:
    user_id: str
    session_key: str
    user_message: str
    task_category: str | None = None
    current_scene: dict[str, Any] | None = None


@dataclass
class WorkingMemoryUpdate:
    update_type: str
    key: str
    value: Any
    confidence: float = 1.0


@dataclass
class EpisodicMemoryEvent:
    event_type: str
    content: str
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticMemoryUpdate:
    update_type: str
    scope: str
    key: str
    value: Any
    target: str = ""
    source: str = "explicit_user"
    confidence: float = 1.0
    priority: int = 0
    reason: str = ""


@dataclass
class MemoryIntakeResult:
    working_updates: list[WorkingMemoryUpdate] = field(default_factory=list)
    episodic_events: list[EpisodicMemoryEvent] = field(default_factory=list)
    semantic_updates: list[SemanticMemoryUpdate] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)


_DEFAULT_OBJECT_ALIASES = {
    "white mug": "white_mug",
    "white_mug": "white_mug",
    "yellow mug": "yellow_mug",
    "yellow_mug": "yellow_mug",
    "red mug": "red_mug",
    "red_mug": "red_mug",
    "left plate": "left_plate",
    "left_plate": "left_plate",
    "right plate": "right_plate",
    "right_plate": "right_plate",
    "bowl": "bowl",
    "碗": "bowl",
    "plate": "plate",
    "盘子": "plate",
    "stove": "stove",
    "炉子": "stove",
    "灶台": "stove",
    "cabinet": "cabinet",
    "柜子": "cabinet",
    "top of the cabinet": "top_of_cabinet",
    "cabinet top": "top_of_cabinet",
    "柜子上面": "top_of_cabinet",
    "柜子顶部": "top_of_cabinet",
    "柜子顶上": "top_of_cabinet",
    "middle drawer": "middle_drawer",
    "中间抽屉": "middle_drawer",
    "top drawer": "top_drawer",
    "上层抽屉": "top_drawer",
    "wine bottle": "wine_bottle",
    "wine_bottle": "wine_bottle",
    "红酒瓶": "wine_bottle",
    "葡萄酒瓶": "wine_bottle",
    "rack": "rack",
    "架子": "rack",
    "cream cheese": "cream_cheese",
    "cream_cheese": "cream_cheese",
    "奶油奶酪": "cream_cheese",
}

_PLACEMENT_OBJECTS = {
    "white_mug",
    "yellow_mug",
    "red_mug",
    "bowl",
    "wine_bottle",
    "cream_cheese",
    "plate",
}

_PLACEMENT_TARGETS = {
    "left_plate",
    "right_plate",
    "plate",
    "stove",
    "cabinet",
    "top_of_cabinet",
    "middle_drawer",
    "top_drawer",
    "rack",
    "bowl",
}

_NEGATION_MARKERS = (
    "不要碰",
    "别碰",
    "不要",
    "别",
    "不能",
    "禁止",
    "hands off",
    "do not touch",
    "don't touch",
    "never touch",
)

_PLACEMENT_MARKERS = (
    "放到",
    "放在",
    "放进",
    "摆到",
    "摆在",
    "收到",
    "收进",
    "整理到",
    "归位到",
    "应该放",
    "应该摆",
    "put",
    "place",
    "move",
    "belongs",
    "goes",
)


class MemoryIntakeClassifier:
    """Rule-first classifier for explicit robotic memory updates."""

    def classify(self, data: MemoryIntakeInput) -> MemoryIntakeResult:
        message = data.user_message.strip()
        result = MemoryIntakeResult()
        if not message:
            result.ignored.append("empty_message")
            return result

        result.working_updates.append(
            WorkingMemoryUpdate(
                update_type="task_command",
                key="current_instruction",
                value=message,
                confidence=1.0,
            )
        )

        aliases = _scene_aliases(data.current_scene)
        lower = _normalize_text(message)

        self._classify_hands_off(lower, aliases, result)
        self._classify_placement(lower, aliases, result)
        self._classify_interaction(lower, result)
        self._classify_feedback_and_corrections(lower, message, result)

        if not result.semantic_updates and not result.episodic_events:
            result.ignored.append("no_stable_memory_update_detected")
        return result

    def _classify_hands_off(
        self,
        lower: str,
        aliases: dict[str, str],
        result: MemoryIntakeResult,
    ) -> None:
        segments = [s.strip() for s in re.split(r"[，,。.;；]", lower) if s.strip()]
        hands_off_targets: list[str] = []
        for segment in segments:
            if not any(marker in segment for marker in _NEGATION_MARKERS):
                continue
            for alias, canonical in aliases.items():
                if alias in segment and canonical not in hands_off_targets:
                    hands_off_targets.append(canonical)

        for canonical in hands_off_targets:
            result.semantic_updates.append(
                SemanticMemoryUpdate(
                    update_type="operational_rule",
                    scope="object",
                    key="hands_off",
                    target=canonical,
                    value=True,
                    priority=100,
                    confidence=0.98,
                    reason=f"User explicitly marked {canonical} as hands-off.",
                )
            )

    def _classify_placement(
        self,
        lower: str,
        aliases: dict[str, str],
        result: MemoryIntakeResult,
    ) -> None:
        if not any(marker in lower for marker in _PLACEMENT_MARKERS):
            return

        segments = [s for s in re.split(r"[，,。.;；]", lower) if s.strip()]
        pairs: list[tuple[str, str]] = []
        for segment in segments:
            directional = _directional_placement_pair(segment, aliases)
            if directional:
                pairs.append(directional)
                continue
            objects = _dedupe_alias_matches([
                (a, c)
                for a, c in aliases.items()
                if c in _PLACEMENT_OBJECTS and a in segment
            ])
            containers = _dedupe_alias_matches([
                (a, c)
                for a, c in aliases.items()
                if c in _PLACEMENT_TARGETS and a in segment
            ])
            containers = [(a, c) for a, c in containers if c not in {obj for _, obj in objects}]
            if len(objects) == 1 and len(containers) == 1:
                pairs.append((objects[0][1], containers[0][1]))
        if not pairs:
            directional = _directional_placement_pair(lower, aliases)
            if directional:
                pairs.append(directional)
            else:
                objects = _dedupe_alias_matches([
                    (a, c)
                    for a, c in aliases.items()
                    if c in _PLACEMENT_OBJECTS and a in lower
                ])
                containers = _dedupe_alias_matches([
                    (a, c)
                    for a, c in aliases.items()
                    if c in _PLACEMENT_TARGETS and a in lower
                ])
                containers = [(a, c) for a, c in containers if c not in {obj for _, obj in objects}]
                if len(objects) == 1 and len(containers) == 1:
                    pairs.append((objects[0][1], containers[0][1]))

        for obj, container in pairs:
            result.semantic_updates.append(
                SemanticMemoryUpdate(
                    update_type="operational_rule",
                    scope="object",
                    key="placement",
                    target=obj,
                    value=container,
                    priority=50,
                    confidence=0.95,
                    reason=f"User explicitly prefers placing {obj} at {container}.",
                )
            )
            result.semantic_updates.append(
                SemanticMemoryUpdate(
                    update_type="preference",
                    scope="object",
                    key=f"preferred_placement.{obj}",
                    value=container,
                    target=obj,
                    confidence=0.95,
                    reason=f"Preferred placement learned for {obj}.",
                )
            )

    def _classify_interaction(self, lower: str, result: MemoryIntakeResult) -> None:
        if "简短" in lower or "terse" in lower or "short reply" in lower:
            result.semantic_updates.append(
                SemanticMemoryUpdate(
                    update_type="preference",
                    scope="interaction",
                    key="use_terse_replies",
                    value=True,
                    confidence=0.9,
                    reason="User prefers terse interaction.",
                )
            )
        if "每一步" in lower or "report each step" in lower:
            result.semantic_updates.append(
                SemanticMemoryUpdate(
                    update_type="preference",
                    scope="interaction",
                    key="report_each_step",
                    value=True,
                    confidence=0.9,
                    reason="User wants step-by-step reporting.",
                )
            )

    def _classify_feedback_and_corrections(
        self,
        lower: str,
        raw: str,
        result: MemoryIntakeResult,
    ) -> None:
        if any(marker in lower for marker in ("不对", "不是这样", "wrong", "instead", "i meant")):
            result.episodic_events.append(
                EpisodicMemoryEvent(
                    event_type="correction",
                    content=raw,
                    confidence=0.9,
                )
            )
        if any(marker in lower for marker in ("成功", "很好", "满意", "good", "great", "failed", "失败")):
            result.episodic_events.append(
                EpisodicMemoryEvent(
                    event_type="execution_feedback",
                    content=raw,
                    confidence=0.8,
                )
            )


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("-", "_").lower()).strip()


def _dedupe_alias_matches(matches: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Prefer specific aliases when a broad alias is a substring of another."""
    winners: list[tuple[str, str]] = []
    for alias, canonical in sorted(matches, key=lambda item: len(item[0]), reverse=True):
        if canonical in {existing for _, existing in winners}:
            continue
        if any(alias in existing_alias and alias != existing_alias for existing_alias, _ in winners):
            continue
        winners.append((alias, canonical))
    return winners


def _directional_placement_pair(segment: str, aliases: dict[str, str]) -> tuple[str, str] | None:
    for marker in _PLACEMENT_MARKERS:
        if marker not in segment:
            continue
        before, after = segment.split(marker, 1)
        objects = _dedupe_alias_matches([
            (a, c)
            for a, c in aliases.items()
            if c in _PLACEMENT_OBJECTS and a in before
        ])
        containers = _dedupe_alias_matches([
            (a, c)
            for a, c in aliases.items()
            if c in _PLACEMENT_TARGETS and a in after
        ])
        if len(objects) == 1 and len(containers) == 1:
            return objects[0][1], containers[0][1]
    return None


def _scene_aliases(scene: dict[str, Any] | None) -> dict[str, str]:
    aliases = dict(_DEFAULT_OBJECT_ALIASES)
    if not scene:
        return aliases
    for obj in scene.get("objects", []) or []:
        if not isinstance(obj, dict):
            continue
        for key in ("object_id", "raw_class_name", "task_label", "class_name"):
            value = obj.get(key)
            if not value:
                continue
            canonical = str(value).replace(" ", "_")
            aliases[str(value).replace("_", " ").lower()] = canonical
            aliases[canonical.lower()] = canonical
    return aliases
