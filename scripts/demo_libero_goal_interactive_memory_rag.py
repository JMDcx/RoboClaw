#!/usr/bin/env python3
"""Interactive LIBERO-goal memory RAG demo.

This script is intentionally demo-oriented: it keeps the already working
StarVLA rollout path untouched, while adding a richer memory/controller layer
in front of the VLA prompt.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LIBERO_GOAL_PROMPTS: dict[int, str] = {
    0: "open the middle drawer of the cabinet",
    1: "put the bowl on the stove",
    2: "put the wine bottle on top of the cabinet",
    3: "open the top drawer and put the bowl inside",
    4: "put the bowl on top of the cabinet",
    5: "push the plate to the front of the stove",
    6: "put the cream cheese in the bowl",
    7: "turn on the stove",
    8: "put the bowl on the plate",
    9: "put the wine bottle on the rack",
}

TASK_MAPPING: dict[tuple[str, str], int] = {
    ("bowl", "stove"): 1,
    ("bowl", "top_drawer"): 3,
    ("bowl", "top_of_cabinet"): 4,
    ("bowl", "plate"): 8,
    ("wine_bottle", "top_of_cabinet"): 2,
    ("wine_bottle", "rack"): 9,
    ("plate", "front_of_stove"): 5,
    ("cream_cheese", "bowl"): 6,
    ("stove", "on"): 7,
    ("cabinet_middle_drawer", "open"): 0,
}

SUPPORTED_OBJECTS = {
    "bowl",
    "wine_bottle",
    "plate",
    "cream_cheese",
    "stove",
    "cabinet_middle_drawer",
}
SUPPORTED_TARGETS = {
    "stove",
    "top_of_cabinet",
    "plate",
    "top_drawer",
    "middle_drawer",
    "rack",
    "bowl",
    "front_of_stove",
    "on",
    "open",
}

OBJECT_ALIASES: dict[str, str] = {
    "bowl": "bowl",
    "碗": "bowl",
    "碗具": "bowl",
    "wine bottle": "wine_bottle",
    "wine_bottle": "wine_bottle",
    "红酒瓶": "wine_bottle",
    "葡萄酒瓶": "wine_bottle",
    "酒瓶": "wine_bottle",
    "plate": "plate",
    "盘子": "plate",
    "cream cheese": "cream_cheese",
    "cream_cheese": "cream_cheese",
    "奶油奶酪": "cream_cheese",
    "stove": "stove",
    "炉子": "stove",
    "灶台": "stove",
    "middle drawer": "cabinet_middle_drawer",
    "中间抽屉": "cabinet_middle_drawer",
    "中层抽屉": "cabinet_middle_drawer",
}

TARGET_ALIASES: dict[str, str] = {
    "stove": "stove",
    "炉子": "stove",
    "灶台": "stove",
    "top of the cabinet": "top_of_cabinet",
    "top_of_cabinet": "top_of_cabinet",
    "cabinet top": "top_of_cabinet",
    "柜子上": "top_of_cabinet",
    "柜子上面": "top_of_cabinet",
    "柜子顶部": "top_of_cabinet",
    "柜子顶上": "top_of_cabinet",
    "plate": "plate",
    "盘子": "plate",
    "top drawer": "top_drawer",
    "top_drawer": "top_drawer",
    "上层抽屉": "top_drawer",
    "上面的抽屉": "top_drawer",
    "middle drawer": "middle_drawer",
    "middle_drawer": "middle_drawer",
    "中间抽屉": "middle_drawer",
    "rack": "rack",
    "架子": "rack",
    "酒架": "rack",
    "bowl": "bowl",
    "碗": "bowl",
    "front of the stove": "front_of_stove",
    "front_of_stove": "front_of_stove",
    "炉子前面": "front_of_stove",
    "灶台前面": "front_of_stove",
    "on": "on",
    "打开": "on",
    "open": "open",
}

TALK_MEMORY_MARKERS = (
    "以后",
    "记住",
    "我喜欢",
    "我希望",
    "偏好",
    "应该",
    "放在",
    "放到",
    "摆在",
    "摆到",
    "belongs",
    "prefer",
    "remember",
    "always",
    "all the time",
    "from now on",
    "when i call you",
    "when i ask you",
)
EXECUTION_NOW_MARKERS = (
    "现在执行",
    "现在开始",
    "现在帮我",
    "现在就",
    "立刻",
    "马上",
    "do it now",
    "execute now",
    "start now",
    "right now",
    "please do it",
    "run it",
)
MANIPULATION_MARKERS = (
    "整理",
    "收拾",
    "收一下",
    "放",
    "打开",
    "开启",
    "turn on",
    "open",
    "put",
    "push",
    "move",
    "tidy",
)
PLACEMENT_MARKERS = (
    "放在",
    "放到",
    "放进",
    "摆在",
    "摆到",
    "收到",
    "整理到",
    "归位到",
    "goes",
    "belongs",
    "put",
    "place",
)
MOVABLE_OBJECTS = {"bowl", "wine_bottle", "plate", "cream_cheese"}


def load_dotenv_minimal(path: Path) -> None:
    """Load KEY=value pairs without adding a python-dotenv dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def normalize_value(text: Any) -> str:
    return str(text or "").strip().lower().replace(" ", "_").replace("-", "_")


def user_dir(workspace: Path, user_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in user_id)
    return workspace / "memory" / "users" / (safe or "user")


def semantic_path(workspace: Path, user_id: str) -> Path:
    return user_dir(workspace, user_id) / "semantic.json"


def working_path(workspace: Path, user_id: str) -> Path:
    return user_dir(workspace, user_id) / "working.json"


def episodes_path(workspace: Path, user_id: str) -> Path:
    return user_dir(workspace, user_id) / "episodes.jsonl"


def load_json_file(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return dict(default)
    return data if isinstance(data, dict) else dict(default)


def save_json_file(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(data) + "\n", encoding="utf-8")


def default_semantic_memory(user_id: str) -> dict[str, Any]:
    now = datetime.now(tz=timezone.utc).isoformat()
    return {
        "profile": {
            "sender_id": user_id,
            "user_id": user_id,
            "preferred_language": "zh",
            "first_seen_iso": now,
            "last_seen_iso": now,
        },
        "interaction": {},
        "kitchen": {
            "preferred_placement": {},
            "hands_off_object_classes": [],
            "fragile_object_classes": [],
            "known_container_labels": [],
        },
        "object_knowledge": {},
        "environment_map": {},
        "failure_patterns": [],
        "preferences": [],
        "operational_rules": [],
        "schema_version": 2,
        "libero_goal": {
            "object_preferences": {},
            "interaction_preferences": {
                "confirm_before_execute": False,
                "report_rewritten_prompt": True,
                "report_memory_used": True,
            },
        },
    }


def ensure_libero_goal_semantic(data: dict[str, Any], user_id: str) -> dict[str, Any]:
    base = default_semantic_memory(user_id)
    data.setdefault("profile", {})
    data["profile"].setdefault("sender_id", user_id)
    data["profile"].setdefault("user_id", user_id)
    data["profile"].setdefault("preferred_language", "zh")
    data.setdefault("interaction", {})
    data.setdefault("kitchen", base["kitchen"])
    data["kitchen"].setdefault("preferred_placement", {})
    data.setdefault("preferences", [])
    data.setdefault("operational_rules", [])
    data.setdefault("schema_version", 2)
    data.setdefault("libero_goal", {})
    data["libero_goal"].setdefault("object_preferences", {})
    data["libero_goal"].setdefault(
        "interaction_preferences",
        base["libero_goal"]["interaction_preferences"],
    )
    return data


def load_semantic_memory(workspace: Path, user_id: str) -> dict[str, Any]:
    path = semantic_path(workspace, user_id)
    return ensure_libero_goal_semantic(load_json_file(path, default_semantic_memory(user_id)), user_id)


def save_semantic_memory(workspace: Path, user_id: str, data: dict[str, Any]) -> None:
    save_json_file(semantic_path(workspace, user_id), ensure_libero_goal_semantic(data, user_id))


def detect_aliases(text: str, aliases: dict[str, str]) -> list[str]:
    lower = text.lower().replace("-", "_")
    hits: list[str] = []
    for alias, canonical in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        if alias.lower() in lower and canonical not in hits:
            hits.append(canonical)
    return hits


def detect_objects(text: str, working: dict[str, Any] | None = None) -> list[str]:
    hits = detect_aliases(text, OBJECT_ALIASES)
    if hits:
        return hits
    if working:
        last = (
            (working.get("main_controller_decision") or {}).get("target_object")
            or (working.get("subagent_decision") or {}).get("target_object")
        )
        if last:
            return [str(last)]
    return []


def detect_targets(text: str) -> list[str]:
    return detect_aliases(text, TARGET_ALIASES)


def infer_directional_placement(text: str) -> tuple[str, str] | None:
    lower = text.lower().replace("-", "_")
    for marker in PLACEMENT_MARKERS:
        if marker not in lower:
            continue
        before, after = lower.split(marker, 1)
        objects = [obj for obj in detect_aliases(before, OBJECT_ALIASES) if obj in MOVABLE_OBJECTS]
        targets = detect_aliases(after, TARGET_ALIASES)
        targets = [target for target in targets if target not in objects]
        if len(objects) == 1 and len(targets) >= 1:
            return objects[0], targets[0]
    return None


def load_recent_episodes(workspace: Path, user_id: str, objects: list[str], limit: int = 3) -> list[dict[str, Any]]:
    path = episodes_path(workspace, user_id)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if objects and row.get("target_object") not in objects:
            continue
        rows.append(row)
    return rows[-limit:]


def build_rag_context(workspace: Path, user_id: str, user_message: str) -> dict[str, Any]:
    semantic = load_semantic_memory(workspace, user_id)
    working = load_json_file(working_path(workspace, user_id), {})
    objects = detect_objects(user_message, working)
    targets = detect_targets(user_message)
    object_preferences = semantic["libero_goal"].get("object_preferences", {})
    relevant_prefs = {
        obj: object_preferences.get(obj, {})
        for obj in objects
        if object_preferences.get(obj)
    }
    if not objects and object_preferences:
        relevant_prefs = dict(object_preferences)
    episodes = load_recent_episodes(workspace, user_id, objects)
    return {
        "detected_objects": objects,
        "detected_targets": targets,
        "semantic_preferences": relevant_prefs,
        "recent_episodes": episodes,
        "working_memory": working,
        "supported_tasks": LIBERO_GOAL_PROMPTS,
        "task_mapping": {f"{obj}->{target}": task_id for (obj, target), task_id in TASK_MAPPING.items()},
    }


def semantic_update_from_pair(obj: str, target: str) -> dict[str, Any]:
    return {
        "object": obj,
        "field": "preferred_target",
        "value": target,
        "confidence": 0.95,
    }


def rule_controller_decision(user_message: str, rag_context: dict[str, Any]) -> dict[str, Any]:
    lower = user_message.lower()
    objects = rag_context.get("detected_objects") or []
    targets = rag_context.get("detected_targets") or []
    prefs = rag_context.get("semantic_preferences") or {}
    directional = infer_directional_placement(user_message)

    if "成功" in lower or "失败" in lower or "不对" in lower or "failed" in lower or "success" in lower:
        return {
            "route": "talk",
            "assistant_reply": "收到，我会把这类反馈留给执行后的 episodic memory 记录。",
            "semantic_updates": [],
            "manipulation_plan": None,
            "source": "rule_feedback_talk",
        }

    if directional:
        object_hint, target_hint = directional
    else:
        object_hint = next((obj for obj in objects if obj in MOVABLE_OBJECTS), objects[0] if objects else "")
        target_hint = targets[0] if targets else ""
    is_memory_statement = is_memory_only_statement(user_message, object_hint, target_hint)
    is_manipulation = any(marker in lower for marker in MANIPULATION_MARKERS)

    if is_memory_statement and not lower.strip().startswith(("整理", "收拾", "收一下", "tidy")):
        return {
            "route": "talk",
            "assistant_reply": f"已记住：{object_hint} 以后优先放到 {target_hint}。",
            "semantic_updates": [semantic_update_from_pair(object_hint, target_hint)],
            "manipulation_plan": None,
            "source": "rule_semantic_update",
        }

    if object_hint:
        if target_hint == object_hint:
            target_hint = ""
        desired = target_hint or normalize_value((prefs.get(object_hint) or {}).get("preferred_target"))
        if not desired and object_hint == "stove" and ("打开" in lower or "turn on" in lower):
            desired = "on"
        if not desired and object_hint == "cabinet_middle_drawer":
            desired = "open"
        facts = [f"preferred_target.{object_hint}={desired}"] if desired else []
        if desired and ((object_hint, desired) in TASK_MAPPING):
            return {
                "route": "manipulation",
                "assistant_reply": f"我会根据记忆把这次任务规划为：{object_hint} -> {desired}。",
                "semantic_updates": [],
                "manipulation_plan": {
                    "target_object": object_hint,
                    "desired_target": desired,
                    "memory_facts_used": facts,
                    "reason": "rule controller used explicit target or semantic preference",
                },
                "source": "rule_manipulation",
            }

    if is_manipulation:
        return {
            "route": "talk",
            "assistant_reply": "我还不能把这个目标映射到当前 10 个 LIBERO-goal 任务，请明确物体和目标位置。",
            "semantic_updates": [],
            "manipulation_plan": None,
            "source": "rule_needs_clarification",
        }

    return {
        "route": "talk",
        "assistant_reply": "收到。你可以告诉我偏好，例如“以后碗放在炉子上”，也可以说“整理碗”。",
        "semantic_updates": [],
        "manipulation_plan": None,
        "source": "rule_general_talk",
    }


def is_memory_only_statement(user_message: str, object_hint: str = "", target_hint: str = "") -> bool:
    lower = user_message.lower()
    has_memory_marker = any(marker in lower for marker in TALK_MEMORY_MARKERS)
    has_now_marker = any(marker in lower for marker in EXECUTION_NOW_MARKERS)
    has_pair = bool(object_hint and target_hint) or infer_directional_placement(user_message) is not None
    return bool(has_memory_marker and has_pair and not has_now_marker)


async def chat_json(
    *,
    api_key: str,
    api_base: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 1200,
    temperature: float = 0.0,
) -> dict[str, Any]:
    from roboclaw.providers.custom_provider import CustomProvider

    provider = CustomProvider(api_key=api_key, api_base=api_base, default_model=model)
    response = await provider.chat(
        messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = response.content or ""
    if response.finish_reason == "error" or text.startswith("Error:"):
        return {"_error": text}
    parsed = extract_json_object(text)
    if not parsed:
        return {"_error": f"model_returned_non_json: {text[:500]}"}
    return parsed


async def call_main_controller(
    *,
    user_message: str,
    rag_context: dict[str, Any],
    api_key: str,
    api_base: str,
    model: str,
    allow_rule_fallback: bool = True,
) -> dict[str, Any]:
    if not api_key:
        return rule_controller_decision(user_message, rag_context)
    messages = [
        {
            "role": "system",
            "content": (
                "You are the main controller for a LIBERO-goal robot demo. "
                "Classify each user message as talk or manipulation. Extract stable "
                "user preferences into semantic_updates. For manipulation, use the "
                "RAG memory context to infer target_object and desired_target. "
                "Route rules are strict: if the user is teaching a persistent preference "
                "or habit, return route=talk even if the sentence contains an action phrase. "
                "Memory-only markers include remember, prefer, always, all the time, "
                "from now on, when I call you, when I ask you, 以后, 记住, 我喜欢, 我希望. "
                "Only return route=manipulation when the user clearly asks the robot to "
                "execute now, for example now, right now, please do it, execute now, start now, "
                "现在执行, 立刻, 马上, or when the sentence is a direct command without a "
                "memory-only marker. If a sentence both teaches a preference and asks for "
                "future behavior, update semantic memory but do not execute. "
                "Allowed target_object values: bowl, wine_bottle, plate, cream_cheese, "
                "stove, cabinet_middle_drawer. Allowed desired_target values: stove, "
                "top_of_cabinet, plate, top_drawer, rack, bowl, front_of_stove, on, open. "
                "Return JSON only with keys route, assistant_reply, semantic_updates, "
                "manipulation_plan."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User message:\n{user_message}\n\n"
                f"JSON RAG memory context:\n{json_dumps(rag_context)}\n\n"
                "Return one of these JSON shapes:\n"
                '{"route":"talk","assistant_reply":"...","semantic_updates":[{"object":"bowl","field":"preferred_target","value":"stove","confidence":0.95}],"manipulation_plan":null}\n'
                '{"route":"manipulation","assistant_reply":"...","semantic_updates":[],"manipulation_plan":{"target_object":"bowl","desired_target":"stove","memory_facts_used":["preferred_target.bowl=stove"],"reason":"..."}}'
            ),
        },
    ]
    parsed = await chat_json(
        api_key=api_key,
        api_base=api_base,
        model=model,
        messages=messages,
        max_tokens=1400,
        temperature=0.0,
    )
    if "_error" in parsed and allow_rule_fallback:
        fallback = rule_controller_decision(user_message, rag_context)
        fallback["controller_error"] = parsed["_error"]
        return fallback
    decision = normalize_controller_decision(parsed, user_message, rag_context, allow_rule_fallback)
    return enforce_memory_only_route(user_message, rag_context, decision)


def normalize_controller_decision(
    parsed: dict[str, Any],
    user_message: str,
    rag_context: dict[str, Any],
    allow_rule_fallback: bool,
) -> dict[str, Any]:
    route = str(parsed.get("route") or "").strip().lower()
    if route not in {"talk", "manipulation"}:
        return rule_controller_decision(user_message, rag_context) if allow_rule_fallback else {
            "route": "talk",
            "assistant_reply": "主控模型没有返回合法 route。",
            "semantic_updates": [],
            "manipulation_plan": None,
            "source": "invalid_controller_route",
        }
    parsed.setdefault("assistant_reply", "")
    parsed.setdefault("semantic_updates", [])
    if not isinstance(parsed["semantic_updates"], list):
        parsed["semantic_updates"] = []
    if route == "talk":
        parsed["manipulation_plan"] = None
    else:
        plan = parsed.get("manipulation_plan")
        if not isinstance(plan, dict):
            return rule_controller_decision(user_message, rag_context) if allow_rule_fallback else {
                "route": "talk",
                "assistant_reply": "主控模型没有返回合法 manipulation_plan。",
                "semantic_updates": parsed["semantic_updates"],
                "manipulation_plan": None,
                "source": "invalid_controller_plan",
            }
        plan["target_object"] = normalize_value(plan.get("target_object"))
        plan["desired_target"] = normalize_value(plan.get("desired_target"))
        if not isinstance(plan.get("memory_facts_used"), list):
            plan["memory_facts_used"] = []
        parsed["manipulation_plan"] = plan
    parsed.setdefault("source", "glm_main_controller")
    return parsed


def enforce_memory_only_route(
    user_message: str,
    rag_context: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    if decision.get("route") != "manipulation":
        return decision
    directional = infer_directional_placement(user_message)
    objects = rag_context.get("detected_objects") or []
    targets = rag_context.get("detected_targets") or []
    if directional:
        object_hint, target_hint = directional
    else:
        object_hint = next((obj for obj in objects if obj in MOVABLE_OBJECTS), objects[0] if objects else "")
        target_hint = targets[0] if targets else ""
        if object_hint == target_hint:
            target_hint = ""
    if not is_memory_only_statement(user_message, object_hint, target_hint):
        return decision

    updates = decision.get("semantic_updates") if isinstance(decision.get("semantic_updates"), list) else []
    if not updates and object_hint and target_hint:
        updates = [semantic_update_from_pair(object_hint, target_hint)]
    rewritten = dict(decision)
    rewritten["route"] = "talk"
    rewritten["semantic_updates"] = updates
    rewritten["manipulation_plan"] = None
    rewritten["assistant_reply"] = (
        f"已记住：以后 {object_hint} 优先放到 {target_hint}。这次只是记录偏好，不执行。"
        if object_hint and target_hint
        else "已记住这个偏好。这次只是记录，不执行。"
    )
    rewritten["route_override"] = "memory_only_statement_no_execute"
    return rewritten


def apply_semantic_updates(
    workspace: Path,
    user_id: str,
    updates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    semantic = load_semantic_memory(workspace, user_id)
    applied: list[dict[str, Any]] = []
    obj_prefs = semantic["libero_goal"]["object_preferences"]
    now = datetime.now(tz=timezone.utc).isoformat()
    for update in updates:
        if not isinstance(update, dict):
            continue
        obj = normalize_value(update.get("object"))
        field = normalize_value(update.get("field"))
        confidence = float(update.get("confidence", 0.0) or 0.0)
        if obj not in SUPPORTED_OBJECTS or confidence < 0.5:
            continue
        pref = obj_prefs.setdefault(obj, {"avoid_targets": [], "synonyms": [], "last_updated_by": "explicit_user"})
        if field == "preferred_target":
            value = normalize_value(update.get("value"))
            if value not in SUPPORTED_TARGETS:
                continue
            pref["preferred_target"] = value
            pref["last_updated_by"] = "explicit_user"
            pref["last_updated_iso"] = now
            semantic["kitchen"]["preferred_placement"][obj] = value
            applied.append({"object": obj, "field": field, "value": value, "confidence": confidence})
        elif field == "avoid_targets":
            raw_values = update.get("value")
            values = raw_values if isinstance(raw_values, list) else [raw_values]
            clean = [normalize_value(v) for v in values if normalize_value(v) in SUPPORTED_TARGETS]
            pref["avoid_targets"] = sorted(set((pref.get("avoid_targets") or []) + clean))
            pref["last_updated_iso"] = now
            applied.append({"object": obj, "field": field, "value": clean, "confidence": confidence})
        elif field == "synonyms":
            raw_values = update.get("value")
            values = raw_values if isinstance(raw_values, list) else [raw_values]
            clean = [str(v).strip() for v in values if str(v).strip()]
            pref["synonyms"] = sorted(set((pref.get("synonyms") or []) + clean))
            pref["last_updated_iso"] = now
            applied.append({"object": obj, "field": field, "value": clean, "confidence": confidence})
    if applied:
        save_semantic_memory(workspace, user_id, semantic)
    return applied


def validate_manipulation_plan(plan: dict[str, Any]) -> tuple[int, str] | None:
    obj = normalize_value(plan.get("target_object"))
    target = normalize_value(plan.get("desired_target"))
    task_id = TASK_MAPPING.get((obj, target))
    if task_id is None:
        return None
    return task_id, LIBERO_GOAL_PROMPTS[task_id]


async def call_prompt_subagent(
    *,
    manipulation_plan: dict[str, Any],
    api_key: str,
    api_base: str,
    model: str,
    allow_rule_fallback: bool = True,
) -> dict[str, Any]:
    deterministic = validate_manipulation_plan(manipulation_plan)
    if not api_key:
        if deterministic is None:
            return {"_error": "plan_cannot_map_to_supported_libero_goal_task"}
        task_id, prompt = deterministic
        return {
            "selected_task_id": task_id,
            "vla_prompt": prompt,
            "reason": "rule subagent mapped controller plan to supported prompt",
            "source": "rule_prompt_subagent",
        }
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict prompt translator for StarVLA LIBERO-goal. "
                "You may only choose one supported prompt exactly as written. "
                "Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Controller manipulation_plan:\n{json_dumps(manipulation_plan)}\n\n"
                "Supported prompts:\n"
                + "\n".join(f"{idx}: {prompt}" for idx, prompt in LIBERO_GOAL_PROMPTS.items())
                + '\n\nReturn {"selected_task_id":1,"vla_prompt":"put the bowl on the stove","reason":"..."}'
            ),
        },
    ]
    parsed = await chat_json(
        api_key=api_key,
        api_base=api_base,
        model=model,
        messages=messages,
        max_tokens=800,
        temperature=0.0,
    )
    if "_error" in parsed and allow_rule_fallback and deterministic is not None:
        task_id, prompt = deterministic
        return {
            "selected_task_id": task_id,
            "vla_prompt": prompt,
            "reason": "qwen failed; deterministic mapping used",
            "source": "rule_prompt_subagent",
            "subagent_error": parsed["_error"],
        }
    return normalize_subagent_decision(parsed, manipulation_plan, allow_rule_fallback)


def normalize_subagent_decision(
    parsed: dict[str, Any],
    manipulation_plan: dict[str, Any],
    allow_rule_fallback: bool,
) -> dict[str, Any]:
    try:
        task_id = int(parsed.get("selected_task_id"))
    except (TypeError, ValueError):
        task_id = -1
    prompt = str(parsed.get("vla_prompt") or "").strip()
    if task_id in LIBERO_GOAL_PROMPTS and prompt == LIBERO_GOAL_PROMPTS[task_id]:
        parsed["selected_task_id"] = task_id
        parsed["vla_prompt"] = prompt
        parsed.setdefault("reason", "")
        parsed.setdefault("source", "qwen_prompt_subagent")
        return parsed
    deterministic = validate_manipulation_plan(manipulation_plan)
    if allow_rule_fallback and deterministic is not None:
        fallback_task_id, fallback_prompt = deterministic
        return {
            "selected_task_id": fallback_task_id,
            "vla_prompt": fallback_prompt,
            "reason": "qwen returned invalid prompt; deterministic mapping used",
            "source": "rule_prompt_subagent",
            "invalid_qwen_output": parsed,
        }
    return {"_error": "invalid_subagent_prompt", "raw": parsed}


def write_working_memory(workspace: Path, user_id: str, payload: dict[str, Any]) -> None:
    save_json_file(working_path(workspace, user_id), payload)


def append_episode(workspace: Path, user_id: str, record: dict[str, Any]) -> None:
    path = episodes_path(workspace, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_vla_command(args: argparse.Namespace, task_id: int, prompt: str, video_out_path: Path) -> tuple[list[str], dict[str, str]]:
    starvla_dir = args.starvla_dir.expanduser().resolve()
    libero_home = args.libero_home.expanduser().resolve()
    ckpt = args.ckpt.expanduser().resolve()
    script = starvla_dir / "examples" / "LIBERO" / "eval_files" / "interactive_prompt_libero.py"
    env = os.environ.copy()
    env["LIBERO_HOME"] = str(libero_home)
    env["LIBERO_CONFIG_PATH"] = str(libero_home / "libero")
    env["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")
    env["PYOPENGL_PLATFORM"] = env.get("PYOPENGL_PLATFORM", "egl")
    pythonpath = [str(libero_home), str(starvla_dir)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    cmd = [
        args.libero_python,
        str(script),
        "--args.pretrained-path",
        str(ckpt),
        "--args.host",
        args.vla_host,
        "--args.port",
        str(args.vla_port),
        "--args.task-suite-name",
        "libero_goal",
        "--args.task-id",
        str(task_id),
        "--args.init-state-id",
        str(args.init_state_id),
        "--args.rollout-steps",
        str(args.rollout_steps),
        "--args.video-out-path",
        str(video_out_path),
        "--args.prompt",
        prompt,
    ]
    return cmd, env


def policy_server_command(args: argparse.Namespace) -> str:
    starvla_dir = args.starvla_dir.expanduser().resolve()
    script = starvla_dir / "examples" / "LIBERO" / "eval_files" / "run_policy_server.sh"
    parts = {
        "STARVLA_PYTHON": args.starvla_python,
        "CKPT": str(args.ckpt.expanduser().resolve()),
        "BASE_VLM": str(args.base_vlm.expanduser().resolve()),
        "GPU_ID": str(args.policy_gpu_id),
        "PORT": str(args.vla_port),
        "ATTN_IMPLEMENTATION": args.attn_implementation,
    }
    prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in parts.items() if value)
    return f"cd {shlex.quote(str(starvla_dir))} && {prefix} bash {shlex.quote(str(script))}"


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


async def process_turn(args: argparse.Namespace, user_message: str) -> None:
    workspace = args.workspace.expanduser().resolve()
    session_key = args.session_key
    rag_context = build_rag_context(workspace, args.user_id, user_message)
    controller = await call_main_controller(
        user_message=user_message,
        rag_context=rag_context,
        api_key=args.glm_api_key,
        api_base=args.glm_api_base,
        model=args.glm_model,
        allow_rule_fallback=args.allow_rule_fallback,
    )
    applied_updates = apply_semantic_updates(
        workspace,
        args.user_id,
        controller.get("semantic_updates") or [],
    )

    base_working = {
        "session_key": session_key,
        "last_user_message": user_message,
        "current_route": controller.get("route"),
        "retrieved_memory": rag_context,
        "main_controller_decision": controller,
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    print("\n" + "=" * 72)
    print(f"User: {user_message}")
    print("\n[JSON RAG memory]")
    print(json_dumps(rag_context))
    print("\n[Main controller]")
    print(json_dumps(controller))
    if applied_updates:
        print("\n[Semantic updates applied]")
        print(json_dumps(applied_updates))

    if controller.get("route") != "manipulation":
        write_working_memory(workspace, args.user_id, {
            **base_working,
            "current_skill": {"task_suite": "libero_goal", "rollout_steps": args.rollout_steps, "status": "no_action"},
        })
        print(f"\nAssistant: {controller.get('assistant_reply') or '收到。'}")
        print("=" * 72 + "\n")
        return

    plan = controller.get("manipulation_plan") or {}
    if validate_manipulation_plan(plan) is None:
        write_working_memory(workspace, args.user_id, {
            **base_working,
            "current_skill": {"task_suite": "libero_goal", "rollout_steps": args.rollout_steps, "status": "unmapped"},
        })
        print("\nAssistant: 这个 manipulation_plan 不能映射到当前 10 个 LIBERO-goal prompt，我不会执行 VLA。")
        print("=" * 72 + "\n")
        return

    subagent = await call_prompt_subagent(
        manipulation_plan=plan,
        api_key=args.qwen_api_key,
        api_base=args.qwen_api_base,
        model=args.qwen_model,
        allow_rule_fallback=args.allow_rule_fallback,
    )
    if "_error" in subagent:
        write_working_memory(workspace, args.user_id, {
            **base_working,
            "subagent_decision": subagent,
            "current_skill": {"task_suite": "libero_goal", "rollout_steps": args.rollout_steps, "status": "invalid_prompt"},
        })
        print("\n[Qwen prompt subagent]")
        print(json_dumps(subagent))
        print("\nAssistant: 子 agent 没有返回合法的 VLA prompt，我不会执行。")
        print("=" * 72 + "\n")
        return

    video_out_path = args.video_out_path.expanduser().resolve() if args.video_out_path else (
        workspace / "vla_rollouts" / "libero_goal"
    )
    video_out_path.mkdir(parents=True, exist_ok=True)
    cmd, env = build_vla_command(
        args,
        int(subagent["selected_task_id"]),
        str(subagent["vla_prompt"]),
        video_out_path,
    )
    working_payload = {
        **base_working,
        "subagent_decision": subagent,
        "current_skill": {
            "task_suite": "libero_goal",
            "rollout_steps": args.rollout_steps,
            "status": "pending_feedback" if args.execute_vla else "dry_run",
        },
    }
    write_working_memory(workspace, args.user_id, working_payload)

    print("\n[Qwen prompt subagent]")
    print(json_dumps(subagent))
    print("\n[Start policy server first if needed]")
    print(policy_server_command(args))
    print("\n[StarVLA rollout command]")
    print(shell_join(cmd))
    print(f"\nAssistant: {controller.get('assistant_reply') or '我会按记忆执行。'}")

    if not args.execute_vla:
        print("\nDry run only. Add --execute-vla to launch StarVLA.")
        print("=" * 72 + "\n")
        return

    completed = subprocess.run(cmd, cwd=args.starvla_dir, env=env, check=False)
    result = ""
    while result not in {"success", "partial", "failed", "skip"}:
        result = input("Result? [success/partial/failed/skip]: ").strip().lower()
    if result != "skip":
        feedback = input("Feedback note (optional): ").strip()
        episode = {
            "episode_id": str(uuid.uuid4()),
            "user_id": args.user_id,
            "session_key": session_key,
            "user_message": user_message,
            "target_object": normalize_value(plan.get("target_object")),
            "desired_target": normalize_value(plan.get("desired_target")),
            "selected_task_id": int(subagent["selected_task_id"]),
            "vla_prompt": str(subagent["vla_prompt"]),
            "memory_facts_used": plan.get("memory_facts_used") or [],
            "result": result,
            "user_feedback": feedback,
            "video_out_path": str(video_out_path),
            "returncode": int(completed.returncode),
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        append_episode(workspace, args.user_id, episode)
        working_payload["current_skill"]["status"] = result
        working_payload["last_episode_id"] = episode["episode_id"]
        write_working_memory(workspace, args.user_id, working_payload)
        print("\n[Episodic memory appended]")
        print(json_dumps(episode))
    print("=" * 72 + "\n")


async def interactive_loop(args: argparse.Namespace) -> None:
    workspace = args.workspace.expanduser().resolve()
    user_dir(workspace, args.user_id).mkdir(parents=True, exist_ok=True)
    save_semantic_memory(workspace, args.user_id, load_semantic_memory(workspace, args.user_id))

    print("\nLIBERO-goal interactive memory RAG demo")
    print(f"Workspace : {workspace}")
    print(f"User      : {args.user_id}")
    print(f"GLM       : {args.glm_api_base} | {args.glm_model}")
    print(f"Qwen      : {args.qwen_api_base} | {args.qwen_model}")
    print("Commands  : /quit, /memory, /working, /server")
    print()

    while True:
        try:
            user_message = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_message:
            continue
        if user_message in {"/quit", "/exit"}:
            break
        if user_message == "/memory":
            print(json_dumps(load_semantic_memory(workspace, args.user_id)))
            continue
        if user_message == "/working":
            print(json_dumps(load_json_file(working_path(workspace, args.user_id), {})))
            continue
        if user_message == "/server":
            print(policy_server_command(args))
            continue
        await process_turn(args, user_message)


def parse_args() -> argparse.Namespace:
    load_dotenv_minimal(Path(".env"))
    default_starvla = Path(os.environ.get("STARVLA_DIR", "/home/xinyuan/starVLA"))
    default_ckpt = default_starvla / "playground" / "Pretrained_models" / "StarVLA" / "Qwen3-VL-OFT-LIBERO-4in1" / "checkpoints" / "steps_50000_pytorch_model.pt"
    default_base_vlm = default_starvla / "playground" / "Pretrained_models" / "Qwen3-VL-4B-Instruct"

    parser = argparse.ArgumentParser(description="Interactive LIBERO-goal memory RAG demo")
    parser.add_argument("--workspace", type=Path, default=Path.home() / "tmp" / "rc_libero_goal_interactive_memory_rag")
    parser.add_argument("--user-id", default=os.environ.get("ROBOCLAW_USER_ID", "user"))
    parser.add_argument("--session-key", default=f"libero-goal-rag:{int(time.time())}")
    parser.add_argument("--execute-vla", action="store_true")
    parser.add_argument("--allow-rule-fallback", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--glm-api-key", default=os.environ.get("GLM_API_KEY", ""))
    parser.add_argument("--glm-api-base", default=os.environ.get("GLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4"))
    parser.add_argument("--glm-model", default=os.environ.get("GLM_MODEL", "glm-5v-turbo"))
    parser.add_argument("--qwen-api-key", default=os.environ.get("QWEN_API_KEY", ""))
    parser.add_argument("--qwen-api-base", default=os.environ.get("QWEN_API_BASE", os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")))
    parser.add_argument("--qwen-model", default=os.environ.get("QWEN_MODEL", "qwen-27b"))

    parser.add_argument("--starvla-dir", type=Path, default=default_starvla)
    parser.add_argument("--starvla-python", default=os.environ.get("STARVLA_PYTHON", sys.executable))
    parser.add_argument("--libero-home", type=Path, default=Path(os.environ.get("LIBERO_HOME", "/home/xinyuan/lerobot/lerobot-libero")))
    parser.add_argument("--libero-python", default=os.environ.get("LIBERO_Python", sys.executable))
    parser.add_argument("--ckpt", type=Path, default=Path(os.environ.get("CKPT", str(default_ckpt))))
    parser.add_argument("--base-vlm", type=Path, default=Path(os.environ.get("BASE_VLM", str(default_base_vlm))))
    parser.add_argument("--policy-gpu-id", default=os.environ.get("GPU_ID", "0"))
    parser.add_argument("--attn-implementation", default=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"))
    parser.add_argument("--vla-host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--vla-port", type=int, default=int(os.environ.get("PORT", "6694")))
    parser.add_argument("--init-state-id", type=int, default=int(os.environ.get("INIT_STATE_ID", "0")))
    parser.add_argument("--rollout-steps", type=int, default=int(os.environ.get("ROLLOUT_STEPS", "150")))
    parser.add_argument("--video-out-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(interactive_loop(args))


if __name__ == "__main__":
    main()
