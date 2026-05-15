"""LIBERO IL skill execution tools.

Two tools share one persistent LIBERO env so that successive skill calls
continue from where the previous one left off:

- ``libero_perception`` renders a task-aware semantic scene.
- ``libero_plan`` maps the goal to ordered skill sub-goals.
- ``libero_cosmos_route`` uses a local Cosmos-Reason2 VLM as a fast skill router.
- ``libero_manipulation`` executes one planned sub-goal via a learned policy.
- ``libero_skill`` / ``libero_observe`` remain as lower-level compatibility tools.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import mimetypes
import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from PIL import Image

from roboclaw.agent.timing import log_event
from roboclaw.agent.tools.base import Tool, ToolResult


# ── Display thread (cv2 must live in one dedicated thread on Linux/X11) ───────

class _DisplayThread:
    """Singleton daemon thread that owns all cv2 windows."""

    _instance: "_DisplayThread | None" = None

    @classmethod
    def get(cls) -> "_DisplayThread":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=4)
        self._running = True
        self._t = threading.Thread(target=self._loop, daemon=True, name="libero-display")
        self._t.start()

    def push(self, window_name: str, frame_bgr: np.ndarray) -> None:
        try:
            self._q.put_nowait((window_name, frame_bgr))
        except queue.Full:
            pass  # drop frame rather than block

    def _loop(self) -> None:
        try:
            import cv2
        except ImportError:
            return
        while self._running:
            try:
                name, frame = self._q.get(timeout=0.05)
                cv2.imshow(name, frame)
                cv2.waitKey(1)
            except queue.Empty:
                cv2.waitKey(1)
            except Exception:
                break


# Default checkpoint locations (override via env vars or constructor)
_DEFAULT_SKILL_CKPTS = {
    "skill_06": "/home/xinyuan/lerobot/outputs/skills/dp_skill_06/checkpoints/last/pretrained_model",
    "skill_07": "/home/xinyuan/lerobot/outputs/skills/act_skill_07/checkpoints/last/pretrained_model",
}
_DEFAULT_SKILL_DATA = {
    "skill_06": "/home/xinyuan/datasets/datasets/libero_subtask_skills/subtask_06",
    "skill_07": "/home/xinyuan/datasets/datasets/libero_subtask_skills/subtask_07",
}
_DEFAULT_TASK_ID = 4
_DEFAULT_TASK = "libero_10"
_DEFAULT_YOLO_MODEL = "runs/detect/runs/yolo/roboclaw/weights/best.pt"
_YOLO_CLASS_NAMES = {
    0: "white_mug",
    1: "yellow_mug",
    2: "red_mug",
    3: "plate_left",
    4: "plate_right",
}
_YOLO_CLASS_ALIASES = {
    "left_plate": "plate_left",
    "right_plate": "plate_right",
}
_SUBGOAL_VERIFY_TARGETS = {
    "white_mug_to_right_plate": {
        "object_class": "white_mug",
        "target_class": "plate_right",
    },
    "yellow_mug_to_left_plate": {
        "object_class": "yellow_mug",
        "target_class": "plate_left",
    },
}
_OBS_HEIGHT = 256
_OBS_WIDTH = 256
_PROPRIO_LAYOUT = [
    "eef_x",
    "eef_y",
    "eef_z",
    "quat_x",
    "quat_y",
    "quat_z",
    "quat_w",
    "gripper_qpos_0_or_2",
]
_LIBERO_OBJECTS = [
    {
        "object_id": "white_mug",
        "raw_class_name": "white_mug",
        "class_name": "mug",
        "task_label": "white mug",
        "pickable": True,
        "container_candidate": False,
        "semantic_position": "right side of table at episode start",
    },
    {
        "object_id": "yellow_mug",
        "raw_class_name": "yellow_mug",
        "class_name": "mug",
        "task_label": "yellow mug",
        "pickable": True,
        "container_candidate": False,
        "semantic_position": "left side of table at episode start",
    },
    {
        "object_id": "red_mug",
        "raw_class_name": "red_mug",
        "class_name": "mug",
        "task_label": "red patterned mug",
        "pickable": False,
        "container_candidate": False,
        "semantic_position": "back center distractor",
    },
    {
        "object_id": "left_plate",
        "raw_class_name": "left_plate",
        "class_name": "plate",
        "task_label": "left plate",
        "pickable": False,
        "container_candidate": True,
        "semantic_position": "left edge of table",
    },
    {
        "object_id": "right_plate",
        "raw_class_name": "right_plate",
        "class_name": "plate",
        "task_label": "right plate",
        "pickable": False,
        "container_candidate": True,
        "semantic_position": "right edge of table",
    },
]
_LIBERO_TARGET_RELATIONS = [
    {
        "relation_id": "white_mug_to_right_plate",
        "object_id": "white_mug",
        "target_container_id": "right_plate",
        "description": "white mug should be on the right plate",
        "skill_id": "skill_06",
    },
    {
        "relation_id": "yellow_mug_to_left_plate",
        "object_id": "yellow_mug",
        "target_container_id": "left_plate",
        "description": "yellow mug should be on the left plate",
        "skill_id": "skill_07",
    },
]
_YOLO_MODEL_CACHE: dict[str, Any] = {}


def _libero_subgoals_for_goal(user_goal: str) -> list[dict[str, Any]]:
    """Return the task-aware LIBERO v1 subgoals in a requested order."""
    subgoals = [
        {
            "subgoal_id": "white_mug_to_right_plate",
            "description": "place the white mug on the right plate",
            "object_id": "white_mug",
            "target_container_id": "right_plate",
            "skill_id": "skill_06",
            "max_steps": 150,
            "requires_verification": False,
        },
        {
            "subgoal_id": "yellow_mug_to_left_plate",
            "description": "place the yellow mug on the left plate",
            "object_id": "yellow_mug",
            "target_container_id": "left_plate",
            "skill_id": "skill_07",
            "max_steps": 150,
            "requires_verification": False,
        },
    ]
    lower = user_goal.lower()
    skill07_idx = lower.find("skill_07")
    skill06_idx = lower.find("skill_06")
    yellow_idx = lower.find("yellow")
    white_idx = lower.find("white")
    requested_yellow_first = (
        (skill07_idx != -1 and (skill06_idx == -1 or skill07_idx < skill06_idx))
        or (yellow_idx != -1 and white_idx != -1 and yellow_idx < white_idx)
        or "yellow first" in lower
        or "skill_07 first" in lower
    )
    if requested_yellow_first:
        return [subgoals[1], subgoals[0]]
    return subgoals


def _parse_planning_memory_context(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _hands_off_from_planning_context(context: dict[str, Any]) -> set[str]:
    constraints = context.get("object_constraints", {})
    hands_off = set(str(v) for v in constraints.get("hands_off", []) if v)
    for rule in context.get("operational_rules", {}).get("hands_off", []):
        target = rule.get("target")
        if target and rule.get("value", True):
            hands_off.add(str(target))
    return hands_off


def _filter_subgoals_with_memory(
    subgoals: list[dict[str, Any]],
    planning_context: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hands_off = _hands_off_from_planning_context(planning_context)
    if not hands_off:
        return subgoals, []

    kept = []
    decisions = []
    for subgoal in subgoals:
        object_id = str(subgoal.get("object_id", ""))
        if object_id in hands_off:
            decisions.append({
                "decision": "exclude_subgoal",
                "subgoal_id": subgoal.get("subgoal_id"),
                "memory_source": f"operational_rule:hands_off:{object_id}",
                "reason": f"User marked {object_id} as hands-off.",
            })
            continue
        kept.append(subgoal)
    return kept, decisions


def _subgoal_by_id(plan: dict[str, Any], subgoal_id: str) -> dict[str, Any]:
    for subgoal in plan.get("ordered_subgoals", []) or []:
        if isinstance(subgoal, dict) and subgoal.get("subgoal_id") == subgoal_id:
            return subgoal
    return {}


def _default_skill_for_subgoal(subgoal_id: str) -> str:
    for relation in _LIBERO_TARGET_RELATIONS:
        if relation.get("relation_id") == subgoal_id:
            return str(relation.get("skill_id", ""))
    return ""


def _image_path_to_data_url(image_path: Path) -> str:
    mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _extract_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        return {}
    try:
        parsed = json.loads(text.strip())
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    # Find the first balanced {...} block, ignoring anything after it.
    start = text.find("{")
    if start < 0:
        return {}
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    parsed = json.loads(candidate)
                    return parsed if isinstance(parsed, dict) else {}
                except json.JSONDecodeError:
                    try:
                        import json_repair
                        parsed = json_repair.loads(candidate)
                        return parsed if isinstance(parsed, dict) else {}
                    except Exception:
                        return {}
    return {}


def _extract_router_decision_from_text(text: str) -> dict[str, Any]:
    """Best-effort fallback for small VLMs that answer in prose instead of JSON."""
    if not isinstance(text, str) or not text.strip():
        return {}
    lower = text.lower()

    skill_id = ""
    if "skill_06" in lower or "white_mug_to_right_plate" in lower:
        skill_id = "skill_06"
    elif "skill_07" in lower or "yellow_mug_to_left_plate" in lower:
        skill_id = "skill_07"
    elif re.search(r"\bdone\b|完成|已完成|satisfied|complete", lower):
        skill_id = "done"
    elif re.search(r"\bwait\b|等待|观察", lower):
        skill_id = "wait"
    elif re.search(r"\brecover\b|恢复|异常", lower):
        skill_id = "recover"
    elif "white" in lower and "right" in lower:
        skill_id = "skill_06"
    elif "yellow" in lower and "left" in lower:
        skill_id = "skill_07"
    if not skill_id:
        return {}

    subgoal_id = ""
    target = ""
    if skill_id == "skill_06":
        subgoal_id = "white_mug_to_right_plate"
        target = "right_plate"
    elif skill_id == "skill_07":
        subgoal_id = "yellow_mug_to_left_plate"
        target = "left_plate"

    completed: list[str] = []
    completed_markers = ("complete", "completed", "satisfied", "done", "完成", "已完成", "成功")
    if any(marker in lower for marker in completed_markers):
        if "white_mug_to_right_plate" in lower or ("white" in lower and "right" in lower):
            completed.append("white_mug_to_right_plate")
        if "yellow_mug_to_left_plate" in lower or ("yellow" in lower and "left" in lower):
            completed.append("yellow_mug_to_left_plate")

    confidence_match = re.search(r"(?:confidence|置信度)\D*([01](?:\.\d+)?)", lower)
    confidence = _coerce_confidence(confidence_match.group(1), 0.75) if confidence_match else 0.75
    should_execute = skill_id in _DEFAULT_SKILL_CKPTS
    return {
        "skill": skill_id,
        "skill_id": skill_id,
        "subgoal_id": subgoal_id,
        "target": target,
        "confidence": confidence,
        "should_execute": should_execute,
        "completed_subgoals": completed,
        "reason": "Parsed from non-JSON Cosmos response.",
        "raw_text": text[:1000],
        "parser_fallback": "text_skill_extraction",
    }


def _parse_plan_json(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    parsed = _extract_json_object(raw)
    return parsed if isinstance(parsed, dict) else {}


def _normalize_router_skill(raw_skill: Any, subgoal_id: str) -> str:
    skill = str(raw_skill or "").strip()
    aliases = {
        "white_mug_to_right_plate": "skill_06",
        "yellow_mug_to_left_plate": "skill_07",
        "place_white_mug": "skill_06",
        "place_yellow_mug": "skill_07",
        "white_mug": "skill_06",
        "yellow_mug": "skill_07",
    }
    if skill in aliases:
        return aliases[skill]
    if not skill:
        return _default_skill_for_subgoal(subgoal_id)
    return skill


def _coerce_confidence(value: Any, default: float = 0.0) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, confidence))


def _fallback_route_decision(
    *,
    intended_skill_id: str,
    subgoal_id: str,
    target: str,
    reason: str,
) -> dict[str, Any]:
    skill_id = intended_skill_id or _default_skill_for_subgoal(subgoal_id) or "wait"
    return {
        "skill": skill_id,
        "skill_id": skill_id,
        "subgoal_id": subgoal_id,
        "target": target,
        "confidence": 1.0,
        "should_execute": skill_id in _DEFAULT_SKILL_CKPTS,
        "reason": reason,
    }


def _cosmos_controller_perception(
    *,
    image_path: Path,
    plan: dict[str, Any],
    completed: set[str],
    execution_trace: list[dict[str, Any]],
    mgr: "LiberoEnvManager",
) -> dict[str, Any]:
    """Build a YOLO-free semantic perception packet for the Cosmos controller."""
    open_subgoals = [
        sg for sg in plan.get("ordered_subgoals", []) or []
        if isinstance(sg, dict) and str(sg.get("subgoal_id")) not in completed
    ]
    return {
        "source": "libero_perception_without_yolo",
        "frame_id": f"cosmos_controller_{int(time.time() * 1000)}",
        "image_path": str(image_path),
        "objects": _LIBERO_OBJECTS,
        "target_relations": _LIBERO_TARGET_RELATIONS,
        "open_subgoals": open_subgoals,
        "completed_subgoals": sorted(completed),
        "recent_execution_trace": execution_trace[-4:],
        "proprioception_8d": mgr.get_proprioception(),
        "detector": {"enabled": False, "status": "disabled"},
    }


def _completed_subgoals_from_decision(
    raw_decision: dict[str, Any],
    fallback_subgoal_id: str = "",
) -> set[str]:
    completed: set[str] = set()
    raw_completed = raw_decision.get("completed_subgoals")
    if isinstance(raw_completed, list):
        completed.update(str(item) for item in raw_completed if item)
    statuses = raw_decision.get("subgoal_statuses")
    if isinstance(statuses, dict):
        for subgoal_id, status in statuses.items():
            if str(status).lower() in {"done", "complete", "completed", "satisfied", "success"}:
                completed.add(str(subgoal_id))
    if raw_decision.get("completed") is True and fallback_subgoal_id:
        completed.add(fallback_subgoal_id)
    return completed


async def _call_cosmos_reason2_router(
    *,
    api_base: str,
    api_key: str,
    model: str,
    image_data_url: str,
    user_text: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> dict[str, Any]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=api_base, api_key=api_key, timeout=timeout_s)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a robot skill router. Look at the image and choose one skill. "
                    "Output ONLY raw JSON — no markdown, no code fences, no explanation, no schema echo. "
                    'Example: {"skill":"skill_06","subgoal_id":"white_mug_to_right_plate",'
                    '"confidence":0.9,"should_execute":true,"completed_subgoals":[],"reason":"white mug on table"}'
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": user_text},
                ],
            },
        ],
        max_tokens=max(1, int(max_tokens)),
        temperature=float(temperature),
    )
    content = response.choices[0].message.content or ""
    parsed = _extract_json_object(content)
    if not parsed:
        parsed = _extract_router_decision_from_text(content)
        if parsed:
            logger.warning(
                "Cosmos router returned prose; recovered decision via text parser (len={}): {!r}",
                len(content),
                content[:500],
            )
        else:
            logger.warning(
                "Cosmos router returned unparseable content (len={}): {!r}",
                len(content),
                content[:500],
            )
    else:
        logger.debug(
            "Cosmos router raw content (len={}): {!r}",
            len(content),
            content[:500],
        )
    return parsed


def _resolve_libero_yolo_model_path() -> Path:
    configured = os.environ.get("ROBOCLAW_LIBERO_YOLO_MODEL", _DEFAULT_YOLO_MODEL)
    path = Path(configured)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _model_names(model: Any) -> dict[int, str]:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        parsed = {int(idx): str(name) for idx, name in names.items()}
    elif isinstance(names, list):
        parsed = {idx: str(name) for idx, name in enumerate(names)}
    else:
        parsed = {}
    if any(name.startswith("class_") for name in parsed.values()):
        return dict(_YOLO_CLASS_NAMES)
    return parsed or dict(_YOLO_CLASS_NAMES)


def _run_yolo_detection(
    image_source: Path | np.ndarray,
    *,
    conf: float = 0.25,
    imgsz: int = 640,
) -> dict[str, Any]:
    """Run the optional local YOLO detector and return JSON-safe detections."""
    model_path = _resolve_libero_yolo_model_path()
    if not model_path.is_file():
        return {
            "enabled": False,
            "status": "model_not_found",
            "model_path": str(model_path),
            "detections": [],
        }

    try:
        from ultralytics import YOLO
    except Exception as exc:
        return {
            "enabled": False,
            "status": "ultralytics_import_error",
            "model_path": str(model_path),
            "error": str(exc),
            "detections": [],
        }

    try:
        cache_key = str(model_path)
        model = _YOLO_MODEL_CACHE.get(cache_key)
        if model is None:
            model = YOLO(cache_key, task="detect")
            _YOLO_MODEL_CACHE[cache_key] = model

        device = os.environ.get("ROBOCLAW_LIBERO_YOLO_DEVICE")
        predict_kwargs: dict[str, Any] = {
            "source": str(image_source) if isinstance(image_source, Path) else image_source,
            "conf": conf,
            "imgsz": imgsz,
            "verbose": False,
        }
        if device:
            predict_kwargs["device"] = device
        results = model.predict(**predict_kwargs)
    except Exception as exc:
        return {
            "enabled": True,
            "status": "predict_error",
            "model_path": str(model_path),
            "error": str(exc),
            "detections": [],
        }

    if not results:
        return {
            "enabled": True,
            "status": "ok",
            "model_path": str(model_path),
            "detections": [],
        }

    result = results[0]
    names = _model_names(model)
    height = int(getattr(result, "orig_shape", [0, 0])[0] or 0)
    width = int(getattr(result, "orig_shape", [0, 0])[1] or 0)
    detections: list[dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is not None:
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
        classes = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
        for idx, (box, score, cls_id) in enumerate(zip(xyxy, confs, classes, strict=False)):
            x1, y1, x2, y2 = [float(value) for value in box]
            class_id = int(cls_id)
            detection = {
                "detection_id": f"yolo_{idx:02d}",
                "class_id": class_id,
                "class_name": names.get(class_id, f"class_{class_id}"),
                "confidence": float(score),
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_center_xy": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
            }
            if width > 0 and height > 0:
                detection["bbox_xyxy_norm"] = [x1 / width, y1 / height, x2 / width, y2 / height]
                detection["bbox_center_xy_norm"] = [
                    ((x1 + x2) / 2.0) / width,
                    ((y1 + y2) / 2.0) / height,
                ]
            detections.append(detection)

    return {
        "enabled": True,
        "status": "ok",
        "model_path": str(model_path),
        "class_names": names,
        "image_size": {"width": width, "height": height},
        "detections": detections,
    }


def _canonical_yolo_class(name: str) -> str:
    return _YOLO_CLASS_ALIASES.get(name, name)


def _best_detection(detections: list[dict[str, Any]], class_name: str) -> dict[str, Any] | None:
    target = _canonical_yolo_class(class_name)
    matches = [
        detection
        for detection in detections
        if _canonical_yolo_class(str(detection.get("class_name"))) == target
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: float(item.get("confidence") or 0.0))


def _expanded_box(box: list[float], margin_ratio: float = 0.2, min_margin: float = 8.0) -> list[float]:
    x1, y1, x2, y2 = box
    margin_x = max(min_margin, (x2 - x1) * margin_ratio)
    margin_y = max(min_margin, (y2 - y1) * margin_ratio)
    return [x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y]


def _point_in_box(point: tuple[float, float], box: list[float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def _intersection_area(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    return width * height


def _box_area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _verify_subgoal_from_detections(
    subgoal_id: str,
    detections: list[dict[str, Any]],
) -> dict[str, Any]:
    target = _SUBGOAL_VERIFY_TARGETS.get(subgoal_id)
    if target is None:
        return {
            "subgoal_id": subgoal_id,
            "status": "unsupported_subgoal",
            "satisfied": None,
            "reason": "No local CV rule is defined for this sub-goal.",
        }

    object_det = _best_detection(detections, target["object_class"])
    target_det = _best_detection(detections, target["target_class"])
    if object_det is None or target_det is None:
        missing = []
        if object_det is None:
            missing.append(target["object_class"])
        if target_det is None:
            missing.append(target["target_class"])
        return {
            "subgoal_id": subgoal_id,
            "status": "uncertain",
            "satisfied": None,
            "reason": f"YOLO did not detect required classes: {', '.join(missing)}.",
            "object_class": target["object_class"],
            "target_class": target["target_class"],
            "missing_classes": missing,
        }

    object_box = [float(value) for value in object_det["bbox_xyxy"]]
    target_box = [float(value) for value in target_det["bbox_xyxy"]]
    expanded_target = _expanded_box(target_box)
    object_bottom_center = ((object_box[0] + object_box[2]) / 2.0, object_box[3])
    object_center = ((object_box[0] + object_box[2]) / 2.0, (object_box[1] + object_box[3]) / 2.0)
    overlap = _intersection_area(object_box, expanded_target)
    overlap_ratio = overlap / max(1.0, min(_box_area(object_box), _box_area(expanded_target)))
    bottom_center_inside = _point_in_box(object_bottom_center, expanded_target)
    center_inside = _point_in_box(object_center, expanded_target)
    satisfied = bottom_center_inside or center_inside or overlap_ratio >= 0.05
    return {
        "subgoal_id": subgoal_id,
        "status": "satisfied" if satisfied else "not_satisfied",
        "satisfied": satisfied,
        "reason": (
            "Target mug geometry overlaps or lies inside the target plate region."
            if satisfied
            else "Target mug is detected away from the target plate region."
        ),
        "object_class": target["object_class"],
        "target_class": target["target_class"],
        "object_detection": object_det,
        "target_detection": target_det,
        "geometry": {
            "object_bottom_center_xy": list(object_bottom_center),
            "object_center_xy": list(object_center),
            "expanded_target_bbox_xyxy": expanded_target,
            "bottom_center_inside_target": bottom_center_inside,
            "center_inside_target": center_inside,
            "overlap_ratio": overlap_ratio,
        },
    }


async def _ensure_libero_env(
    mgr: LiberoEnvManager,
    *,
    task_id: int,
    episode_length: int,
    reset: bool,
    seed: int,
    protect_existing_episode: bool = False,
) -> str | None:
    try:
        await asyncio.to_thread(mgr.ensure_env, _DEFAULT_TASK, task_id, episode_length)
    except Exception as exc:
        return f"Error initializing LIBERO env: {exc}"

    if protect_existing_episode and reset and mgr._last_obs is not None:
        return None

    if reset or mgr._last_obs is None:
        try:
            await asyncio.to_thread(mgr.reset, seed)
        except Exception as exc:
            return f"Error resetting LIBERO env: {exc}"
    return None


async def _render_libero_frame(
    mgr: LiberoEnvManager,
    frames_dir: Path,
    prefix: str,
) -> tuple[np.ndarray | None, Path | None, str | None]:
    try:
        frame = await asyncio.to_thread(mgr.render)
    except Exception as exc:
        return None, None, f"Error rendering LIBERO scene: {exc}"
    ts = int(time.time() * 1000)
    image_path = frames_dir / f"{prefix}_{ts}.png"
    Image.fromarray(frame).save(image_path)
    return frame, image_path, None


@dataclass
class _LoadedPolicy:
    policy: Any
    env_pre: Any
    policy_pre: Any
    policy_post: Any


class LiberoEnvManager:
    """Module-level singleton holding the LIBERO env + lazily loaded policies."""

    _instance: "LiberoEnvManager | None" = None

    @classmethod
    def get(cls) -> "LiberoEnvManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._env = None
        self._env_cfg = None
        self._task_id: int | None = None
        self._policies: dict[str, _LoadedPolicy] = {}
        self._last_obs: dict | None = None
        self._device = "cuda"
        self._session_frames: list[np.ndarray] = []  # accumulates across all skill calls
        self._record_video: bool = False
        self._pipeline_subgoal_attempts: dict[str, int] = {}
        self._overlay: dict[str, str] = {"stage": "", "memory": "", "plan": ""}
        self._episode_done: bool = False

    # ── env lifecycle ────────────────────────────────────────────────────────

    def ensure_env(self, task: str, task_id: int, episode_length: int) -> None:
        """Boot the LIBERO env on first use."""
        if self._env is not None:
            return
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig

        self._env_cfg = LiberoEnvConfig(
            task=task,
            task_ids=[task_id],
            fps=10,
            episode_length=episode_length,
            obs_type="pixels_agent_pos",
            observation_height=_OBS_HEIGHT,
            observation_width=_OBS_WIDTH,
        )
        envs_dict = self._env_cfg.create_envs(n_envs=1)
        self._env = envs_dict[task][task_id]
        self._task_id = task_id

    def reset(self, seed: int | None = None) -> None:
        if self._env is None:
            raise RuntimeError("env not initialized; call ensure_env first")
        obs, _ = self._env.reset(seed=seed)
        self._last_obs = obs
        self._pipeline_subgoal_attempts: dict[str, int] = {}
        self._episode_done = False

    def start_recording(self) -> None:
        self._session_frames = []
        self._record_video = True

    def stop_recording(self) -> None:
        self._record_video = False

    def set_stage(self, label: str) -> None:
        self._overlay["stage"] = label or ""

    def set_memory_summary(self, text: str) -> None:
        self._overlay["memory"] = text or ""

    def set_plan_summary(self, text: str) -> None:
        self._overlay["plan"] = text or ""

    def _record_frame(self, frame_rgb: np.ndarray) -> None:
        """Upscale, annotate, and append one RGB frame to the session buffer."""
        if not self._record_video:
            return
        try:
            annotated = self._apply_overlay(frame_rgb)
        except Exception:
            annotated = frame_rgb.copy()
        self._session_frames.append(annotated)

    def _apply_overlay(self, frame_rgb: np.ndarray) -> np.ndarray:
        import cv2  # noqa: PLC0415

        h, w = frame_rgb.shape[:2]
        target_w = 640
        if w < target_w:
            scale_xy = target_w / w
            frame_rgb = cv2.resize(
                frame_rgb,
                (target_w, int(round(h * scale_xy))),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            frame_rgb = frame_rgb.copy()

        h, w = frame_rgb.shape[:2]
        font_scale = 0.55
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1

        def put(text: str, x: int, y: int) -> None:
            if not text:
                return
            cv2.putText(frame_rgb, text, (x + 1, y + 1), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            cv2.putText(frame_rgb, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        stage = self._overlay.get("stage", "")
        memory = self._overlay.get("memory", "")
        plan = self._overlay.get("plan", "")
        if stage:
            put(f"Stage: {stage}", 8, 22)
        if plan:
            put(f"Plan: {plan}", 8, h - 10)
        if memory:
            put(f"Memory: {memory}", 8, h - 30)
        return frame_rgb

    def save_video(self, path: Path, fps: int = 10) -> Path | None:
        if not self._session_frames:
            return None
        from lerobot.utils.io_utils import write_video
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_video(path, self._session_frames, fps=fps)
        return path

    def render(self) -> np.ndarray:
        if self._env is None:
            raise RuntimeError("env not initialized")
        frame = self._env.envs[0].render()  # HWC uint8 RGB
        self._record_frame(frame)
        return frame

    def get_proprioception(self) -> list[float] | None:
        """Extract 8-d proprioception from last_obs (eef pos+axis-angle + gripper)."""
        if self._last_obs is None:
            return None
        rs = self._last_obs.get("robot_state")
        if not rs:
            return None
        try:
            eef_pos = np.asarray(rs["eef"]["pos"]).flatten()
            eef_quat = np.asarray(rs["eef"]["quat"]).flatten()
            gripper = np.asarray(rs["gripper"]["qpos"]).flatten()
            return [*eef_pos.tolist(), *eef_quat.tolist(), *gripper.tolist()]
        except Exception:
            return None

    # ── policy lifecycle ─────────────────────────────────────────────────────

    def ensure_policy(self, skill_id: str) -> _LoadedPolicy:
        if skill_id in self._policies:
            return self._policies[skill_id]
        if skill_id not in _DEFAULT_SKILL_CKPTS:
            raise ValueError(f"Unknown skill_id={skill_id!r}; known: {list(_DEFAULT_SKILL_CKPTS)}")

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.policies import make_policy, make_pre_post_processors

        ckpt_path = Path(_DEFAULT_SKILL_CKPTS[skill_id])
        data_path = Path(_DEFAULT_SKILL_DATA[skill_id])

        cfg = PreTrainedConfig.from_pretrained(ckpt_path)
        cfg.pretrained_path = ckpt_path
        cfg.device = self._device
        ds_meta = LeRobotDatasetMetadata(repo_id=data_path.name, root=data_path)
        policy = make_policy(cfg=cfg, ds_meta=ds_meta)
        policy.eval()

        env_pre, _ = make_env_pre_post_processors(env_cfg=self._env_cfg, policy_cfg=cfg)
        policy_pre, policy_post = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=ckpt_path,
            preprocessor_overrides={"device_processor": {"device": self._device}},
        )
        loaded = _LoadedPolicy(policy=policy, env_pre=env_pre, policy_pre=policy_pre, policy_post=policy_post)
        self._policies[skill_id] = loaded
        return loaded

    # ── execution ────────────────────────────────────────────────────────────

    def run_skill(
        self,
        skill_id: str,
        max_steps: int,
        show_window: bool = False,
        verifier_subgoal_id: str | None = None,
        verifier_hz: float = 5.0,
        verifier_conf: float = 0.25,
        verifier_start_step: int = 10,
        verifier_early_stop: bool = False,
        verifier_settle_steps: int = 30,
        reset_policy: bool = True,
    ) -> dict[str, Any]:
        """Synchronous skill rollout. Returns a summary dict.

        Caller is responsible for ensuring env is reset to a known state if needed.
        """
        import torch

        from lerobot.envs.utils import preprocess_observation

        if self._env is None:
            raise RuntimeError("env not initialized")
        if self._last_obs is None:
            raise RuntimeError("no current obs; call reset() before running a skill")

        loaded = self.ensure_policy(skill_id)
        if reset_policy:
            loaded.policy.reset()

        # Start display thread if live window requested
        display: _DisplayThread | None = None
        if show_window:
            try:
                import cv2  # noqa: PLC0415 — just check it's available
                display = _DisplayThread.get()
            except ImportError:
                display = None

        steps = 0
        success = False
        last_reward = 0.0
        window_name = f"LIBERO – {skill_id}"
        local_verifier: dict[str, Any] = {
            "enabled": bool(verifier_subgoal_id),
            "subgoal_id": verifier_subgoal_id,
            "status": "not_run" if verifier_subgoal_id else "disabled",
            "checks": 0,
            "hz": verifier_hz,
            "step_interval": None,
            "last_report": None,
            "early_stop_enabled": bool(verifier_early_stop),
        }
        verify_interval = None
        verifier_active = bool(verifier_subgoal_id)
        if verifier_subgoal_id:
            verify_interval = max(1, round(10.0 / max(0.1, verifier_hz)))
            local_verifier["step_interval"] = verify_interval

        first_satisfied_step: int | None = None
        stop_reason = "max_steps"
        t_run_start = time.time()
        log_event(
            "skill.run.start",
            skill_id=skill_id,
            max_steps=max_steps,
            subgoal_id=verifier_subgoal_id,
            verifier_hz=verifier_hz,
            verifier_early_stop=bool(verifier_early_stop),
            verifier_settle_steps=int(verifier_settle_steps),
        )

        for _ in range(max_steps):
            steps += 1

            if (
                verifier_early_stop
                and first_satisfied_step is not None
                and steps - first_satisfied_step > verifier_settle_steps
            ):
                success = True
                stop_reason = "cv_early_stop_after_settle"
                break

            obs_dict = preprocess_observation(self._last_obs)
            obs_t = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                     for k, v in obs_dict.items()}
            obs_t = loaded.env_pre(obs_t)
            obs_t = loaded.policy_pre(obs_t)

            with torch.inference_mode():
                action = loaded.policy.select_action(obs_t)
            action = loaded.policy_post(action)

            if isinstance(action, torch.Tensor):
                action_np = action.cpu().numpy()
            else:
                action_np = np.asarray(action)

            obs, reward, terminated, _truncated, _info = self._env.step(action_np)
            self._last_obs = obs

            # Render once; reuse for both display and video recording
            if display is not None or self._record_video:
                try:
                    import cv2  # noqa: PLC0415
                    frame = self._env.envs[0].render()
                    self._record_frame(frame)
                    if display is not None:
                        display.push(window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                except Exception:
                    display = None

            r = float(reward[0]) if hasattr(reward, "__len__") else float(reward)
            t = bool(terminated[0]) if hasattr(terminated, "__len__") else bool(terminated)
            last_reward = r
            if r > 0 or t:
                success = True
                stop_reason = "env_reward" if r > 0 else "env_terminated"
                self._episode_done = True
                break

            if (
                verifier_active
                and verifier_subgoal_id
                and verify_interval is not None
                and steps >= verifier_start_step
                and steps % verify_interval == 0
            ):
                try:
                    frame = self._env.envs[0].render()
                    self._record_frame(frame)
                    yolo = _run_yolo_detection(frame, conf=verifier_conf)
                    if yolo.get("status") != "ok":
                        local_verifier["checks"] = int(local_verifier["checks"]) + 1
                        local_verifier["status"] = "unavailable"
                        local_verifier["last_report"] = {
                            "subgoal_id": verifier_subgoal_id,
                            "status": "unavailable",
                            "satisfied": None,
                            "reason": "Local YOLO detector is unavailable; continuing rollout without CV early stop.",
                            "detector": {
                                key: value for key, value in yolo.items() if key != "detections"
                            },
                            "step": steps,
                        }
                        verifier_active = False
                        continue
                    report = _verify_subgoal_from_detections(
                        verifier_subgoal_id,
                        list(yolo.get("detections", [])),
                    )
                    report["detector"] = {
                        key: value for key, value in yolo.items() if key != "detections"
                    }
                    report["step"] = steps
                    local_verifier["checks"] = int(local_verifier["checks"]) + 1
                    local_verifier["status"] = report.get("status")
                    local_verifier["last_report"] = report
                    if report.get("satisfied") is True:
                        local_verifier["stop_reason"] = (
                            "cv_subgoal_satisfied"
                            if verifier_early_stop
                            else "cv_subgoal_satisfied_continue_rollout"
                        )
                        if first_satisfied_step is None:
                            first_satisfied_step = steps
                            log_event(
                                "skill.cv_first_satisfied",
                                skill_id=skill_id,
                                subgoal_id=verifier_subgoal_id,
                                step=steps,
                            )
                except Exception as exc:
                    local_verifier["status"] = "error"
                    local_verifier["last_report"] = {
                        "subgoal_id": verifier_subgoal_id,
                        "status": "error",
                        "satisfied": None,
                        "reason": str(exc),
                        "step": steps,
                    }
                    # Keep the rollout going; a verifier error should not kill control.

        elapsed_ms = int((time.time() - t_run_start) * 1000)
        frames_after_first_satisfied = (
            steps - first_satisfied_step if first_satisfied_step is not None else None
        )
        log_event(
            "skill.run.end",
            skill_id=skill_id,
            subgoal_id=verifier_subgoal_id,
            steps=steps,
            success=success,
            final_reward=last_reward,
            stop_reason=stop_reason,
            first_satisfied_step=first_satisfied_step,
            frames_after_first_satisfied=frames_after_first_satisfied,
            verifier_early_stop=bool(verifier_early_stop),
            elapsed_ms=elapsed_ms,
        )
        return {
            "skill_id": skill_id,
            "steps": steps,
            "success": success,
            "final_reward": last_reward,
            "local_verifier": local_verifier,
        }

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None


# ── Tools ────────────────────────────────────────────────────────────────────


class LiberoPerceptionTool(Tool):
    """Task-aware semantic perception for the LIBERO mug/plate scene."""

    def __init__(self, workspace: Path):
        self._workspace = Path(workspace)
        self._frames_dir = self._workspace / ".roboclaw_tmp" / "libero" / "perception_frames"
        self._frames_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "libero_perception"

    @property
    def description(self) -> str:
        return (
            "Analyze the current LIBERO mug/plate scene as task-aware semantic perception. "
            "Returns the rendered RGB image, known scene objects, target relations, and "
            "8-d proprioception for the LIBERO perception-plan-manipulation pipeline."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["analyze_scene"],
                    "description": "The LIBERO perception action to perform.",
                },
                "task_id": {
                    "type": "integer",
                    "default": _DEFAULT_TASK_ID,
                    "description": "LIBERO task_id (only used on first call to boot env).",
                },
                "reset": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Reset env only to start the first fresh episode. Mid-task reset "
                        "requests are ignored to preserve completed progress."
                    ),
                },
                "seed": {
                    "type": "integer",
                    "default": 0,
                    "description": "Reset seed (only when reset=true).",
                },
                "run_yolo": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Run the optional local YOLO detector and return detected_objects. "
                        "Leave false for Cosmos-controlled demos."
                    ),
                },
                "yolo_conf": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.25,
                    "description": "Confidence threshold for optional YOLO detections.",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        task_id: int = _DEFAULT_TASK_ID,
        reset: bool = False,
        seed: int = 0,
        run_yolo: bool = False,
        yolo_conf: float = 0.25,
        **_: Any,
    ) -> str | ToolResult:
        if action != "analyze_scene":
            return "Error: Unsupported libero_perception action."

        t_start = time.time()
        mgr = LiberoEnvManager.get()
        mgr.set_stage("Perception")
        reset_ignored = bool(reset and mgr._last_obs is not None)
        error = await _ensure_libero_env(
            mgr,
            task_id=task_id,
            episode_length=250,
            reset=reset,
            seed=seed,
            protect_existing_episode=True,
        )
        if error:
            return error

        t_render_start = time.time()
        _frame, image_path, error = await _render_libero_frame(mgr, self._frames_dir, "perception")
        render_ms = int((time.time() - t_render_start) * 1000)
        if error:
            return error
        yolo_ms = 0
        yolo_result = {
            "enabled": False,
            "status": "disabled",
            "detections": [],
        }
        if run_yolo and image_path is not None:
            t_yolo_start = time.time()
            yolo_result = await asyncio.to_thread(
                _run_yolo_detection,
                image_path,
                conf=float(yolo_conf),
            )
            yolo_ms = int((time.time() - t_yolo_start) * 1000)

        payload = {
            "frame_id": f"libero_{int(time.time() * 1000)}",
            "image_path": str(image_path),
            "task": _DEFAULT_TASK,
            "task_id": task_id,
            "objects": [dict(obj) for obj in _LIBERO_OBJECTS],
            "detected_objects": yolo_result.get("detections", []),
            "detector": {
                key: value
                for key, value in yolo_result.items()
                if key != "detections"
            },
            "target_relations": [dict(rel) for rel in _LIBERO_TARGET_RELATIONS],
            "proprioception_8d": mgr.get_proprioception(),
            "proprioception_layout": list(_PROPRIO_LAYOUT),
            "reset_requested": bool(reset),
            "reset_applied": bool(reset and not reset_ignored),
            "reset_ignored": reset_ignored,
            "timing_ms": {
                "render": render_ms,
                "yolo": yolo_ms,
                "total_tool": int((time.time() - t_start) * 1000),
            },
            "notes": (
                "LIBERO v1 perception is task-aware and image-backed: use the attached RGB "
                "image and optional detected_objects list to visually confirm actual mug/plate "
                "positions. If a mug is visibly "
                "on its target plate, mark that sub-goal complete and move to the next planned "
                "sub-goal; do not retry the same skill just because the reward flag is uncertain. "
                "Mid-task reset requests are ignored to preserve completed progress."
            ),
        }
        content = (
            "LIBERO semantic scene analyzed. Use this perception JSON for libero_plan, "
            "and inspect the attached RGB image for visual verification.\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
        log_event(
            "tool.libero_perception",
            elapsed_ms=int((time.time() - t_start) * 1000),
            render_ms=render_ms,
            yolo_ms=yolo_ms,
            yolo_status=yolo_result.get("status"),
            num_detections=len(yolo_result.get("detections", [])),
            reset_applied=bool(reset and not reset_ignored),
        )
        return ToolResult(content=content, media=[str(image_path)] if image_path else [])


class LiberoPlanTool(Tool):
    """Structured planner for the task-specific LIBERO skill scheduler."""

    @property
    def name(self) -> str:
        return "libero_plan"

    @property
    def description(self) -> str:
        return (
            "Plan a LIBERO mug/plate task from semantic perception. Produces ordered "
            "sub-goals and the next skill action for the perception-plan-manipulation pipeline."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["plan_task"],
                    "description": "The LIBERO planning action to perform.",
                },
                "user_goal": {
                    "type": "string",
                    "description": "Natural-language user goal for the LIBERO scene.",
                },
                "perception_json": {
                    "type": "string",
                    "description": "JSON payload returned by libero_perception.",
                },
                "memory_preferences": {
                    "type": "string",
                    "default": "",
                    "description": (
                        "Optional personalized memory preferences relevant to planning, "
                        "for example preferred execution order or conservative verification policy."
                    ),
                },
                "planning_memory_context": {
                    "type": "string",
                    "default": "",
                    "description": (
                        "Optional structured PlanningMemoryContext JSON from personalized robotic memory. "
                        "Use this to enforce hands-off objects, preferred placements, and operational rules."
                    ),
                },
            },
            "required": ["action", "user_goal", "perception_json"],
        }

    async def execute(
        self,
        action: str,
        user_goal: str,
        perception_json: str,
        memory_preferences: str = "",
        planning_memory_context: str = "",
        **_: Any,
    ) -> str:
        if action != "plan_task":
            return "Error: Unsupported libero_plan action."
        if not perception_json.strip():
            return "Error: libero_plan requires perception_json."
        try:
            perception = json.loads(perception_json)
        except json.JSONDecodeError as exc:
            return f"Error: perception_json must be valid JSON: {exc}"

        t_plan_start = time.time()
        planning_text = f"{user_goal}\n{memory_preferences}"
        planning_context = _parse_planning_memory_context(planning_memory_context)
        subgoals = _libero_subgoals_for_goal(planning_text)
        subgoals, personalization_decisions = _filter_subgoals_with_memory(
            subgoals,
            planning_context,
        )
        plan_basis = json.dumps(
            {
                "goal": user_goal,
                "frame_id": perception.get("frame_id"),
                "object_ids": [obj.get("object_id") for obj in perception.get("objects", [])],
                "memory_context": planning_context,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        plan_id = f"libero_plan_{hashlib.sha1(plan_basis.encode('utf-8')).hexdigest()[:10]}"
        payload = {
            "plan_id": plan_id,
            "task": perception.get("task") or _DEFAULT_TASK,
            "task_id": perception.get("task_id", _DEFAULT_TASK_ID),
            "user_goal": user_goal,
            "memory_preferences_used": memory_preferences,
            "planning_memory_context_used": planning_context,
            "ordered_subgoals": subgoals,
            "personalization_decisions": personalization_decisions,
            "execution_policy": {
                "default_max_attempts_per_subgoal": 1,
                "cosmos_controller": True,
                "local_cv_verifier_enabled": False,
                "local_cv_verifier_policy": (
                    "Do not use YOLO/CV in Cosmos-controlled demos. Let Cosmos-Reason2 "
                    "use live RGB frames and execution trace for strategy switching."
                ),
                "llm_only_when_uncertain": True,
                "retry_requires_allow_retry_true": True,
                "retry_only_if": "the mug is visibly not moved or the scene is clearly unchanged/corrupted",
                "never_reset_mid_task": True,
                "reward_zero_policy": (
                    "A zero reward is not a visual failure. Treat status=needs_visual_verification "
                    "as pending until the attached image or follow-up perception image is inspected."
                ),
                "visual_success_rule": (
                    "Cosmos-Reason2 should decide from live RGB whether the target mug "
                    "appears on its target plate and whether to proceed to the next sub-goal."
                ),
            },
            "next_action": (
                {
                    "action": "execute_skill",
                    "subgoal_id": subgoals[0]["subgoal_id"],
                    "skill_id": subgoals[0]["skill_id"],
                    "max_steps": subgoals[0]["max_steps"],
                    "requires_verification": False,
                }
                if subgoals
                else {
                    "action": "no_op",
                    "reason": "all_candidate_subgoals_filtered_by_memory",
                }
            ),
            "verification_action": None,
        }
        log_event(
            "tool.libero_plan",
            elapsed_ms=int((time.time() - t_plan_start) * 1000),
            num_subgoals=len(subgoals),
            subgoals=[sg.get("subgoal_id") for sg in subgoals],
            has_memory_preferences=bool(memory_preferences.strip()),
            has_planning_memory_context=bool(planning_context),
        )
        try:
            LiberoEnvManager.get().set_plan_summary(
                "[" + ", ".join(sg.get("skill_id", "?") for sg in subgoals) + "]"
            )
        except Exception:
            pass
        return json.dumps(payload, ensure_ascii=False, indent=2)


class LiberoCosmosRouteTool(Tool):
    """Fast local VLM router for one planned LIBERO sub-goal."""

    def __init__(self, workspace: Path):
        self._workspace = Path(workspace)
        self._frames_dir = self._workspace / ".roboclaw_tmp" / "libero" / "cosmos_route_frames"
        self._frames_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "libero_cosmos_route"

    @property
    def description(self) -> str:
        return (
            "Ask the local Cosmos-Reason2 small VLM subagent to route one planned "
            "LIBERO sub-goal to a concrete skill from the current RGB scene. Use this "
            "between libero_plan and libero_manipulation."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["route_skill"],
                    "description": "The Cosmos routing action to perform.",
                },
                "plan_json": {
                    "type": "string",
                    "description": "Full JSON payload returned by libero_plan.",
                },
                "subgoal_id": {
                    "type": "string",
                    "description": "The planned sub-goal currently being routed.",
                },
                "intended_skill_id": {
                    "type": "string",
                    "default": "",
                    "description": "Planner's intended skill_id for this sub-goal.",
                },
                "allowed_skill_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": (
                        "Skills the router is allowed to choose. Usually planned skill ids "
                        "plus wait/recover."
                    ),
                },
                "planner_instruction": {
                    "type": "string",
                    "default": "",
                    "description": "Concise instruction from the main agent for this sub-goal.",
                },
                "task_id": {
                    "type": "integer",
                    "default": _DEFAULT_TASK_ID,
                    "description": "LIBERO task_id (only used on first call to boot env).",
                },
                "reset": {
                    "type": "boolean",
                    "default": False,
                    "description": "Reset requests are ignored once an episode exists.",
                },
                "seed": {
                    "type": "integer",
                    "default": 0,
                    "description": "Reset seed (only if no episode exists yet).",
                },
                "cosmos_api_base": {
                    "type": "string",
                    "default": "",
                    "description": "OpenAI-compatible Cosmos/vLLM base URL.",
                },
                "cosmos_model": {
                    "type": "string",
                    "default": "",
                    "description": "Cosmos model id served by local vLLM.",
                },
                "confidence_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.6,
                    "description": "Minimum confidence required to execute a learned skill.",
                },
                "allow_fallback": {
                    "type": "boolean",
                    "default": True,
                    "description": (
                        "If true, fall back to the planner skill when Cosmos is unavailable "
                        "or returns invalid JSON."
                    ),
                },
            },
            "required": ["action", "plan_json", "subgoal_id"],
        }

    async def execute(
        self,
        action: str,
        plan_json: str,
        subgoal_id: str,
        intended_skill_id: str = "",
        allowed_skill_ids: list[str] | None = None,
        planner_instruction: str = "",
        task_id: int = _DEFAULT_TASK_ID,
        reset: bool = False,
        seed: int = 0,
        cosmos_api_base: str = "",
        cosmos_model: str = "",
        confidence_threshold: float = 0.6,
        allow_fallback: bool = True,
        **_: Any,
    ) -> str | ToolResult:
        if action != "route_skill":
            return "Error: Unsupported libero_cosmos_route action."

        t_start = time.time()
        plan = _parse_plan_json(plan_json)
        subgoal = _subgoal_by_id(plan, subgoal_id)
        intended_skill_id = (
            intended_skill_id
            or str(subgoal.get("skill_id") or "")
            or _default_skill_for_subgoal(subgoal_id)
        )
        target = str(
            subgoal.get("target_container_id")
            or subgoal.get("target")
            or _SUBGOAL_VERIFY_TARGETS.get(subgoal_id, {}).get("target_class", "")
        )
        allowed = [str(v) for v in (allowed_skill_ids or []) if v]
        if not allowed:
            planned = [
                str(sg.get("skill_id"))
                for sg in plan.get("ordered_subgoals", []) or []
                if isinstance(sg, dict) and sg.get("skill_id")
            ]
            allowed = planned or list(_DEFAULT_SKILL_CKPTS)
            allowed.extend(["wait", "recover"])
        allowed = list(dict.fromkeys(allowed))

        mgr = LiberoEnvManager.get()
        mgr.set_stage(f"Cosmos route - {subgoal_id}")
        reset_ignored = bool(reset and mgr._last_obs is not None)
        error = await _ensure_libero_env(
            mgr,
            task_id=task_id,
            episode_length=250,
            reset=reset,
            seed=seed,
            protect_existing_episode=True,
        )
        if error:
            return error

        _frame, image_path, error = await _render_libero_frame(
            mgr,
            self._frames_dir,
            f"cosmos_route_{subgoal_id}",
        )
        if error or image_path is None:
            return error or "Error: failed to render LIBERO frame for Cosmos routing."

        threshold = _coerce_confidence(confidence_threshold, 0.6)
        api_base = cosmos_api_base or os.environ.get(
            "ROBOCLAW_COSMOS_API_BASE",
            "http://localhost:8000/v1",
        )
        model = cosmos_model or os.environ.get(
            "ROBOCLAW_COSMOS_MODEL",
            "nvidia/Cosmos-Reason2-2B",
        )
        api_key = os.environ.get("ROBOCLAW_COSMOS_API_KEY", "EMPTY")
        timeout_s = float(os.environ.get("ROBOCLAW_COSMOS_TIMEOUT", "20"))
        skill_meanings = {
            "skill_06": "white mug → right plate",
            "skill_07": "yellow mug → left plate",
            "wait": "pause, request fresh observation",
            "recover": "scene unsafe or inconsistent",
        }
        allowed_desc = ", ".join(
            f"{s}({skill_meanings.get(s, s)})" for s in allowed
        )
        subgoal_desc = subgoal.get("description") or subgoal_id
        user_text = (
            f"Subgoal: {subgoal_desc}\n"
            f"Intended: {intended_skill_id}\n"
            f"Allowed: {allowed_desc}\n"
            + (f"Planner note: {planner_instruction}\n" if planner_instruction else "")
        )

        status = "ok"
        fallback_used = False
        router_error = ""
        raw_decision: dict[str, Any] = {}
        try:
            raw_decision = await _call_cosmos_reason2_router(
                api_base=api_base,
                api_key=api_key,
                model=model,
                image_data_url=_image_path_to_data_url(image_path),
                user_text=user_text,
                max_tokens=128,
                temperature=0.0,
                timeout_s=timeout_s,
            )
        except Exception as exc:
            router_error = str(exc)

        if not raw_decision and allow_fallback:
            status = "fallback"
            fallback_used = True
            raw_decision = _fallback_route_decision(
                intended_skill_id=intended_skill_id,
                subgoal_id=subgoal_id,
                target=target,
                reason=f"Cosmos routing unavailable or invalid: {router_error or 'empty JSON'}",
            )
        elif not raw_decision:
            status = "error"
            raw_decision = {
                "skill": "wait",
                "skill_id": "wait",
                "subgoal_id": subgoal_id,
                "target": target,
                "confidence": 0.0,
                "should_execute": False,
                "reason": router_error or "Cosmos returned no valid JSON.",
            }

        skill_id = _normalize_router_skill(
            raw_decision.get("skill_id") or raw_decision.get("skill"),
            subgoal_id,
        )
        confidence = _coerce_confidence(raw_decision.get("confidence"), 0.0)
        invalid_skill = skill_id not in allowed
        if invalid_skill and allow_fallback:
            status = "fallback"
            fallback_used = True
            skill_id = intended_skill_id if intended_skill_id in allowed else "wait"
            confidence = 0.0
            raw_decision["reason"] = (
                f"Router chose disallowed skill; fell back to {skill_id}."
            )
        elif invalid_skill:
            status = "error"
            skill_id = "wait"
            confidence = 0.0
            raw_decision["reason"] = "Router chose a disallowed skill."

        should_execute = (
            skill_id in _DEFAULT_SKILL_CKPTS
            and confidence >= threshold
            and status != "error"
        )
        raw_should_execute = raw_decision.get("should_execute", should_execute)
        if isinstance(raw_should_execute, str):
            raw_should_execute = raw_should_execute.strip().lower() in {"1", "true", "yes"}
        decision = {
            "skill": skill_id,
            "skill_id": skill_id,
            "subgoal_id": str(raw_decision.get("subgoal_id") or subgoal_id),
            "target": str(raw_decision.get("target") or target),
            "confidence": confidence,
            "should_execute": bool(raw_should_execute) and should_execute,
            "reason": str(raw_decision.get("reason") or ""),
        }
        payload = {
            "action_type": "libero_cosmos_route",
            "status": status,
            "router_role": "fast_vlm_subagent",
            "model": model,
            "api_base": api_base,
            "image_path": str(image_path),
            "subgoal": subgoal or {"subgoal_id": subgoal_id},
            "intended_skill_id": intended_skill_id,
            "allowed_skill_ids": allowed,
            "confidence_threshold": threshold,
            "decision": decision,
            "raw_decision": raw_decision,
            "fallback_used": fallback_used,
            "error": router_error,
            "reset_requested": bool(reset),
            "reset_applied": bool(reset and not reset_ignored),
            "reset_ignored": reset_ignored,
            "timing_ms": {"total_tool": int((time.time() - t_start) * 1000)},
            "next_recommendation": (
                "call_libero_manipulation_with_decision_skill_id"
                if decision["should_execute"]
                else "refresh_perception_or_report_router_uncertainty"
            ),
        }
        log_event(
            "tool.libero_cosmos_route",
            status=status,
            skill_id=decision["skill_id"],
            subgoal_id=decision["subgoal_id"],
            confidence=decision["confidence"],
            should_execute=decision["should_execute"],
            fallback_used=fallback_used,
            elapsed_ms=payload["timing_ms"]["total_tool"],
        )
        content = (
            "Cosmos-Reason2 fast VLM skill-routing report. The main agent should "
            "execute only when decision.should_execute is true.\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
        return ToolResult(content=content, media=[str(image_path)])


class LiberoVerifyTool(Tool):
    """Local YOLO/CV verifier for LIBERO sub-goal completion."""

    def __init__(self, workspace: Path):
        self._workspace = Path(workspace)
        self._frames_dir = self._workspace / ".roboclaw_tmp" / "libero" / "verify_frames"
        self._frames_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "libero_verify"

    @property
    def description(self) -> str:
        return (
            "Verify a LIBERO sub-goal locally with YOLO detections and geometric CV rules. "
            "Use this after manipulation; call the LLM/VLM only when status is uncertain."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["verify_subgoal"],
                    "description": "The local verification action to perform.",
                },
                "subgoal_id": {
                    "type": "string",
                    "enum": list(_SUBGOAL_VERIFY_TARGETS.keys()),
                    "description": "Planner sub-goal id to verify.",
                },
                "task_id": {
                    "type": "integer",
                    "default": _DEFAULT_TASK_ID,
                    "description": "LIBERO task_id (only used on first call to boot env).",
                },
                "reset": {
                    "type": "boolean",
                    "default": False,
                    "description": "Reset requests are ignored once an episode exists.",
                },
                "seed": {
                    "type": "integer",
                    "default": 0,
                    "description": "Reset seed (only if no episode exists yet).",
                },
                "yolo_conf": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.25,
                    "description": "YOLO confidence threshold.",
                },
            },
            "required": ["action", "subgoal_id"],
        }

    async def execute(
        self,
        action: str,
        subgoal_id: str,
        task_id: int = _DEFAULT_TASK_ID,
        reset: bool = False,
        seed: int = 0,
        yolo_conf: float = 0.25,
        **_: Any,
    ) -> str | ToolResult:
        if action != "verify_subgoal":
            return "Error: Unsupported libero_verify action."

        t_start = time.time()
        mgr = LiberoEnvManager.get()
        mgr.set_stage("Verifying")
        reset_ignored = bool(reset and mgr._last_obs is not None)
        error = await _ensure_libero_env(
            mgr,
            task_id=task_id,
            episode_length=250,
            reset=reset,
            seed=seed,
            protect_existing_episode=True,
        )
        if error:
            return error

        t_render = time.time()
        _frame, image_path, error = await _render_libero_frame(mgr, self._frames_dir, "verify")
        render_ms = int((time.time() - t_render) * 1000)
        if error:
            return error

        t_yolo = time.time()
        yolo_result = await asyncio.to_thread(
            _run_yolo_detection,
            image_path,
            conf=float(yolo_conf),
        )
        yolo_ms = int((time.time() - t_yolo) * 1000)
        report = _verify_subgoal_from_detections(
            subgoal_id,
            list(yolo_result.get("detections", [])),
        )
        payload = {
            "action_type": "libero_verify",
            "subgoal_id": subgoal_id,
            "status": report.get("status"),
            "satisfied": report.get("satisfied"),
            "verification_report": report,
            "detected_objects": yolo_result.get("detections", []),
            "detector": {
                key: value for key, value in yolo_result.items() if key != "detections"
            },
            "image_path": str(image_path) if image_path else None,
            "task_id": task_id,
            "reset_requested": bool(reset),
            "reset_applied": bool(reset and not reset_ignored),
            "reset_ignored": reset_ignored,
            "timing_ms": {
                "render": render_ms,
                "yolo": yolo_ms,
                "total_tool": int((time.time() - t_start) * 1000),
            },
            "next_recommendation": (
                "continue_to_next_subgoal"
                if report.get("satisfied") is True
                else "ask_llm_or_use_rgb_image_if_uncertain"
                if report.get("status") == "uncertain"
                else "retry_only_if_scene_visually_failed"
            ),
        }
        content = (
            "LIBERO local CV verification report. Use this deterministic report first; "
            "fall back to LLM visual reasoning only when status is uncertain.\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
        log_event(
            "tool.libero_verify",
            subgoal_id=subgoal_id,
            elapsed_ms=int((time.time() - t_start) * 1000),
            render_ms=render_ms,
            yolo_ms=yolo_ms,
            status=report.get("status"),
            satisfied=report.get("satisfied"),
        )
        return ToolResult(content=content, media=[str(image_path)] if image_path else [])


class LiberoManipulationTool(Tool):
    """Execute LIBERO manipulation under Cosmos-Reason2 skill-routing control."""

    def __init__(self, workspace: Path):
        self._workspace = Path(workspace)
        self._frames_dir = self._workspace / ".roboclaw_tmp" / "libero" / "manipulation_frames"
        self._frames_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "libero_manipulation"

    @property
    def description(self) -> str:
        return (
            "Execute LIBERO manipulation. Prefer passing plan_json with "
            "use_cosmos_controller=true so the local Cosmos-Reason2 VLM controls "
            "skill routing and policy switching from live RGB observations."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["execute_skill"],
                    "description": "The LIBERO manipulation action to perform.",
                },
                "skill_id": {
                    "type": "string",
                    "enum": list(_DEFAULT_SKILL_CKPTS.keys()),
                    "description": (
                        "Legacy fixed-skill mode: which trained skill to execute. "
                        "Optional when use_cosmos_controller=true."
                    ),
                },
                "subgoal_id": {
                    "type": "string",
                    "description": (
                        "Legacy fixed-skill mode: planner sub-goal id associated with "
                        "this execution. Optional when use_cosmos_controller=true."
                    ),
                },
                "plan_json": {
                    "type": "string",
                    "default": "",
                    "description": (
                        "Full JSON payload returned by libero_plan. When provided with "
                        "use_cosmos_controller=true, Cosmos controls the manipulation loop."
                    ),
                },
                "previous_summary": {
                    "type": "string",
                    "default": "",
                    "description": (
                        "Short main-agent summary of previous perception, memory constraints, "
                        "and completed/failed attempts for Cosmos to condition on."
                    ),
                },
                "use_cosmos_controller": {
                    "type": "boolean",
                    "default": True,
                    "description": (
                        "If true and plan_json is provided, delegate skill routing and "
                        "strategy switching inside this tool to Cosmos-Reason2."
                    ),
                },
                "allowed_skill_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Allowed skill/controller actions; defaults to plan skills plus wait/recover.",
                },
                "cosmos_api_base": {
                    "type": "string",
                    "default": "",
                    "description": "OpenAI-compatible Cosmos/vLLM base URL.",
                },
                "cosmos_model": {
                    "type": "string",
                    "default": "",
                    "description": "Cosmos model id served by local vLLM.",
                },
                "cosmos_confidence_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.6,
                    "description": "Minimum Cosmos confidence required to execute a learned skill.",
                },
                "cosmos_allow_fallback": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "If true, fall back to the next planner skill if Cosmos is unavailable "
                        "or returns invalid JSON."
                    ),
                },
                "cosmos_max_decisions": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 60,
                    "default": 30,
                    "description": "Maximum high-level Cosmos routing decisions in one manipulation call.",
                },
                "cosmos_chunk_steps": {
                    "type": "integer",
                    "minimum": 10,
                    "maximum": 250,
                    "default": 10,
                    "description": (
                        "Max rollout steps between Cosmos perception decisions. LIBERO runs "
                        "at about 10Hz, so 10 steps is approximately 1Hz."
                    ),
                },
                "max_steps": {
                    "type": "integer",
                    "minimum": 10,
                    "maximum": 400,
                    "default": 150,
                    "description": "Max env steps to run before returning.",
                },
                "task_id": {
                    "type": "integer",
                    "default": _DEFAULT_TASK_ID,
                    "description": "LIBERO task_id (only used on first call to boot the env).",
                },
                "reset": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Reset env only before the first action of a fresh episode. Mid-task "
                        "reset requests are ignored to preserve completed progress."
                    ),
                },
                "seed": {
                    "type": "integer",
                    "default": 0,
                    "description": "Reset seed (only when reset=true).",
                },
                "show_window": {
                    "type": "boolean",
                    "default": False,
                    "description": "Open a live OpenCV window to watch the rollout.",
                },
                "allow_retry": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Set true only when visual verification clearly shows the previous "
                        "attempt failed. Duplicate sub-goal executions are skipped by default."
                    ),
                },
                "local_verify": {
                    "type": "boolean",
                    "default": True,
                    "description": (
                        "Run the local YOLO/CV verifier during rollout at verifier_hz. "
                        "By default this monitors completion without stopping the learned skill early."
                    ),
                },
                "early_stop_on_verify": {
                    "type": "boolean",
                    "default": True,
                    "description": (
                        "If true, end the skill once local CV verifies the sub-goal AND the "
                        "verifier_settle_steps tail has elapsed (lets the policy finish "
                        "release/retract before switching to the next skill)."
                    ),
                },
                "verifier_settle_steps": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 200,
                    "default": 30,
                    "description": (
                        "When early_stop_on_verify=true, keep running this many extra rollout "
                        "steps after CV first reports satisfied so the gripper can release and "
                        "the arm can retract before the next skill starts."
                    ),
                },
                "verifier_hz": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 10.0,
                    "default": 5.0,
                    "description": "Approximate local verifier frequency during rollout.",
                },
                "verifier_start_step": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 200,
                    "default": 10,
                    "description": "Do not run local verification before this rollout step.",
                },
                "yolo_conf": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.25,
                    "description": "YOLO confidence threshold for local verification.",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        skill_id: str = "",
        subgoal_id: str = "",
        plan_json: str = "",
        previous_summary: str = "",
        use_cosmos_controller: bool = True,
        allowed_skill_ids: list[str] | None = None,
        cosmos_api_base: str = "",
        cosmos_model: str = "",
        cosmos_confidence_threshold: float = 0.6,
        cosmos_allow_fallback: bool = False,
        cosmos_max_decisions: int = 30,
        cosmos_chunk_steps: int = 10,
        max_steps: int = 150,
        task_id: int = _DEFAULT_TASK_ID,
        reset: bool = False,
        seed: int = 0,
        show_window: bool = False,
        allow_retry: bool = False,
        local_verify: bool = True,
        early_stop_on_verify: bool = True,
        verifier_hz: float = 5.0,
        verifier_start_step: int = 10,
        verifier_settle_steps: int = 30,
        yolo_conf: float = 0.25,
        **_: Any,
    ) -> str | ToolResult:
        if action != "execute_skill":
            return "Error: Unsupported libero_manipulation action."
        if use_cosmos_controller and plan_json.strip():
            return await self._execute_cosmos_controlled_plan(
                plan_json=plan_json,
                previous_summary=previous_summary,
                allowed_skill_ids=allowed_skill_ids or [],
                task_id=task_id,
                reset=reset,
                seed=seed,
                show_window=show_window,
                local_verify=local_verify,
                early_stop_on_verify=early_stop_on_verify,
                verifier_hz=verifier_hz,
                verifier_start_step=verifier_start_step,
                verifier_settle_steps=verifier_settle_steps,
                yolo_conf=yolo_conf,
                cosmos_api_base=cosmos_api_base,
                cosmos_model=cosmos_model,
                cosmos_confidence_threshold=cosmos_confidence_threshold,
                cosmos_allow_fallback=cosmos_allow_fallback,
                cosmos_max_decisions=cosmos_max_decisions,
                cosmos_chunk_steps=cosmos_chunk_steps,
            )
        if not skill_id or not subgoal_id:
            return (
                "Error: libero_manipulation requires either plan_json with "
                "use_cosmos_controller=true, or both skill_id and subgoal_id for legacy mode."
            )

        mgr = LiberoEnvManager.get()
        mgr.set_stage(f"{skill_id} - {subgoal_id}")
        if mgr._episode_done:
            payload = {
                "action_type": "libero_skill",
                "status": "task_already_complete",
                "reason": "Episode reached env_reward/env_terminated; further skill calls are ignored to preserve the final scene state.",
                "skill_id": skill_id,
                "subgoal_id": subgoal_id,
                "next_recommendation": "summarize_results_and_stop",
            }
            log_event(
                "tool.libero_manipulation",
                skill_id=skill_id,
                subgoal_id=subgoal_id,
                status="task_already_complete",
                steps=0,
                success_flag=True,
                cv_satisfied=True,
                elapsed_ms=0,
                attempt_count=0,
                early_stop_on_verify=bool(early_stop_on_verify),
                verifier_settle_steps=int(verifier_settle_steps),
            )
            return json.dumps(payload, ensure_ascii=False, indent=2)
        reset_ignored = bool(reset and mgr._last_obs is not None)
        error = await _ensure_libero_env(
            mgr,
            task_id=task_id,
            episode_length=max(max_steps, 250),
            reset=reset,
            seed=seed,
            protect_existing_episode=True,
        )
        if error:
            return error

        attempts = getattr(mgr, "_pipeline_subgoal_attempts", None)
        if attempts is None:
            attempts = {}
            setattr(mgr, "_pipeline_subgoal_attempts", attempts)
        prior_attempts = int(attempts.get(subgoal_id, 0))
        if prior_attempts >= 1 and not allow_retry:
            _frame, keyframe_path, _error = await _render_libero_frame(
                mgr,
                self._frames_dir,
                f"{subgoal_id}_{skill_id}_duplicate_skipped",
            )
            payload = {
                "action_type": "libero_skill",
                "status": "skipped_duplicate",
                "reason": "subgoal_already_executed; duplicate execution skipped to preserve progress",
                "attempt_count": prior_attempts,
                "observed_effect": {
                    "skill_id": skill_id,
                    "subgoal_id": subgoal_id,
                    "keyframe_path": str(keyframe_path) if keyframe_path else None,
                    "reset_requested": bool(reset),
                    "reset_applied": False,
                    "reset_ignored": reset_ignored,
                },
                "next_recommendation": "verify_with_libero_perception_or_continue_to_next_subgoal",
                "verification_hint": (
                    "This sub-goal was already executed once. If the mug is visibly on the "
                    "target plate, mark it complete and continue; do not repeat the skill."
                ),
            }
            content = (
                "LIBERO manipulation duplicate execution skipped.\n"
                f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
            )
            log_event(
                "tool.libero_manipulation",
                skill_id=skill_id,
                subgoal_id=subgoal_id,
                status="skipped_duplicate",
                attempt_count=prior_attempts,
                elapsed_ms=0,
            )
            return ToolResult(content=content, media=[str(keyframe_path)] if keyframe_path else [])
        if prior_attempts >= 2:
            payload = {
                "action_type": "libero_skill",
                "status": "skipped_max_attempts",
                "reason": "subgoal already reached the maximum attempt count",
                "attempt_count": prior_attempts,
                "observed_effect": {
                    "skill_id": skill_id,
                    "subgoal_id": subgoal_id,
                    "reset_requested": bool(reset),
                    "reset_applied": False,
                    "reset_ignored": reset_ignored,
                },
                "next_recommendation": "continue_or_report_partial_result",
            }
            return json.dumps(payload, ensure_ascii=False, indent=2)

        t0 = time.time()
        try:
            summary = await asyncio.to_thread(
                mgr.run_skill,
                skill_id,
                max_steps,
                show_window,
                subgoal_id if local_verify else None,
                float(verifier_hz),
                float(yolo_conf),
                int(verifier_start_step),
                bool(early_stop_on_verify),
                int(verifier_settle_steps),
            )
        except Exception as exc:
            return f"Error running skill {skill_id}: {exc}"
        elapsed_ms = int((time.time() - t0) * 1000)
        attempts[subgoal_id] = prior_attempts + 1

        _frame, keyframe_path, _error = await _render_libero_frame(
            mgr,
            self._frames_dir,
            f"{subgoal_id}_{skill_id}",
        )
        local_verifier = summary.get("local_verifier") or {}
        cv_satisfied = (
            isinstance(local_verifier, dict)
            and isinstance(local_verifier.get("last_report"), dict)
            and local_verifier["last_report"].get("satisfied") is True
        )
        status = (
            "success"
            if bool(summary.get("success")) or cv_satisfied
            else "needs_visual_verification"
        )
        payload = {
            "action_type": "libero_skill",
            "status": status,
            "reason": (
                "local_cv_verifier_satisfied"
                if cv_satisfied
                else "skill_reported_success"
                if bool(summary.get("success"))
                else "reward_zero_visual_verification_required"
            ),
            "attempt_count": attempts[subgoal_id],
            "observed_effect": {
                "skill_id": skill_id,
                "subgoal_id": subgoal_id,
                "steps": int(summary.get("steps") or 0),
                "final_reward": float(summary.get("final_reward") or 0.0),
                "success_flag": bool(summary.get("success")),
                "local_verifier": local_verifier,
                "elapsed_ms": elapsed_ms,
                "task_id": task_id,
                "keyframe_path": str(keyframe_path) if keyframe_path else None,
                "reset_requested": bool(reset),
                "reset_applied": bool(reset and not reset_ignored),
                "reset_ignored": reset_ignored,
            },
            "next_recommendation": "verify_with_libero_perception",
            "verification_hint": (
                "Inspect the attached keyframe or the next libero_perception image. "
                "A zero reward is not enough to mark the sub-goal failed. "
                "If the target mug is visibly on the target plate, mark the sub-goal complete "
                "and continue to the next planned sub-goal instead of repeating this skill."
            ),
        }
        content = (
            "LIBERO manipulation execution report. Verify the sub-goal with "
            "libero_perception and the attached keyframe.\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
        log_event(
            "tool.libero_manipulation",
            skill_id=skill_id,
            subgoal_id=subgoal_id,
            status=status,
            steps=int(summary.get("steps") or 0),
            success_flag=bool(summary.get("success")),
            cv_satisfied=cv_satisfied,
            elapsed_ms=elapsed_ms,
            attempt_count=attempts[subgoal_id],
            early_stop_on_verify=bool(early_stop_on_verify),
            verifier_settle_steps=int(verifier_settle_steps),
        )
        return ToolResult(content=content, media=[str(keyframe_path)] if keyframe_path else [])

    async def _execute_cosmos_controlled_plan(
        self,
        *,
        plan_json: str,
        previous_summary: str,
        allowed_skill_ids: list[str],
        task_id: int,
        reset: bool,
        seed: int,
        show_window: bool,
        local_verify: bool,
        early_stop_on_verify: bool,
        verifier_hz: float,
        verifier_start_step: int,
        verifier_settle_steps: int,
        yolo_conf: float,
        cosmos_api_base: str,
        cosmos_model: str,
        cosmos_confidence_threshold: float,
        cosmos_allow_fallback: bool,
        cosmos_max_decisions: int,
        cosmos_chunk_steps: int,
    ) -> str | ToolResult:
        plan = _parse_plan_json(plan_json)
        # In Cosmos-controlled mode, Cosmos is the visual controller. Keep YOLO/CV
        # out of policy switching even if an older prompt passes local_verify=true.
        local_verify = False
        subgoals = [
            sg for sg in plan.get("ordered_subgoals", []) or []
            if isinstance(sg, dict) and sg.get("subgoal_id")
        ]
        if not subgoals:
            return "Error: Cosmos-controlled manipulation requires plan_json.ordered_subgoals."

        planned_skills = [
            str(sg.get("skill_id"))
            for sg in subgoals
            if sg.get("skill_id")
        ]
        allowed = [str(v) for v in allowed_skill_ids if v] or planned_skills
        allowed.extend(["wait", "recover"])
        allowed = list(dict.fromkeys(allowed))

        mgr = LiberoEnvManager.get()
        mgr.set_stage("Cosmos controller")
        reset_ignored = bool(reset and mgr._last_obs is not None)
        error = await _ensure_libero_env(
            mgr,
            task_id=task_id,
            episode_length=max(int(cosmos_chunk_steps), 250),
            reset=reset,
            seed=seed,
            protect_existing_episode=True,
        )
        if error:
            return error

        threshold = _coerce_confidence(cosmos_confidence_threshold, 0.6)
        api_base = cosmos_api_base or os.environ.get(
            "ROBOCLAW_COSMOS_API_BASE",
            "http://localhost:8000/v1",
        )
        model = cosmos_model or os.environ.get(
            "ROBOCLAW_COSMOS_MODEL",
            "nvidia/Cosmos-Reason2-2B",
        )
        api_key = os.environ.get("ROBOCLAW_COSMOS_API_KEY", "EMPTY")
        timeout_s = float(os.environ.get("ROBOCLAW_COSMOS_TIMEOUT", "20"))
        max_decisions = max(1, min(60, int(cosmos_max_decisions)))
        chunk_steps = max(10, min(250, int(cosmos_chunk_steps)))

        completed: set[str] = set()
        execution_trace: list[dict[str, Any]] = []
        media: list[str] = []
        stop_reason = "max_decisions"
        status = "partial"
        router_error = ""
        last_skill_id = ""
        t_start = time.time()

        for decision_idx in range(max_decisions):
            open_subgoals = [
                sg for sg in subgoals
                if str(sg.get("subgoal_id")) not in completed
            ]
            if not open_subgoals:
                stop_reason = "all_subgoals_completed"
                status = "success"
                break

            _frame, image_path, error = await _render_libero_frame(
                mgr,
                self._frames_dir,
                f"cosmos_controller_decision_{decision_idx + 1}",
            )
            if error or image_path is None:
                return error or "Error: failed to render LIBERO frame for Cosmos controller."
            media.append(str(image_path))
            controller_perception = _cosmos_controller_perception(
                image_path=image_path,
                plan=plan,
                completed=completed,
                execution_trace=execution_trace,
                mgr=mgr,
            )

            open_desc = ", ".join(
                f"{sg.get('subgoal_id')}→{sg.get('skill_id','?')}"
                for sg in open_subgoals
            )
            done_desc = ", ".join(sorted(completed)) or "none"
            task_str = plan.get("user_goal") or plan.get("task") or _DEFAULT_TASK
            user_text = (
                f"Task: {task_str}\n"
                f"Open: {open_desc}\n"
                f"Done: {done_desc}\n"
                + (f"Context: {previous_summary}\n" if previous_summary else "")
                + "Skills: skill_06=white→right, skill_07=yellow→left, wait, recover, done\n"
                + f"controller_perception: {json.dumps(controller_perception, ensure_ascii=False)}\n"
                + "Step 1: inspect the image — list any open subgoals whose target placement is "
                "already visually satisfied (mug resting on correct plate) in completed_subgoals.\n"
                "Step 2: pick the next skill for the first remaining open subgoal, or 'done' if all are satisfied.\n"
            )

            raw_decision: dict[str, Any] = {}
            router_status = "ok"
            try:
                raw_decision = await _call_cosmos_reason2_router(
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    image_data_url=_image_path_to_data_url(image_path),
                    user_text=user_text,
                    max_tokens=128,
                    temperature=0.0,
                    timeout_s=timeout_s,
                )
            except Exception as exc:
                router_error = str(exc)

            if not raw_decision and cosmos_allow_fallback:
                router_status = "fallback"
                next_subgoal = open_subgoals[0]
                raw_decision = _fallback_route_decision(
                    intended_skill_id=str(next_subgoal.get("skill_id") or ""),
                    subgoal_id=str(next_subgoal.get("subgoal_id") or ""),
                    target=str(next_subgoal.get("target_container_id") or ""),
                    reason=f"Cosmos controller unavailable or invalid: {router_error or 'empty JSON'}",
                )
            elif not raw_decision:
                stop_reason = "cosmos_unavailable_or_invalid"
                execution_trace.append({
                    "decision_index": decision_idx + 1,
                    "router_status": "error",
                    "image_path": str(image_path),
                    "error": router_error or "Cosmos returned no valid JSON.",
                })
                break

            completed.update(_completed_subgoals_from_decision(raw_decision))
            open_subgoals = [
                sg for sg in subgoals
                if str(sg.get("subgoal_id")) not in completed
            ]
            if not open_subgoals:
                stop_reason = "all_subgoals_completed_by_cosmos_perception"
                status = "success"
                execution_trace.append({
                    "decision_index": decision_idx + 1,
                    "router_status": router_status,
                    "image_path": str(image_path),
                    "decision": raw_decision,
                    "completed_subgoals": sorted(completed),
                })
                break

            skill_id = _normalize_router_skill(
                raw_decision.get("skill_id") or raw_decision.get("skill"),
                str(raw_decision.get("subgoal_id") or open_subgoals[0].get("subgoal_id")),
            )
            if skill_id == "done":
                stop_reason = "cosmos_reported_done_without_all_subgoals_completed"
                execution_trace.append({
                    "decision_index": decision_idx + 1,
                    "router_status": router_status,
                    "image_path": str(image_path),
                    "decision": raw_decision,
                    "completed_subgoals": sorted(completed),
                })
                break
            confidence = _coerce_confidence(raw_decision.get("confidence"), 0.0)
            selected_subgoal_id = str(raw_decision.get("subgoal_id") or "")
            if not selected_subgoal_id:
                for candidate in open_subgoals:
                    if candidate.get("skill_id") == skill_id:
                        selected_subgoal_id = str(candidate.get("subgoal_id"))
                        break
            if not selected_subgoal_id:
                selected_subgoal_id = str(open_subgoals[0].get("subgoal_id"))

            raw_should_execute = raw_decision.get("should_execute", True)
            if isinstance(raw_should_execute, str):
                raw_should_execute = raw_should_execute.strip().lower() in {"1", "true", "yes"}
            should_execute = (
                bool(raw_should_execute)
                and skill_id in _DEFAULT_SKILL_CKPTS
                and skill_id in allowed
                and confidence >= threshold
            )
            controller_decision = {
                "skill_id": skill_id,
                "subgoal_id": selected_subgoal_id,
                "target": str(raw_decision.get("target") or ""),
                "confidence": confidence,
                "should_execute": should_execute,
                "reason": str(raw_decision.get("reason") or ""),
            }

            if not should_execute:
                execution_trace.append({
                    "decision_index": decision_idx + 1,
                    "router_status": router_status,
                    "image_path": str(image_path),
                    "decision": controller_decision,
                    "raw_decision": raw_decision,
                })
                if skill_id in {"wait", "recover"}:
                    # Explicit pause: stop and let the outer loop decide.
                    stop_reason = "cosmos_requested_" + skill_id
                    break
                # Unknown/hallucinated skill or low confidence: skip this step, retry.
                logger.warning(
                    "Cosmos returned non-executable skill {!r} (confidence={:.2f}, allowed={}); skipping.",
                    skill_id, confidence, allowed,
                )
                continue

            mgr.set_stage(f"Cosmos -> {skill_id} / {selected_subgoal_id}")
            t_skill = time.time()
            try:
                summary = await asyncio.to_thread(
                    mgr.run_skill,
                    skill_id,
                    chunk_steps,
                    show_window,
                    selected_subgoal_id if local_verify else None,
                    float(verifier_hz),
                    float(yolo_conf),
                    int(verifier_start_step),
                    bool(early_stop_on_verify),
                    int(verifier_settle_steps),
                    bool(skill_id != last_skill_id),
                )
            except Exception as exc:
                stop_reason = "skill_execution_error"
                execution_trace.append({
                    "decision_index": decision_idx + 1,
                    "router_status": router_status,
                    "image_path": str(image_path),
                    "decision": controller_decision,
                    "error": str(exc),
                })
                break
            last_skill_id = skill_id

            _end_frame, keyframe_path, _error = await _render_libero_frame(
                mgr,
                self._frames_dir,
                f"cosmos_controller_{selected_subgoal_id}_{skill_id}",
            )
            if keyframe_path is not None:
                media.append(str(keyframe_path))

            # Visual completion check: ask Cosmos to verify from the post-skill frame.
            # Reward is ignored — only visual confirmation counts.
            cosmos_verified: set[str] = set()
            if keyframe_path is not None:
                verify_text = (
                    f"Task: {task_str}\n"
                    f"Just executed: {skill_id} for subgoal {selected_subgoal_id}\n"
                    f"Inspect the image. Which subgoals are NOW visually satisfied "
                    f"(mug resting on correct plate)?\n"
                    f"Subgoals: {open_desc}\n"
                )
                try:
                    verify_decision = await _call_cosmos_reason2_router(
                        api_base=api_base,
                        api_key=api_key,
                        model=model,
                        image_data_url=_image_path_to_data_url(keyframe_path),
                        user_text=verify_text,
                        max_tokens=128,
                        temperature=0.0,
                        timeout_s=timeout_s,
                    )
                    cosmos_verified = _completed_subgoals_from_decision(
                        verify_decision, selected_subgoal_id
                    )
                except Exception:
                    pass

            completed.update(cosmos_verified)

            execution_trace.append({
                "decision_index": decision_idx + 1,
                "router_status": router_status,
                "image_path": str(image_path),
                "decision": controller_decision,
                "raw_decision": raw_decision,
                "skill_summary": {
                    "skill_id": skill_id,
                    "subgoal_id": selected_subgoal_id,
                    "steps": int(summary.get("steps") or 0),
                    "final_reward": float(summary.get("final_reward") or 0.0),
                    "success_flag": bool(summary.get("success")),
                    "cosmos_verified": sorted(cosmos_verified),
                    "elapsed_ms": int((time.time() - t_skill) * 1000),
                    "keyframe_path": str(keyframe_path) if keyframe_path else None,
                },
            })

        else:
            open_subgoals = [
                sg for sg in subgoals
                if str(sg.get("subgoal_id")) not in completed
            ]
            if not open_subgoals:
                status = "success"
                stop_reason = "all_subgoals_completed"

        if status != "success":
            status = "success" if len(completed) == len(subgoals) else "partial"
            if len(completed) == len(subgoals):
                stop_reason = "all_subgoals_completed"

        payload = {
            "action_type": "libero_cosmos_manipulation",
            "status": status,
            "controller_role": "fast_vlm_manipulation_controller",
            "model": model,
            "api_base": api_base,
            "plan_id": plan.get("plan_id"),
            "task_id": task_id,
            "completed_subgoals": sorted(completed),
            "remaining_subgoals": [
                sg for sg in subgoals
                if str(sg.get("subgoal_id")) not in completed
            ],
            "execution_trace": execution_trace,
            "local_verify": False,
            "local_verifier_policy": "disabled; Cosmos-Reason2 controls switching from live RGB",
            "stop_reason": stop_reason,
            "confidence_threshold": threshold,
            "reset_requested": bool(reset),
            "reset_applied": bool(reset and not reset_ignored),
            "reset_ignored": reset_ignored,
            "timing_ms": {"total_tool": int((time.time() - t_start) * 1000)},
            "next_recommendation": (
                "summarize_results"
                if status == "success"
                else "refresh_perception_or_report_partial_result"
            ),
        }
        log_event(
            "tool.libero_cosmos_manipulation",
            status=status,
            completed_subgoals=sorted(completed),
            stop_reason=stop_reason,
            decisions=len(execution_trace),
            elapsed_ms=payload["timing_ms"]["total_tool"],
        )
        content = (
            "LIBERO manipulation was controlled by Cosmos-Reason2 from live RGB "
            "observations and the main-agent plan summary.\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
        return ToolResult(content=content, media=media[-4:])


class LiberoSkillTool(Tool):
    """Run a single learned IL skill in the persistent LIBERO env."""

    def __init__(self, workspace: Path):
        self._workspace = Path(workspace)
        self._frames_dir = self._workspace / ".roboclaw_tmp" / "libero" / "skill_frames"
        self._frames_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "libero_skill"

    @property
    def description(self) -> str:
        return (
            "Execute a learned imitation-learning manipulation skill in the LIBERO simulator. "
            "Each skill performs one primitive: skill_06 puts the white mug on the right plate; "
            "skill_07 puts the yellow mug on the left plate. "
            "The simulation state persists across calls — successive skills continue from where "
            "the previous one left off. After every skill, call libero_observe to verify progress."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "skill_id": {
                    "type": "string",
                    "enum": list(_DEFAULT_SKILL_CKPTS.keys()),
                    "description": "Which trained ACT skill to execute.",
                },
                "max_steps": {
                    "type": "integer",
                    "minimum": 10,
                    "maximum": 400,
                    "default": 150,
                    "description": "Max env steps to run before returning.",
                },
                "task_id": {
                    "type": "integer",
                    "default": _DEFAULT_TASK_ID,
                    "description": "LIBERO task_id (only used on first call to boot the env).",
                },
                "reset": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Request reset only before the first fresh episode. If an episode already "
                        "exists, reset requests are ignored to preserve progress."
                    ),
                },
                "seed": {
                    "type": "integer",
                    "default": 0,
                    "description": "Reset seed (only used when reset=true).",
                },
                "show_window": {
                    "type": "boolean",
                    "default": False,
                    "description": "Open a live OpenCV window to watch the robot (requires display).",
                },
            },
            "required": ["skill_id"],
        }

    async def execute(
        self,
        skill_id: str,
        max_steps: int = 150,
        task_id: int = _DEFAULT_TASK_ID,
        reset: bool = False,
        seed: int = 0,
        show_window: bool = False,
        **_: Any,
    ) -> str | ToolResult:
        mgr = LiberoEnvManager.get()
        if mgr._episode_done:
            return json.dumps({
                "action_type": "libero_skill",
                "status": "task_already_complete",
                "reason": "Episode reached env_reward/env_terminated; further skill calls are ignored to preserve the final scene state.",
                "skill_id": skill_id,
                "next_recommendation": "summarize_results_and_stop",
            }, ensure_ascii=False, indent=2)
        reset_ignored = bool(reset and mgr._last_obs is not None)
        error = await _ensure_libero_env(
            mgr,
            task_id=task_id,
            episode_length=max(max_steps, 250),
            reset=reset,
            seed=seed,
            protect_existing_episode=True,
        )
        if error:
            return error

        t0 = time.time()
        try:
            summary = await asyncio.to_thread(mgr.run_skill, skill_id, max_steps, show_window)
        except Exception as exc:
            return f"Error running skill {skill_id}: {exc}"
        elapsed_ms = int((time.time() - t0) * 1000)

        # Save a final keyframe so the LLM sees what state we ended in
        keyframe_path: Path | None = None
        try:
            frame = await asyncio.to_thread(mgr.render)
            ts = int(time.time() * 1000)
            keyframe_path = self._frames_dir / f"{skill_id}_end_{ts}.png"
            Image.fromarray(frame).save(keyframe_path)
        except Exception:
            keyframe_path = None

        payload = {
            **summary,
            "elapsed_ms": elapsed_ms,
            "task_id": task_id,
            "keyframe_path": str(keyframe_path) if keyframe_path else None,
            "reset_requested": bool(reset),
            "reset_applied": bool(reset and not reset_ignored),
            "reset_ignored": reset_ignored,
        }
        content = (
            f"Ran {skill_id} for {summary['steps']} steps. "
            f"success={summary['success']}, final_reward={summary['final_reward']:.3f}. "
            f"Final keyframe attached. Call libero_observe next to assess scene state.\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
        media = [str(keyframe_path)] if keyframe_path else []
        return ToolResult(content=content, media=media)


class LiberoObserveTool(Tool):
    """Render the current LIBERO scene + return 8-d proprioception."""

    def __init__(self, workspace: Path):
        self._workspace = Path(workspace)
        self._frames_dir = self._workspace / ".roboclaw_tmp" / "libero" / "observe_frames"
        self._frames_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "libero_observe"

    @property
    def description(self) -> str:
        return (
            "Render the current LIBERO simulation scene (RGB image) and return the robot's "
            "8-d proprioception state. Use this to assess whether the previous skill completed "
            "the desired sub-goal before deciding what to do next."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "default": _DEFAULT_TASK_ID,
                    "description": "LIBERO task_id (only used on first call to boot env).",
                },
                "reset": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Request reset only before the first fresh episode. If an episode already "
                        "exists, reset requests are ignored to preserve progress."
                    ),
                },
                "seed": {
                    "type": "integer",
                    "default": 0,
                    "description": "Reset seed (only when reset=true).",
                },
            },
        }

    async def execute(
        self,
        task_id: int = _DEFAULT_TASK_ID,
        reset: bool = False,
        seed: int = 0,
        **_: Any,
    ) -> str | ToolResult:
        mgr = LiberoEnvManager.get()
        reset_ignored = bool(reset and mgr._last_obs is not None)
        error = await _ensure_libero_env(
            mgr,
            task_id=task_id,
            episode_length=250,
            reset=reset,
            seed=seed,
            protect_existing_episode=True,
        )
        if error:
            return error

        try:
            frame = await asyncio.to_thread(mgr.render)
        except Exception as exc:
            return f"Error rendering LIBERO scene: {exc}"

        ts = int(time.time() * 1000)
        image_path = self._frames_dir / f"observe_{ts}.png"
        Image.fromarray(frame).save(image_path)

        proprio = mgr.get_proprioception()
        payload = {
            "image_path": str(image_path),
            "task_id": task_id,
            "proprioception_8d": proprio,
            "proprioception_layout": list(_PROPRIO_LAYOUT),
            "reset_requested": bool(reset),
            "reset_applied": bool(reset and not reset_ignored),
            "reset_ignored": reset_ignored,
        }
        content = (
            "Current LIBERO scene rendered (RGB attached). Inspect the image to determine "
            "whether the previous sub-goal was achieved (e.g. white mug on right plate, "
            "yellow mug on left plate).\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
        return ToolResult(content=content, media=[str(image_path)])
