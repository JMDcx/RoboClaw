"""LLM-based intent classification for flexible user interaction."""

from __future__ import annotations

import json as _json
import re
from dataclasses import dataclass
from typing import Any


_SETUP_START_KEYWORDS = (
    "connect",
    "real robot",
    "setup",
    "onboard",
    "真实机器人",
    "真实的机器人",
    "机械臂",
    "机器人",
)
_SIM_KEYWORDS = (
    "simulation",
    "no robot",
    "try simulation",
    "virtual robot",
    "仿真",
    "没有机器人",
    "试试仿真",
    "虚拟机器人",
)
_SETUP_EDIT_KEYWORDS = (
    "camera",
    "sensor",
    "serial",
    "/dev/",
    "ros2",
    "deployment",
    "adapter",
    "installed",
    "replace",
)
_SERIAL_RE = re.compile(r"(/dev/[^\s,;]+)")


from roboclaw.utils.helpers import normalize_token as _normalize_token


@dataclass(frozen=True)
class UserIntent:
    """Structured intent extracted from a user message."""

    wants_setup: bool = False
    wants_simulation: bool = False
    wants_edit: bool = False
    robot_id: str | None = None
    connection_confirmed: bool | None = None
    wants_calibration: bool = False
    sensor_mount: str | None = None
    serial_path: str | None = None
    is_embodied: bool = False


class IntentClassifier:
    """Understands user messages via LLM, with keyword fallback."""

    def __init__(
        self,
        llm_caller=None,
        known_robots: tuple[str, ...] = (),
        robot_aliases: dict[str, tuple[str, ...]] | None = None,
    ):
        self._llm = llm_caller
        self._known_robots = tuple(dict.fromkeys(known_robots))
        self._robot_aliases = {
            robot_id: tuple(dict.fromkeys(alias.lower() for alias in aliases))
            for robot_id, aliases in (robot_aliases or {}).items()
        }

    async def classify(self, message: str, context: str = "") -> UserIntent:
        """Classify a user message into structured intent."""

        if self._llm is not None:
            try:
                return await self._llm_classify(message, context)
            except Exception:
                pass
        return self._keyword_fallback(message)

    async def _llm_classify(self, message: str, context: str) -> UserIntent:
        system = self._build_prompt(context)
        response = await self._llm(system, f"User message: {message}")
        return self._parse_response(response)

    def _build_prompt(self, context: str) -> str:
        robots = ", ".join(self._known_robots) if self._known_robots else "so101, piperx"
        return (
            "You are classifying a user's message in a robot control assistant.\n"
            f"Available robots: {robots}\n"
            f"Current context: {context or 'initial conversation'}\n\n"
            "Extract the user's intent as JSON with these fields:\n"
            "{\n"
            '  "wants_setup": true/false,\n'
            '  "wants_simulation": true/false,\n'
            '  "wants_edit": true/false,\n'
            '  "robot_id": "so101"/null,\n'
            '  "connection_confirmed": true/false/null,\n'
            '  "wants_calibration": true/false,\n'
            '  "sensor_mount": "wrist"/"overhead"/"external"/null,\n'
            '  "serial_path": "/dev/serial/..."/null,\n'
            '  "is_embodied": true/false\n'
            "}\n\n"
            "Respond with ONLY valid JSON, no explanation."
        )

    def _parse_response(self, response: str) -> UserIntent:
        from roboclaw.utils.helpers import strip_code_fences
        text = strip_code_fences(response)

        data = _json.loads(text)
        valid_fields = set(UserIntent.__dataclass_fields__)
        filtered = {key: value for key, value in data.items() if key in valid_fields}

        robot_id = filtered.get("robot_id")
        if robot_id is not None:
            filtered["robot_id"] = self._canonical_robot_id(robot_id)

        sensor_mount = filtered.get("sensor_mount")
        if sensor_mount is not None:
            filtered["sensor_mount"] = str(sensor_mount).strip().lower() or None

        serial_path = filtered.get("serial_path")
        if serial_path is not None:
            filtered["serial_path"] = str(serial_path).strip() or None

        connection = filtered.get("connection_confirmed")
        if isinstance(connection, str):
            lowered = connection.strip().lower()
            if lowered in {"true", "yes", "connected"}:
                filtered["connection_confirmed"] = True
            elif lowered in {"false", "no", "not connected"}:
                filtered["connection_confirmed"] = False
            else:
                filtered["connection_confirmed"] = None

        return UserIntent(**filtered)

    def _keyword_fallback(self, message: str) -> UserIntent:
        """Simple keyword matching when LLM is not available."""

        lower = message.lower()
        collapsed = " ".join(lower.split())

        robot_id = self._detect_robot_id(lower)
        wants_simulation = any(keyword in collapsed for keyword in _SIM_KEYWORDS)
        wants_setup = robot_id is not None or wants_simulation or any(keyword in lower for keyword in _SETUP_START_KEYWORDS)
        wants_edit = any(keyword in lower for keyword in _SETUP_EDIT_KEYWORDS)

        connection_confirmed: bool | None = None
        if any(
            token in lower
            for token in (
                "connected",
                "已连接",
                "连接好了",
                "连好了",
                "接好了",
                "接上了",
                "连上了",
                "都连好了",
                "已经接好了",
                "已经连接好了",
            )
        ):
            connection_confirmed = True
        elif (
            ("connect" in lower and any(token in lower for token in ("not", "no")))
            or any(
                token in lower
                for token in (
                    "没连",
                    "没有连接",
                    "未连接",
                    "还没连",
                    "没接好",
                    "还没有接好",
                    "还没有连接好",
                )
            )
        ):
            connection_confirmed = False

        wants_calibration = any(
            token in collapsed
            for token in (
                "calibrate",
                "calibration",
                "start calibration",
                "help me calibrate",
                "need calibration",
                "标定",
                "校准",
                "帮我标定",
                "开始标定",
                "帮我校准",
                "开始校准",
            )
        )

        sensor_mount: str | None = None
        if any(token in lower for token in ("camera", "sensor")):
            mounts: list[str] = []
            if "wrist" in lower:
                mounts.append("wrist")
            if "overhead" in lower:
                mounts.append("overhead")
            if "external" in lower:
                mounts.append("external")
            sensor_mount = mounts[0] if mounts else "wrist"

        match = _SERIAL_RE.search(message)
        serial_path = match.group(1) if match else None

        is_embodied = any(
            (
                wants_setup,
                wants_simulation,
                wants_edit,
                robot_id is not None,
                connection_confirmed is not None,
                wants_calibration,
                serial_path is not None,
                sensor_mount is not None,
            )
        )

        return UserIntent(
            wants_setup=wants_setup,
            wants_simulation=wants_simulation,
            wants_edit=wants_edit,
            robot_id=robot_id,
            connection_confirmed=connection_confirmed,
            wants_calibration=wants_calibration,
            sensor_mount=sensor_mount if wants_edit else None,
            serial_path=serial_path,
            is_embodied=is_embodied,
        )

    def _detect_robot_id(self, lower: str) -> str | None:
        for robot_id, aliases in self._robot_aliases.items():
            if any(alias in lower for alias in aliases):
                return robot_id
        for robot_id in self._known_robots:
            if robot_id.lower() in lower:
                return robot_id
        return None

    def _canonical_robot_id(self, value: Any) -> str | None:
        normalized = _normalize_token(str(value))
        if not normalized:
            return None

        for robot_id in self._known_robots:
            if _normalize_token(robot_id) == normalized:
                return robot_id
        for robot_id, aliases in self._robot_aliases.items():
            if _normalize_token(robot_id) == normalized:
                return robot_id
            for alias in aliases:
                if _normalize_token(alias) == normalized:
                    return robot_id
        return str(value).strip().lower() or None
