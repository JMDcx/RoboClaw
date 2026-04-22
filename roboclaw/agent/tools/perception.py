"""Perception tool for structured scene analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from roboclaw.agent.tools.base import Tool, ToolResult
from roboclaw.embodied.perception.service import PerceptionService


class PerceptionTool(Tool):
    """Analyze the current scene and return structured object detections."""

    def __init__(self, workspace: Path):
        self._service = PerceptionService(workspace=workspace)

    @property
    def name(self) -> str:
        return "perception"

    @property
    def description(self) -> str:
        return (
            "Run RGB-D perception on the latest simulated head camera frame and return "
            "structured object detections with temporal track IDs and grasp-oriented geometry."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["analyze_scene"],
                    "description": "The perception action to perform.",
                },
                "camera_name": {
                    "type": "string",
                    "enum": ["head"],
                    "description": "Only the simulated head camera is supported in v1.",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        camera_name: str = "head",
        **kwargs: Any,
    ) -> str | ToolResult:
        if action != "analyze_scene":
            return "Error: Unsupported perception action."
        if camera_name != "head":
            return "Error: Only camera_name='head' is supported."
        try:
            frame = self._service.analyze_scene(camera_name=camera_name)
        except Exception as exc:
            return f"Error: {exc}"
        payload = frame.to_dict()
        payload["stable_objects"] = [obj.to_dict() for obj in frame.objects if obj.stable]
        content = json.dumps(payload, ensure_ascii=False, indent=2)
        media = [frame.overlay_path] if frame.overlay_path else []
        return ToolResult(content=content, media=media)
