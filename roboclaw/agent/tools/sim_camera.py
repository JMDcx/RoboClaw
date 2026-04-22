"""Tool for fetching the latest Isaac Lab camera frame."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from roboclaw.agent.tools.base import Tool, ToolResult
from roboclaw.sim_camera import IsaacSimCameraReader


class SimCameraTool(Tool):
    """Read the latest Isaac Lab RGB-D frame and make it available to the agent."""

    def __init__(self, workspace: Path):
        self._workspace = workspace
        self._reader = IsaacSimCameraReader(camera_name="head")
        self._cache_dir = self._workspace / ".roboclaw_tmp" / "sim_camera"

    @property
    def name(self) -> str:
        return "sim_camera"

    @property
    def description(self) -> str:
        return (
            "Read the latest Isaac Lab RGB-D frame from shared memory. "
            "Use this for the simulated front_camera/head camera."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["get_latest_frame"],
                    "description": "The sim camera action to perform.",
                },
                "camera_name": {
                    "type": "string",
                    "enum": ["head"],
                    "description": "Only the simulated head/front camera is supported.",
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
        if action != "get_latest_frame":
            return "Error: Unsupported sim_camera action."
        if camera_name != "head":
            return "Error: Only camera_name='head' is supported."

        try:
            frame, image_path, depth_path, depth_vis_path = self._reader.save_latest_rgbd_frame(self._cache_dir)
        except Exception as exc:
            return f"Error: {exc}"

        payload = {
            "camera_name": camera_name,
            "image_path": str(image_path),
            "depth_path": str(depth_path) if depth_path else None,
            "depth_visualization_path": str(depth_vis_path) if depth_vis_path else None,
            "timestamp_ms": frame.timestamp_ms,
            "width": frame.width,
            "height": frame.height,
            "encoding": frame.encoding,
            "has_depth": frame.depth_m is not None,
            "depth_shape": list(frame.depth_m.shape) if frame.depth_m is not None else None,
            "depth_min_m": float(frame.depth_m.min()) if frame.depth_m is not None else None,
            "depth_max_m": float(frame.depth_m.max()) if frame.depth_m is not None else None,
            "fresh": frame.fresh,
        }
        content = (
            "Loaded the latest Isaac Lab front camera RGB-D frame. "
            "The RGB image and depth visualization are attached for immediate visual reasoning.\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
        media = [str(image_path)]
        if depth_vis_path is not None:
            media.append(str(depth_vis_path))
        return ToolResult(content=content, media=media)
