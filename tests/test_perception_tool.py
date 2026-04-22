import json
from pathlib import Path

import pytest

from roboclaw.agent.tools.base import ToolResult
from roboclaw.agent.tools.perception import PerceptionTool
from roboclaw.embodied.perception.schemas import DetectedObject, PerceptionFrame


class _StubService:
    def __init__(self, tmp_path: Path) -> None:
        self._overlay = tmp_path / "overlay.png"
        self._overlay.write_bytes(b"\x89PNG\r\n\x1a\n")

    def analyze_scene(self, camera_name: str = "head") -> PerceptionFrame:
        return PerceptionFrame(
            frame_id="head_1",
            timestamp_ms=1,
            camera_name=camera_name,
            image_path="/tmp/image.png",
            overlay_path=str(self._overlay),
            objects=[
                DetectedObject(
                    object_id="object_1",
                    raw_class_name="cup",
                    task_label="object",
                    track_id=1,
                    class_name="cup",
                    confidence=0.9,
                    bbox_xyxy=[1.0, 2.0, 3.0, 4.0],
                    mask_rle=None,
                    center_xy=[2.0, 3.0],
                    stable=True,
                    age=3,
                    visibility=1.0,
                    pickable=True,
                    container_candidate=False,
                )
            ],
        )


@pytest.mark.asyncio
async def test_perception_tool_returns_structured_json_and_overlay(tmp_path: Path) -> None:
    tool = PerceptionTool(workspace=tmp_path)
    tool._service = _StubService(tmp_path)
    result = await tool.execute(action="analyze_scene", camera_name="head")
    assert isinstance(result, ToolResult)
    payload = json.loads(result.content)
    assert payload["camera_name"] == "head"
    assert payload["objects"][0]["class_name"] == "cup"
    assert payload["stable_objects"][0]["track_id"] == 1
    assert result.media == [str(tmp_path / "overlay.png")]
