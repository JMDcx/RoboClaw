from pathlib import Path

import numpy as np
import pytest

from roboclaw.agent.tools.base import ToolResult
from roboclaw.agent.tools.sim_camera import SimCameraTool
from roboclaw.sim_camera import SimCameraFrame


class _StubReader:
    def __init__(self, tmp_path: Path) -> None:
        self._image_path = tmp_path / "rgb.png"
        self._image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
        self._depth_path = tmp_path / "depth.npy"
        np.save(self._depth_path, np.ones((2, 2), dtype=np.float32))
        self._depth_vis_path = tmp_path / "depth.png"
        self._depth_vis_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    def save_latest_rgbd_frame(self, output_dir: Path):
        frame = SimCameraFrame(
            timestamp_ms=99,
            width=2,
            height=2,
            encoding="raw_bgr",
            image_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
            depth_m=np.ones((2, 2), dtype=np.float32),
            fresh=True,
        )
        return frame, self._image_path, self._depth_path, self._depth_vis_path


@pytest.mark.asyncio
async def test_sim_camera_tool_reports_rgbd_metadata(tmp_path: Path) -> None:
    tool = SimCameraTool(workspace=tmp_path)
    tool._reader = _StubReader(tmp_path)
    result = await tool.execute(action="get_latest_frame", camera_name="head")
    assert isinstance(result, ToolResult)
    assert '"has_depth": true' in result.content.lower()
    assert result.media == [str(tmp_path / "rgb.png"), str(tmp_path / "depth.png")]
