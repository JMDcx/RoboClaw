from __future__ import annotations

import importlib
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from roboclaw.embodied.perception import CameraManager, MjpegStreamServer, take_snapshot
from roboclaw.embodied.perception import streaming as streaming_module


class _FakeEncodedFrame:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def tobytes(self) -> bytes:
        return self._payload


class _FakeVideoCapture:
    def __init__(self, device_index: int, opened: bool = True, frame: object = object()) -> None:
        self.device_index = device_index
        self._opened = opened
        self._frame = frame
        self.properties: dict[int, int] = {}
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802
        return self._opened

    def read(self) -> tuple[bool, object]:
        return True, self._frame

    def set(self, prop: int, value: int) -> bool:
        self.properties[prop] = value
        return True

    def release(self) -> None:
        self.released = True
        self._opened = False


def _fake_cv2_module(
    *,
    opened: bool = True,
    frame: object = object(),
    encoded: bytes = b"fake-jpeg-bytes",
) -> tuple[SimpleNamespace, list[_FakeVideoCapture]]:
    captures: list[_FakeVideoCapture] = []

    def _video_capture(device_index: int) -> _FakeVideoCapture:
        capture = _FakeVideoCapture(device_index, opened=opened, frame=frame)
        captures.append(capture)
        return capture

    module = SimpleNamespace(
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_FPS=5,
        VideoCapture=_video_capture,
        imencode=lambda ext, source_frame: (True, _FakeEncodedFrame(encoded)),
    )
    return module, captures


def _patch_import_module(monkeypatch: pytest.MonkeyPatch, *, cv2_module: object | None) -> None:
    original_import_module = importlib.import_module

    def _import_module(name: str, package: str | None = None):
        if name == "cv2":
            if cv2_module is None:
                raise ModuleNotFoundError()
            return cv2_module
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _import_module)


def test_camera_manager_open_grab_close_and_is_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    cv2_module, captures = _fake_cv2_module(encoded=b"jpeg-payload")
    _patch_import_module(monkeypatch, cv2_module=cv2_module)

    camera = CameraManager("cam0", {"device_index": 2, "width": 800, "height": 600, "fps": 24})

    assert camera.is_opened is False

    camera.open()

    assert camera.is_opened is True
    assert len(captures) == 1
    assert captures[0].device_index == 2
    assert captures[0].properties == {3: 800, 4: 600, 5: 24}
    assert camera.grab_frame() == b"jpeg-payload"

    camera.close()

    assert camera.is_opened is False
    assert captures[0].released is True


def test_camera_manager_without_cv2_raises_module_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_import_module(monkeypatch, cv2_module=None)

    with pytest.raises(ModuleNotFoundError, match="opencv-python"):
        CameraManager("cam0").open()


def test_take_snapshot_saves_file_and_returns_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cv2_module, _ = _fake_cv2_module(encoded=b"snapshot-bytes")
    _patch_import_module(monkeypatch, cv2_module=cv2_module)
    camera = CameraManager("cam0")
    camera.open()

    result = take_snapshot(camera, output_dir=str(tmp_path))

    assert result.ok is True
    assert result.camera_id == "cam0"
    assert Path(result.path).exists()
    assert Path(result.path).read_bytes() == b"snapshot-bytes"
    assert Path(result.path).parent == tmp_path
    assert Path(result.path).name.startswith("snapshot_cam0_")
    assert datetime.fromisoformat(result.timestamp)


def test_take_snapshot_failure_returns_error(tmp_path: Path) -> None:
    camera = CameraManager("cam0")

    result = take_snapshot(camera, output_dir=str(tmp_path))

    assert result.ok is False
    assert result.path == ""
    assert result.message is not None
    assert "not opened" in result.message


def test_mjpeg_stream_server_start_and_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    camera = CameraManager("cam0")
    camera.grab_frame = lambda: b"jpeg-payload"  # type: ignore[method-assign]
    serve_forever_calls: list[str] = []
    shutdown_calls: list[str] = []
    server_close_calls: list[str] = []

    class _FakeHttpServer:
        def __init__(self, host: str, port: int, handler, server_camera: CameraManager) -> None:
            self.server_address = (host, 4321 if port == 0 else port)
            self.camera = server_camera
            self.stop_event = threading.Event()

        def serve_forever(self) -> None:
            serve_forever_calls.append("serve_forever")
            self.stop_event.wait(timeout=1.0)

        def shutdown(self) -> None:
            shutdown_calls.append("shutdown")

        def server_close(self) -> None:
            server_close_calls.append("server_close")

    monkeypatch.setattr(streaming_module, "_StreamingHttpServer", _FakeHttpServer)

    server = MjpegStreamServer(camera, host="127.0.0.1", port=0)

    thread = server.start()

    assert thread.is_alive() is True
    assert server.is_running is True
    assert server.port == 4321
    assert serve_forever_calls == ["serve_forever"]

    server.stop()

    assert server.is_running is False
    assert shutdown_calls == ["shutdown"]
    assert server_close_calls == ["server_close"]
