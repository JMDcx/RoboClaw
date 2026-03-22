import importlib
import threading
from types import SimpleNamespace

import pytest

import roboclaw.embodied.simulation.viewer as viewer_module
from roboclaw.embodied.simulation import SimulationViewer


class _FakeFrame:
    def __init__(self, width, height):
        self.shape = (height, width, 3)
        self._bytes = bytes([16, 32, 48]) * (width * height)

    def tobytes(self):
        return self._bytes


def _fake_mujoco():
    class _Data:
        def __init__(self, model):
            self.model = model
            self.copied_from = None

    class _Renderer:
        def __init__(self, model, height, width):
            self.model = model
            self.height = height
            self.width = width
            self.updated_with = None
            self.closed = False

        def update_scene(self, data):
            self.updated_with = data

        def render(self):
            return _FakeFrame(self.width, self.height)

        def close(self):
            self.closed = True

    def _copy_data(dest, model, src):
        dest.model = model
        dest.copied_from = src

    return SimpleNamespace(Renderer=_Renderer, MjData=_Data, mj_copyData=_copy_data)


def _patch_modules(monkeypatch: pytest.MonkeyPatch, mujoco_module):
    original_import_module = importlib.import_module

    def _import_module(name, package=None):
        if name == "mujoco":
            return mujoco_module
        if name == "PIL.Image":
            raise ModuleNotFoundError()
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _import_module)


def _runtime():
    mujoco_mod = _fake_mujoco()
    m, d = object(), object()
    return SimpleNamespace(
        _model=m,
        _data=d,
        model=m,
        data=d,
        _mujoco=mujoco_mod,
        _import_mujoco=lambda: mujoco_mod,
    )


def test_simulation_viewer_instantiation():
    viewer = SimulationViewer(_runtime())

    assert viewer.runtime._model is not None
    assert viewer.width == 640
    assert viewer.height == 480
    assert viewer.fps == 30
    assert viewer.is_running is False


def test_render_frame_returns_bytes(monkeypatch: pytest.MonkeyPatch):
    _patch_modules(monkeypatch, _fake_mujoco())
    viewer = SimulationViewer(_runtime(), width=8, height=6)

    frame = viewer.render_frame()

    assert isinstance(frame, bytes)
    assert frame.startswith(b"P6\n8 6\n255\n")


def test_start_stop_lifecycle(monkeypatch: pytest.MonkeyPatch):
    _patch_modules(monkeypatch, _fake_mujoco())

    class _FakeServer:
        def __init__(self, host, port, handler, viewer):
            self.host = host
            self.port = port
            self.handler = handler
            self.viewer = viewer
            self.stop_event = threading.Event()
            self.server_address = (host, 19878 if port == 0 else port)

        def serve_forever(self):
            self.stop_event.wait(timeout=1.0)

        def shutdown(self):
            self.stop_event.set()

        def server_close(self):
            return None

    monkeypatch.setattr(viewer_module, "_StreamingHttpServer", _FakeServer)
    viewer = SimulationViewer(_runtime(), port=0)

    thread = viewer.start()
    renderer = viewer._renderer

    assert thread.is_alive()
    assert viewer.is_running is True
    assert viewer.port > 0

    viewer.stop()

    assert viewer.is_running is False
    assert renderer.closed is True
