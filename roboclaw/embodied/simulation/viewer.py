"""MuJoCo simulation web viewer with multipart HTTP streaming."""

from __future__ import annotations

from http import server
import importlib
import io
import threading
import time
from typing import Any

from roboclaw.embodied.simulation.mujoco_runtime import MujocoRuntime


class _StreamingHttpServer(server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        host: str,
        port: int,
        handler: type[server.BaseHTTPRequestHandler],
        viewer: "SimulationViewer",
    ) -> None:
        super().__init__((host, port), handler)
        self.viewer = viewer
        self.stop_event = threading.Event()


class SimulationViewer:
    """Small HTTP viewer that renders MuJoCo frames offscreen."""

    def __init__(
        self,
        runtime: MujocoRuntime,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        host: str = "0.0.0.0",
        port: int = 9878,
    ) -> None:
        self.runtime = runtime
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.host = host
        self.port = int(port)
        self._server: _StreamingHttpServer | None = None
        self._thread: threading.Thread | None = None
        self._renderer: Any | None = None
        self._lock = threading.RLock()
        self._frame_content_type = "image/jpeg"

    def _import_mujoco(self) -> Any:
        return self.runtime._import_mujoco()

    def _import_pil_image(self) -> Any | None:
        if not hasattr(self, "_pil_image_cache"):
            try:
                self._pil_image_cache = importlib.import_module("PIL.Image")
            except ModuleNotFoundError:
                self._pil_image_cache = None
        return self._pil_image_cache

    def _require_runtime(self) -> tuple[Any, Any]:
        model = self.runtime.model
        data = self.runtime.data
        if model is None or data is None:
            raise RuntimeError("MuJoCo runtime is not running.")
        return model, data

    def _ensure_renderer(self) -> Any:
        with self._lock:
            if self._renderer is not None:
                return self._renderer
            mujoco = self._import_mujoco()
            model, _ = self._require_runtime()
            self._renderer = mujoco.Renderer(model, self.height, self.width)
            return self._renderer

    def _copy_render_data(self, mujoco: Any, model: Any, data: Any) -> Any:
        copy_data = getattr(mujoco, "mj_copyData", None)
        mj_data_cls = getattr(mujoco, "MjData", None)
        if copy_data is None or mj_data_cls is None:
            return data
        try:
            snapshot = mj_data_cls(model)
            copy_data(snapshot, model, data)
            return snapshot
        except Exception:
            return data

    def _frame_dimensions(self, frame: Any) -> tuple[int, int]:
        shape = getattr(frame, "shape", None)
        if shape is not None and len(shape) >= 2:
            return int(shape[1]), int(shape[0])
        return self.width, self.height

    def _frame_bytes(self, frame: Any) -> bytes:
        if isinstance(frame, bytes):
            return frame
        if isinstance(frame, bytearray):
            return bytes(frame)
        if isinstance(frame, memoryview):
            return frame.tobytes()
        if hasattr(frame, "tobytes"):
            return frame.tobytes()
        if isinstance(frame, list):
            return bytes(
                channel
                for row in frame
                for pixel in row
                for channel in (pixel if isinstance(pixel, (list, tuple)) else [pixel])
            )
        raise TypeError("Rendered MuJoCo frame could not be converted to bytes.")

    def _encode_frame(self, frame: Any) -> bytes:
        width, height = self._frame_dimensions(frame)
        rgb = self._frame_bytes(frame)
        image = self._import_pil_image()
        if image is not None:
            buffer = io.BytesIO()
            image.frombytes("RGB", (width, height), rgb).save(buffer, format="JPEG")
            self._frame_content_type = "image/jpeg"
            return buffer.getvalue()

        self._frame_content_type = "image/x-portable-pixmap"
        header = f"P6\n{width} {height}\n255\n".encode("ascii")
        return header + rgb

    @property
    def frame_content_type(self) -> str:
        return self._frame_content_type

    @property
    def is_running(self) -> bool:
        return self._server is not None and self._thread is not None and self._thread.is_alive()

    def render_frame(self) -> bytes:
        with self._lock:
            mujoco = self._import_mujoco()
            model, data = self._require_runtime()
            renderer = self._ensure_renderer()
            render_data = self._copy_render_data(mujoco, model, data)
            renderer.update_scene(render_data)
            return self._encode_frame(renderer.render())

    def start(self) -> threading.Thread:
        if self.is_running and self._thread is not None:
            return self._thread

        self._ensure_renderer()
        outer = self

        class _Handler(server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                path = self.path.split("?", 1)[0]
                if path == "/":
                    self._serve_index()
                    return
                if path == "/stream":
                    self._serve_stream()
                    return
                if path == "/snapshot":
                    self._serve_snapshot()
                    return
                self.send_error(404, "not found")

            def log_message(self, format: str, *args: object) -> None:
                return None

            def _serve_index(self) -> None:
                body = (
                    "<!doctype html><html><head><meta charset='utf-8'>"
                    "<title>RoboClaw Simulation Viewer</title></head>"
                    "<body><h1>RoboClaw Simulation Viewer</h1>"
                    "<img src='/stream' alt='simulation stream'></body></html>"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _serve_snapshot(self) -> None:
                try:
                    frame = outer.render_frame()
                except Exception as exc:
                    self.send_error(500, str(exc))
                    return
                self.send_response(200)
                self.send_header("Content-Type", outer.frame_content_type)
                self.send_header("Content-Length", str(len(frame)))
                self.end_headers()
                self.wfile.write(frame)

            def _serve_stream(self) -> None:
                boundary = "frame"
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
                self.end_headers()
                try:
                    while not self.server.stop_event.is_set():
                        frame = outer.render_frame()
                        self.wfile.write(f"--{boundary}\r\n".encode("utf-8"))
                        self.wfile.write(f"Content-Type: {outer.frame_content_type}\r\n".encode("utf-8"))
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("utf-8"))
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                        time.sleep(1.0 / max(outer.fps, 1))
                except (BrokenPipeError, ConnectionResetError, RuntimeError):
                    return

        self._server = _StreamingHttpServer(self.host, self.port, _Handler, self)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        with self._lock:
            server_instance = self._server
            thread = self._thread
            renderer = self._renderer
            self._server = None
            self._thread = None
            self._renderer = None
        if server_instance is not None:
            server_instance.stop_event.set()
            server_instance.shutdown()
            server_instance.server_close()
        if thread is not None:
            thread.join(timeout=2.0)
        if renderer is not None and hasattr(renderer, "close"):
            renderer.close()
