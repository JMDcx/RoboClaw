"""Minimal MJPEG debug streaming server."""

from __future__ import annotations

from http import server
import threading
import time

from roboclaw.embodied.perception.camera import CameraManager


class _StreamingHttpServer(server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        host: str,
        port: int,
        handler: type[server.BaseHTTPRequestHandler],
        camera: CameraManager,
    ) -> None:
        super().__init__((host, port), handler)
        self.camera = camera
        self.stop_event = threading.Event()


class MjpegStreamServer:
    """Small HTTP MJPEG server for local debugging."""

    def __init__(self, camera: CameraManager, host: str = "0.0.0.0", port: int = 9877) -> None:
        self.camera = camera
        self.host = host
        self.port = port
        self._server: _StreamingHttpServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._server is not None and self._thread is not None and self._thread.is_alive()

    def start(self) -> threading.Thread:
        if self.is_running and self._thread is not None:
            return self._thread

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
                self.send_error(404, "not found")

            def log_message(self, format: str, *args: object) -> None:
                return None

            def _serve_index(self) -> None:
                body = (
                    "<!doctype html><html><head><meta charset='utf-8'>"
                    "<title>RoboClaw Camera Stream</title></head>"
                    "<body><h1>RoboClaw Camera Stream</h1><img src='/stream' alt='camera stream'></body></html>"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _serve_stream(self) -> None:
                boundary = "frame"
                fps = max(int(outer.camera.config.get("fps", 30)), 1)
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
                self.end_headers()
                try:
                    while not self.server.stop_event.is_set():
                        frame = outer.camera.grab_frame()
                        self.wfile.write(f"--{boundary}\r\n".encode("utf-8"))
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("utf-8"))
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                        time.sleep(1.0 / fps)
                except (BrokenPipeError, ConnectionResetError):
                    return

        self._server = _StreamingHttpServer(self.host, self.port, _Handler, self.camera)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.stop_event.set()
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None
