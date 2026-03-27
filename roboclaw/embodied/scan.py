"""Hardware scanning — detect serial ports and cameras."""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path


def _read_symlink_map(directory: str) -> dict[str, str]:
    """Read a directory of symlinks, return {resolved_target: symlink_path}."""
    d = Path(directory)
    if not d.exists():
        return {}
    result = {}
    for entry in d.iterdir():
        if entry.is_symlink():
            target = os.path.realpath(str(entry))
            result[target] = str(entry)
    return result


def scan_serial_ports() -> list[dict[str, str]]:
    """Scan serial devices, return list with by_path, by_id, dev."""
    from roboclaw.embodied.stub import is_stub_mode, stub_ports

    if is_stub_mode():
        return stub_ports()

    by_path = _read_symlink_map("/dev/serial/by-path")
    by_id = _read_symlink_map("/dev/serial/by-id")
    all_devs = set(by_path.keys()) | set(by_id.keys())
    ports = []
    for dev in sorted(all_devs):
        if not os.path.exists(dev):
            continue
        ports.append({
            "by_path": by_path.get(dev, ""),
            "by_id": by_id.get(dev, ""),
            "dev": dev,
        })
    return ports


def suppress_stderr() -> int:
    """Redirect stderr to /dev/null. Returns saved fd for restore_stderr."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    return saved


def restore_stderr(saved: int) -> None:
    """Restore stderr from saved fd."""
    os.dup2(saved, 2)
    os.close(saved)


def scan_cameras() -> list[dict[str, str | int]]:
    """Scan cameras, return list with by_path, by_id, dev, resolution."""
    from roboclaw.embodied.stub import is_stub_mode, stub_cameras

    if is_stub_mode():
        return stub_cameras()

    try:
        import cv2
    except ImportError:
        return []

    saved = suppress_stderr()
    try:
        by_path = _read_symlink_map("/dev/v4l/by-path")
        by_id = _read_symlink_map("/dev/v4l/by-id")
        return _probe_cameras(cv2, by_path, by_id)
    finally:
        restore_stderr(saved)


def capture_camera_frames(
    scanned_cameras: list[dict[str, str | int]], output_dir: str | Path,
) -> list[dict[str, str]]:
    """Capture one JPEG preview for each scanned camera."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for camera previews.") from exc

    previews: list[dict[str, str]] = []
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    saved = suppress_stderr()
    try:
        for index, camera in enumerate(scanned_cameras):
            preview = _capture_camera_frame(cv2, camera, target_dir, index)
            if preview is not None:
                previews.append(preview)
        return previews
    finally:
        restore_stderr(saved)


def _probe_cameras(cv2, by_path: dict, by_id: dict) -> list[dict[str, str | int]]:
    """Try opening each /dev/videoN, return those that work."""
    cameras = []
    for dev in sorted(glob.glob("/dev/video*")):
        m = re.match(r"/dev/video(\d+)$", dev)
        if not m:
            continue
        info = _try_open_camera(cv2, int(m.group(1)), dev, by_path, by_id)
        if info:
            cameras.append(info)
    return cameras


def _try_open_camera(cv2, index: int, dev: str, by_path: dict, by_id: dict) -> dict[str, str | int] | None:
    """Open a single camera by index, return info dict or None."""
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            return None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        real = os.path.realpath(dev)
        return {
            "by_path": by_path.get(real, ""),
            "by_id": by_id.get(real, ""),
            "dev": dev,
            "width": w,
            "height": h,
        }
    finally:
        cap.release()


def _capture_camera_frame(
    cv2, camera: dict[str, str | int], output_dir: Path, index: int,
) -> dict[str, str] | None:
    source = str(camera.get("by_path") or camera.get("by_id") or camera.get("dev") or "")
    label = source
    if not source:
        return None

    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        image_path = output_dir / f"{index:02d}_{Path(label).name}.jpg"
        if not cv2.imwrite(str(image_path), frame):
            raise RuntimeError(f"Failed to write camera preview to {image_path}")
        return {
            "camera": label,
            "image_path": str(image_path),
        }
    finally:
        cap.release()
