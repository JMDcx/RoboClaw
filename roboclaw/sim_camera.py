"""Isaac Lab shared-memory camera reader."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from multiprocessing import shared_memory
try:
    from multiprocessing import resource_tracker
except ImportError:
    resource_tracker = None
from pathlib import Path

import numpy as np


def get_shm_name(camera_name: str) -> str:
    """Return the Isaac Lab shared-memory segment name for one camera."""
    return f"isaac_{camera_name}_image_shm"


class SimpleImageHeader(ctypes.LittleEndianStructure):
    """Mirror of unitree_sim_isaaclab.tools.shared_memory_utils.SimpleImageHeader."""

    _fields_ = [
        ("timestamp", ctypes.c_uint64),
        ("height", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("channels", ctypes.c_uint32),
        ("image_name", ctypes.c_char * 16),
        ("data_size", ctypes.c_uint32),
        ("encoding", ctypes.c_uint32),
        ("quality", ctypes.c_uint32),
        ("dtype_code", ctypes.c_uint32),
    ]


@dataclass(slots=True)
class SimCameraFrame:
    """Decoded frame metadata plus the image payload."""

    timestamp_ms: int
    width: int
    height: int
    encoding: str
    image_bgr: np.ndarray
    depth_m: np.ndarray | None
    fresh: bool


class IsaacSimCameraReader:
    """Read one RGB-D camera stream from Isaac Lab shared memory."""

    _ENCODING_MAP = {0: "raw_bgr", 1: "jpeg"}
    _DTYPE_MAP = {0: np.uint8, 1: np.uint16, 2: np.float32}

    def __init__(self, camera_name: str = "head") -> None:
        if camera_name != "head":
            raise ValueError("Only camera_name='head' is supported right now.")
        self.camera_name = camera_name
        self.shm_name = get_shm_name(camera_name)
        self.depth_shm_name = get_shm_name(f"{camera_name}_depth")
        self._shm: shared_memory.SharedMemory | None = None
        self._depth_shm: shared_memory.SharedMemory | None = None
        self._last_timestamp_ms: int = 0

    def read_latest_frame(self) -> SimCameraFrame:
        """Read and decode the latest RGB-D frame from shared memory."""
        shm = self._open_shm()
        header_size = ctypes.sizeof(SimpleImageHeader)
        header = SimpleImageHeader.from_buffer_copy(bytes(shm.buf[:header_size]))

        if not header.timestamp or not header.width or not header.height or not header.channels:
            raise RuntimeError("Shared memory exists but does not contain a valid frame yet.")
        if header.data_size <= 0:
            raise RuntimeError("Shared memory frame has an empty payload.")

        data_start = header_size
        data_end = data_start + header.data_size
        payload = bytes(shm.buf[data_start:data_end])
        if len(payload) != header.data_size:
            raise RuntimeError("Shared memory payload is truncated.")

        image = self._decode_image(header, payload)
        depth = self._read_depth_frame()
        fresh = header.timestamp > self._last_timestamp_ms
        self._last_timestamp_ms = max(self._last_timestamp_ms, int(header.timestamp))
        return SimCameraFrame(
            timestamp_ms=int(header.timestamp),
            width=int(header.width),
            height=int(header.height),
            encoding=self._ENCODING_MAP.get(int(header.encoding), f"unknown:{header.encoding}"),
            image_bgr=image,
            depth_m=depth,
            fresh=fresh,
        )

    def save_latest_frame(self, output_dir: Path) -> tuple[SimCameraFrame, Path]:
        """Read the latest RGB frame and save it as a PNG file."""
        cv2 = self._import_cv2()
        frame = self.read_latest_frame()
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / f"{self.camera_name}_{frame.timestamp_ms}.png"
        if not cv2.imwrite(str(image_path), frame.image_bgr):
            raise RuntimeError(f"Failed to save frame to {image_path}.")
        return frame, image_path

    def save_latest_rgbd_frame(self, output_dir: Path) -> tuple[SimCameraFrame, Path, Path | None, Path | None]:
        """Read the latest RGB-D frame and save RGB plus optional depth artifacts."""
        cv2 = self._import_cv2()
        frame = self.read_latest_frame()
        output_dir.mkdir(parents=True, exist_ok=True)

        image_path = output_dir / f"{self.camera_name}_{frame.timestamp_ms}.png"
        if not cv2.imwrite(str(image_path), frame.image_bgr):
            raise RuntimeError(f"Failed to save frame to {image_path}.")

        depth_path: Path | None = None
        depth_vis_path: Path | None = None
        if frame.depth_m is not None:
            depth_path = output_dir / f"{self.camera_name}_{frame.timestamp_ms}_depth.npy"
            np.save(depth_path, frame.depth_m)

            depth_vis = self._colorize_depth(frame.depth_m)
            depth_vis_path = output_dir / f"{self.camera_name}_{frame.timestamp_ms}_depth.png"
            if not cv2.imwrite(str(depth_vis_path), depth_vis):
                raise RuntimeError(f"Failed to save depth visualization to {depth_vis_path}.")

        return frame, image_path, depth_path, depth_vis_path

    def close(self) -> None:
        """Close the shared-memory handle."""
        if self._shm is not None:
            self._shm.close()
            self._shm = None
        if self._depth_shm is not None:
            self._depth_shm.close()
            self._depth_shm = None

    def _open_shm(self) -> shared_memory.SharedMemory:
        if self._shm is None:
            try:
                self._shm = self._attach_shared_memory(self.shm_name)
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Shared memory '{self.shm_name}' not found. "
                    "Make sure the Isaac Lab simulation is running and front_camera is enabled."
                ) from exc
        return self._shm

    def _open_depth_shm(self) -> shared_memory.SharedMemory:
        if self._depth_shm is None:
            try:
                self._depth_shm = self._attach_shared_memory(self.depth_shm_name)
            except FileNotFoundError:
                return None
        return self._depth_shm

    def _read_depth_frame(self) -> np.ndarray | None:
        shm = self._open_depth_shm()
        if shm is None:
            return None

        header_size = ctypes.sizeof(SimpleImageHeader)
        header = SimpleImageHeader.from_buffer_copy(bytes(shm.buf[:header_size]))
        if not header.timestamp or not header.width or not header.height or header.data_size <= 0:
            return None

        data_start = header_size
        data_end = data_start + header.data_size
        payload = bytes(shm.buf[data_start:data_end])
        if len(payload) != header.data_size:
            raise RuntimeError("Depth shared memory payload is truncated.")
        return self._decode_raw_array(header, payload)

    @staticmethod
    def _decode_image(header: SimpleImageHeader, payload: bytes) -> np.ndarray:
        if int(header.encoding) == 1:
            cv2 = IsaacSimCameraReader._import_cv2()
            encoded = np.frombuffer(payload, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("Failed to decode JPEG frame from shared memory.")
            return image

        image = np.frombuffer(payload, dtype=np.uint8)
        expected_size = int(header.height) * int(header.width) * int(header.channels)
        if image.size != expected_size:
            raise RuntimeError(
                f"Frame size mismatch: expected {expected_size} bytes, got {image.size}."
            )
        return image.reshape(int(header.height), int(header.width), int(header.channels))

    @classmethod
    def _decode_raw_array(cls, header: SimpleImageHeader, payload: bytes) -> np.ndarray:
        dtype = cls._DTYPE_MAP.get(int(header.dtype_code), np.uint8)
        array = np.frombuffer(payload, dtype=dtype)
        expected_size = int(header.height) * int(header.width) * int(header.channels)
        if array.size != expected_size:
            raise RuntimeError(
                f"Depth frame size mismatch: expected {expected_size} values, got {array.size}."
            )
        if int(header.channels) == 1:
            return array.reshape(int(header.height), int(header.width))
        return array.reshape(int(header.height), int(header.width), int(header.channels))

    @staticmethod
    def _colorize_depth(depth_m: np.ndarray) -> np.ndarray:
        cv2 = IsaacSimCameraReader._import_cv2()
        depth = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        nonzero = depth[depth > 0]
        if nonzero.size == 0:
            normalized = np.zeros(depth.shape, dtype=np.uint8)
        else:
            near = float(np.percentile(nonzero, 5))
            far = float(np.percentile(nonzero, 95))
            if far <= near:
                far = near + 1e-3
            clipped = np.clip(depth, near, far)
            normalized = ((clipped - near) / (far - near) * 255.0).astype(np.uint8)
        return cv2.applyColorMap(255 - normalized, cv2.COLORMAP_JET)

    @staticmethod
    def _attach_shared_memory(name: str) -> shared_memory.SharedMemory:
        shm = shared_memory.SharedMemory(name=name)
        if resource_tracker is not None:
            try:
                resource_tracker.unregister(shm._name, "shared_memory")
            except Exception:
                pass
        return shm

    @staticmethod
    def _import_cv2():
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV (cv2) is required for Isaac Lab camera decoding. "
                "Install opencv-python-headless in the RoboClaw runtime."
            ) from exc
        return cv2
