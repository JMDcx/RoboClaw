"""Perception module — camera management, streaming, and snapshot for agent feedback."""

from roboclaw.embodied.perception.camera import CameraManager
from roboclaw.embodied.perception.snapshot import SnapshotResult, take_snapshot
from roboclaw.embodied.perception.streaming import MjpegStreamServer

__all__ = ["CameraManager", "MjpegStreamServer", "SnapshotResult", "take_snapshot"]
