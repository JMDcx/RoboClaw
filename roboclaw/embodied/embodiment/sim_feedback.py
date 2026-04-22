"""Shared-memory readers for Isaac Lab sim feedback channels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any


@dataclass(slots=True)
class SimVerification:
    """Structured verification result for a sim grasp test."""

    object_height_before: float | None
    object_height_after: float | None
    height_delta: float | None
    reward_snapshot: list[float] | None
    success_boolean: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_height_before": self.object_height_before,
            "object_height_after": self.object_height_after,
            "height_delta": self.height_delta,
            "reward_snapshot": self.reward_snapshot,
            "success_boolean": self.success_boolean,
        }


class IsaacSimFeedbackReader:
    """Read sim state and reward snapshots from Isaac Lab shared memory."""

    def __init__(self) -> None:
        self._sim_state = _JsonSharedMemoryReader("isaac_sim_state")
        self._rewards = _JsonSharedMemoryReader("isaac_rewards")

    def read_sim_state(self) -> dict[str, Any] | None:
        return self._sim_state.read()

    def read_rewards(self) -> dict[str, Any] | None:
        return self._rewards.read()

    def verify_object_lift(self, *, min_height_m: float = 0.5) -> SimVerification:
        before = self.read_sim_state()
        before_height = _extract_object_height(before)
        rewards_before = self.read_rewards()
        reward_snapshot = _extract_rewards(rewards_before)
        after = self.read_sim_state()
        after_height = _extract_object_height(after)
        if before_height is None and after_height is not None:
            before_height = after_height
        height_delta = None
        if before_height is not None and after_height is not None:
            height_delta = float(after_height - before_height)
        success = bool(after_height is not None and after_height >= min_height_m)
        if reward_snapshot:
            success = success and max(reward_snapshot) >= 0.0
        return SimVerification(
            object_height_before=before_height,
            object_height_after=after_height,
            height_delta=height_delta,
            reward_snapshot=reward_snapshot,
            success_boolean=success,
        )


class _JsonSharedMemoryReader:
    """Read the simple JSON shared-memory envelope used by unitree_sim_isaaclab."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._shm: shared_memory.SharedMemory | None = None

    def read(self) -> dict[str, Any] | None:
        shm = self._open()
        if shm is None:
            return None
        try:
            data_len = int.from_bytes(shm.buf[4:8], "little")
            if data_len <= 0:
                return None
            payload = bytes(shm.buf[8:8 + data_len])
            return json.loads(payload.decode("utf-8"))
        except Exception:
            return None

    def _open(self) -> shared_memory.SharedMemory | None:
        if self._shm is None:
            try:
                self._shm = shared_memory.SharedMemory(name=self._name)
            except FileNotFoundError:
                return None
        return self._shm


def _extract_rewards(payload: dict[str, Any] | None) -> list[float] | None:
    if not payload:
        return None
    rewards = payload.get("rewards")
    if not isinstance(rewards, list):
        return None
    try:
        return [float(value) for value in rewards]
    except Exception:
        return None


def _extract_object_height(payload: Any) -> float | None:
    if payload is None:
        return None
    if isinstance(payload, dict):
        for key in ("object_height", "height", "z"):
            value = payload.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        for key in ("object", "Object", "root_pos_w", "position", "pos"):
            if key in payload:
                height = _extract_object_height(payload[key])
                if height is not None:
                    return height
        if "init_state" in payload:
            height = _extract_object_height(payload["init_state"])
            if height is not None:
                return height
        for key, value in payload.items():
            if "object" in str(key).lower():
                height = _extract_object_height(value)
                if height is not None:
                    return height
    elif isinstance(payload, list):
        if len(payload) >= 3 and all(isinstance(value, (int, float)) for value in payload[:3]):
            return float(payload[2])
        for item in payload:
            height = _extract_object_height(item)
            if height is not None:
                return height
    return None
