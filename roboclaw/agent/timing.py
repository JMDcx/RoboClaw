"""Lightweight JSONL timing logger for the LIBERO pipeline demo.

Writes one JSON line per event to ``$ROBOCLAW_TIMING_LOG`` if set, otherwise
``runs/timing/<YYYYmmdd_HHMMSS>_<pid>.jsonl`` under the current working
directory. The log file path is resolved on the first call and reused for
the rest of the process.

Usage::

    from roboclaw.agent.timing import log_event, timed

    log_event("tool.libero_perception", elapsed_ms=42, yolo_ms=18)

    with timed("skill.run", skill_id="skill_06") as ctx:
        ...
        ctx["steps"] = 87  # extra fields merged into the end event

The timer context logs ``<event>.start`` on entry and ``<event>.end`` on
exit with ``elapsed_ms`` plus any fields added to the dict.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_lock = threading.Lock()
_path: Path | None = None
_disabled = False


def _resolve_path() -> Path | None:
    global _path, _disabled
    if _path is not None or _disabled:
        return _path
    with _lock:
        if _path is not None or _disabled:
            return _path
        env = os.environ.get("ROBOCLAW_TIMING_LOG", "").strip()
        if env:
            target = Path(env).expanduser()
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target = Path.cwd() / "runs" / "timing" / f"{stamp}_{os.getpid()}.jsonl"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            _disabled = True
            return None
        _path = target
        return _path


def log_event(event: str, **fields: Any) -> None:
    """Append one JSON line to the timing log. Never raises."""
    target = _resolve_path()
    if target is None:
        return
    record = {
        "ts_iso": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "ts_ns": time.time_ns(),
        "event": event,
        **fields,
    }
    try:
        line = json.dumps(record, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        line = json.dumps({"ts_ns": time.time_ns(), "event": event, "error": "unserializable_fields"})
    with _lock:
        with target.open("a", encoding="utf-8") as fp:
            fp.write(line + "\n")


@contextmanager
def timed(event: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Log ``<event>.start`` and ``<event>.end`` with elapsed_ms.

    Yields a mutable dict; fields added inside the ``with`` block are merged
    into the ``.end`` event (handy for emitting steps/status/etc).
    """
    extra: dict[str, Any] = {}
    log_event(f"{event}.start", **fields)
    t0 = time.time()
    try:
        yield extra
    finally:
        elapsed_ms = int((time.time() - t0) * 1000)
        log_event(f"{event}.end", elapsed_ms=elapsed_ms, **fields, **extra)


def current_log_path() -> Path | None:
    """Return the resolved log path, or None if disabled / not yet used."""
    return _resolve_path()
