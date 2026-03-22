"""Utility functions for roboclaw."""

import re
from datetime import datetime
from pathlib import Path


def detect_image_mime(data: bytes) -> str | None:
    """Detect image MIME type from magic bytes, ignoring file extension."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def ensure_jpeg(data: bytes, mime: str | None) -> tuple[bytes, str]:
    """Convert non-JPEG images to JPEG for API compatibility.

    Some LLM providers reject PNG or other formats. Converting to JPEG
    also significantly reduces payload size, protecting conversation context.

    Returns (jpeg_bytes, "image/jpeg") on success, or (original, original_mime) on failure.
    """
    if mime == "image/jpeg" or (mime is None and data[:3] == b"\xff\xd8\xff"):
        return data, "image/jpeg"

    # Try PIL first
    try:
        import importlib
        pil_image = importlib.import_module("PIL.Image")
        import io
        img = pil_image.open(io.BytesIO(data)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue(), "image/jpeg"
    except Exception:
        pass

    # Try cv2 fallback
    try:
        import importlib
        import io
        cv2 = importlib.import_module("cv2")
        import numpy as np
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                return encoded.tobytes(), "image/jpeg"
    except Exception:
        pass

    return data, mime or "application/octet-stream"


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    """Current ISO timestamp."""
    return datetime.now().isoformat()


def normalize_token(value: str | None) -> str:
    """Normalize a string token for fuzzy matching (strip, lowercase, remove separators)."""
    return re.sub(r"[\s\-_]+", "", str(value or "").strip().lower())


_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*]')

def safe_filename(name: str) -> str:
    """Replace unsafe path characters with underscores."""
    return _UNSAFE_CHARS.sub("_", name).strip()


def strip_code_fences(text: str) -> str:
    """Strip leading/trailing markdown code fences (```json ... ```)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:\w+)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def split_message(content: str, max_len: int = 2000) -> list[str]:
    """
    Split content into chunks within max_len, preferring line breaks.

    Args:
        content: The text content to split.
        max_len: Maximum length per chunk (default 2000 for Discord compatibility).

    Returns:
        List of message chunks, each within max_len.
    """
    if not content:
        return []
    if len(content) <= max_len:
        return [content]
    chunks: list[str] = []
    while content:
        if len(content) <= max_len:
            chunks.append(content)
            break
        cut = content[:max_len]
        # Try to break at newline first, then space, then hard break
        pos = cut.rfind('\n')
        if pos <= 0:
            pos = cut.rfind(' ')
        if pos <= 0:
            pos = max_len
        chunks.append(content[:pos])
        content = content[pos:].lstrip()
    return chunks


def sync_workspace_templates(workspace: Path, silent: bool = False) -> list[str]:
    """Sync bundled templates to workspace. Only creates missing files."""
    from importlib.resources import files as pkg_files
    try:
        tpl = pkg_files("roboclaw") / "templates"
    except Exception:
        return []
    if not tpl.is_dir():
        return []

    added: list[str] = []

    def _write(src, dest: Path):
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(src.read_text(encoding="utf-8") if src else "", encoding="utf-8")
        added.append(str(dest.relative_to(workspace)))

    def _sync_tree(root, rel_prefix: Path = Path()) -> None:
        for item in root.iterdir():
            rel = rel_prefix / item.name
            if item.name == "__pycache__":
                continue
            if item.is_dir():
                _sync_tree(item, rel)
                continue
            if rel.name == "__init__.py":
                continue
            if rel.suffix == ".pyc":
                continue
            if rel == Path("memory/HISTORY.md"):
                continue
            _write(item, workspace / rel)

    _sync_tree(tpl)

    _write(None, workspace / "memory" / "HISTORY.md")
    (workspace / "skills").mkdir(exist_ok=True)

    if added and not silent:
        from rich.console import Console
        for name in added:
            Console().print(f"  [dim]Created {name}[/dim]")
    return added
