from importlib import resources
from pathlib import Path

from roboclaw.config import paths as config_paths
from roboclaw.utils.helpers import sync_workspace_templates


def test_sync_workspace_templates_skips_python_cache_files(monkeypatch, tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    templates_root = package_root / "templates"
    templates_root.mkdir(parents=True)
    (templates_root / "AGENTS.md").write_text("agent template", encoding="utf-8")

    pycache_dir = templates_root / "__pycache__"
    pycache_dir.mkdir()
    (pycache_dir / "__init__.cpython-311.pyc").write_bytes(b"\xa7\r\r\nbinary")

    monkeypatch.setattr(resources, "files", lambda _package: package_root)

    workspace = tmp_path / "workspace"
    added = sync_workspace_templates(workspace, silent=True)

    assert "AGENTS.md" in added
    assert (workspace / "AGENTS.md").read_text(encoding="utf-8") == "agent template"
    assert not (workspace / "__pycache__").exists()


def test_ensure_robot_calibration_file_imports_legacy_cache_once(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config_paths, "get_config_path", lambda: tmp_path / "config.json")
    monkeypatch.setattr(config_paths, "LEGACY_CALIBRATION_ROOT", tmp_path / "home" / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots")

    legacy = (
        tmp_path
        / "home"
        / ".cache"
        / "huggingface"
        / "lerobot"
        / "calibration"
        / "robots"
        / "so_follower"
        / "so101_real.json"
    )
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        "{\"gripper\": {\"id\": 6, \"drive_mode\": 0, \"homing_offset\": 0, \"range_min\": 0, \"range_max\": 4095}}",
        encoding="utf-8",
    )

    canonical = config_paths.ensure_robot_calibration_file("so101", "so101_real")

    assert canonical == tmp_path / "calibration" / "so101" / "so101_real.json"
    assert canonical.read_text(encoding="utf-8") == legacy.read_text(encoding="utf-8")
