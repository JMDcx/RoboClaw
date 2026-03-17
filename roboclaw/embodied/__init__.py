"""High-level embodied entrypoints.

Keep the root namespace intentionally small. Most callers should import typed
contracts from leaf packages so importing ``roboclaw.embodied`` does not fan
out across the entire definition and execution stack.
"""

from roboclaw.embodied.catalog import (
    EmbodiedCatalog,
    build_catalog,
    build_default_catalog,
    inspect_workspace,
)
from roboclaw.embodied.definition.components.robots import SO101_ROBOT
from roboclaw.embodied.definition.components.sensors import RGB_CAMERA

__all__ = [
    "EmbodiedCatalog",
    "RGB_CAMERA",
    "SO101_ROBOT",
    "build_catalog",
    "build_default_catalog",
    "inspect_workspace",
]
