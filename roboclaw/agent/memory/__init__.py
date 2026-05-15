"""Personalized robotic memory system.

Re-exports MemoryStore and MemoryConsolidator from the legacy flat module
so existing imports continue to work now that memory/ is a package.
"""

import importlib
import importlib.util
import sys
from pathlib import Path

# Load the flat memory.py module under an aliased name to avoid shadowing.
_flat_path = Path(__file__).parent.parent / "memory.py"
_spec = importlib.util.spec_from_file_location("roboclaw.agent._memory_flat", _flat_path)
_flat = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _flat
_spec.loader.exec_module(_flat)

MemoryStore = _flat.MemoryStore
MemoryConsolidator = _flat.MemoryConsolidator
PersonalizedMemoryManager = importlib.import_module(
    "roboclaw.agent.memory.manager"
).PersonalizedMemoryManager

__all__ = ["MemoryStore", "MemoryConsolidator", "PersonalizedMemoryManager"]
