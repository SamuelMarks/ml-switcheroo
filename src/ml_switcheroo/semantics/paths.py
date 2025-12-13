"""
Path Resolution Utilities for Semantics.

Handles locating the 'semantics/' and 'snapshots/' directories
within the package or source tree.
"""

import sys
from pathlib import Path

# Use files API for package resources in Python 3.9+
if sys.version_info >= (3, 9):
  from importlib.resources import files
else:
  files = None


def resolve_semantics_dir() -> Path:
  """
  Locates the directory containing semantic JSON definitions.

  Prioritizes the local file system (relative to this file) to ensure
  tests and editable installs find the source of truth correctly.
  Falls back to package resources for installed distributions.

  Returns:
      Path: The absolute path to the 'semantics' directory.
  """
  # 1. Local Source Priority (Dev/Test/Editable)
  local_path = Path(__file__).parent
  # Simple check: does the main neural file exist here?
  if (local_path / "k_neural_net.json").exists():
    return local_path

  # 2. Installed Package Fallback
  if sys.version_info >= (3, 9) and files:
    try:
      # Note: wrapping in Path(str(...)) ensures compatibility issues
      # with early 3.9 implementations are smoothed over.
      return Path(str(files("ml_switcheroo.semantics")))
    except Exception:
      pass

  # Fallback to local path if discovery fails
  return local_path


def resolve_snapshots_dir() -> Path:
  """
  Locates the directory containing framework snapshots and mapping overlays.
  Defaults to the sibling 'snapshots' directory relative to 'semantics'.

  Returns:
      Path: The absolute path to the 'snapshots' directory.
  """
  return resolve_semantics_dir().parent / "snapshots"
