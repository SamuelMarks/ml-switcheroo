"""Path Resolution Utilities for Semantics.

Handles locating the 'semantics/' and 'snapshots/' directories
within the package or source tree.
"""

import os
import sys
from pathlib import Path

# Use files API for package resources in Python 3.9+
if sys.version_info >= (3, 9):
  from importlib.resources import files
else:
  files = None


def resolve_semantics_dir() -> Path:
  """Locate the directory containing semantic JSON definitions.

  Prioritizes the local file system (relative to this file) to ensure
  tests and editable installs find the source of truth correctly.
  Falls back to package resources for installed distributions.

  Returns:
      Path: The absolute path to the 'semantics' directory.

  """
  # 1. Local Source Priority (Dev/Test/Editable)
  local_path = Path(__file__).parent
  # Simple check: does the main neural file exist here?
  if (local_path / "odl").exists() or (local_path / "k_neural_net.json").exists():
    return local_path

  # 2. Installed Package Fallback
  if sys.version_info >= (3, 9) and files is not None:
    try:
      # Note: wrapping in Path(str(...)) ensures compatibility issues
      # with early 3.9 implementations are smoothed over.
      return Path(str(files("ml_switcheroo.semantics")))
    except Exception:
      pass

  # Fallback to local path if discovery fails
  return local_path


def resolve_snapshots_dir() -> Path:
  """Locate the directory containing framework snapshots and mapping overlays.

  Prioritizes the ecosystem snapshot paths and user environment configurations
  before falling back to legacy repositories.

  Priority Order:
      1. `$ML_ECOSYSTEM_SNAPSHOTS_DIR` environment variable.
      2. Sibling repository `../ml-ecosystem-snapshots/src/ml_ecosystem_snapshots/snapshots/`
         or `../ml-ecosystem-snapshots/src/ml_framework_snapshots/snapshots/`.
      3. User cache directory `~/.cache/ml_ecosystem_snapshots/`.
      4. Legacy `$ML_FRAMEWORK_SNAPSHOTS_DIR` environment variable.
      5. Legacy sibling repositories `../ml-compiler-snapshots` or `../ml-framework-snapshots`.
      6. Fallback candidate path.

  Returns:
      Path: The absolute path to the resolved 'snapshots' directory.

  """
  # 1. Check $ML_ECOSYSTEM_SNAPSHOTS_DIR environment variable
  env_eco = os.environ.get("ML_ECOSYSTEM_SNAPSHOTS_DIR")
  if env_eco:
    eco_path = Path(env_eco)
    if eco_path.exists():
      return eco_path

  # 2. Check sibling ml-ecosystem-snapshots repository
  repos_root = resolve_semantics_dir().parent.parent.parent.parent
  eco_snap = repos_root / "ml-ecosystem-snapshots" / "src" / "ml_ecosystem_snapshots" / "snapshots"
  if eco_snap.exists():
    return eco_snap

  eco_fw_snap = repos_root / "ml-ecosystem-snapshots" / "src" / "ml_framework_snapshots" / "snapshots"
  if eco_fw_snap.exists():
    return eco_fw_snap

  # 3. Check user cache ~/.cache/ml_ecosystem_snapshots/
  cache_snap = Path.home() / ".cache" / "ml_ecosystem_snapshots" / "snapshots"
  if cache_snap.exists():
    return cache_snap
  cache_dir = Path.home() / ".cache" / "ml_ecosystem_snapshots"
  if cache_dir.exists():
    return cache_dir

  # 4. Check legacy $ML_FRAMEWORK_SNAPSHOTS_DIR environment variable
  env_fw = os.environ.get("ML_FRAMEWORK_SNAPSHOTS_DIR")
  if env_fw:
    fw_path = Path(env_fw)
    if fw_path.exists():
      return fw_path

  # 5. Check legacy sibling repositories
  candidate = repos_root / "ml-compiler-snapshots"
  if candidate.exists():
    return candidate
  framework_candidate = repos_root / "ml-framework-snapshots" / "src" / "ml_framework_snapshots" / "snapshots"
  if framework_candidate.exists():
    return framework_candidate

  return candidate
