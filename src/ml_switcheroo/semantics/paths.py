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

try:
  from ml_ecosystem_snapshots.utils import get_custom_snapshots_paths
except ImportError:

  def get_custom_snapshots_paths() -> list[str]:
    """Retrieve custom snapshot directory paths configured via environment variables.

    Returns:
        list[str]: Empty list fallback when ml_ecosystem_snapshots is not installed.
    """
    return []


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

  Prioritizes the snapshot paths and user environment configurations
  before falling back to legacy repositories.

  Priority Order:
      1. `$ML_SNAPSHOTS_PATH` / `$ML_FRAMEWORK_SNAPSHOTS_PATH` configured custom paths.
      2. Custom snapshot paths from ecosystem utils.
      3. `$ML_FRAMEWORK_SNAPSHOTS_DIR` / `$ML_ECOSYSTEM_SNAPSHOTS_DIR` environment variables.
      4. Sibling repository `../ml-framework-snapshots/src/ml_framework_snapshots/snapshots/`
         or `../ml-framework-snapshots/snapshots/`.
      5. Sibling repository `../ml-ecosystem-snapshots/src/ml_ecosystem_snapshots/snapshots/`
         or `../ml-ecosystem-snapshots/src/ml_framework_snapshots/snapshots/`.
      6. User cache directory `~/.cache/ml_ecosystem_snapshots/`.
      7. Legacy sibling repository `../ml-compiler-snapshots`.
      8. Fallback candidate path.

  Returns:
      Path: The absolute path to the resolved 'snapshots' directory.

  """
  # 1. Check direct environment variables ML_SNAPSHOTS_PATH or ML_FRAMEWORK_SNAPSHOTS_PATH
  for env_key in ("ML_SNAPSHOTS_PATH", "ML_FRAMEWORK_SNAPSHOTS_PATH"):
    env_val = os.environ.get(env_key)
    if env_val:
      p = Path(env_val)
      if p.exists():
        return p

  # 2. Check custom snapshot paths configured via ecosystem utilities
  for custom_dir in get_custom_snapshots_paths():
    c_path = Path(custom_dir)
    if c_path.exists():
      return c_path

  # 3. Check $ML_FRAMEWORK_SNAPSHOTS_DIR environment variable
  env_fw = os.environ.get("ML_FRAMEWORK_SNAPSHOTS_DIR")
  if env_fw:
    fw_path = Path(env_fw)
    if fw_path.exists():
      return fw_path

  # 4. Check $ML_ECOSYSTEM_SNAPSHOTS_DIR environment variable
  env_eco = os.environ.get("ML_ECOSYSTEM_SNAPSHOTS_DIR")
  if env_eco:
    eco_path = Path(env_eco)
    if eco_path.exists():
      return eco_path

  # 5. Check sibling ml-framework-snapshots repository
  repos_root = resolve_semantics_dir().parent.parent.parent.parent
  fw_snap = repos_root / "ml-framework-snapshots" / "src" / "ml_framework_snapshots" / "snapshots"
  if fw_snap.exists():
    return fw_snap

  fw_snap_direct = repos_root / "ml-framework-snapshots" / "snapshots"
  if fw_snap_direct.exists():
    return fw_snap_direct

  # 6. Check sibling ml-ecosystem-snapshots repository
  eco_snap = repos_root / "ml-ecosystem-snapshots" / "src" / "ml_ecosystem_snapshots" / "snapshots"
  if eco_snap.exists():
    return eco_snap

  eco_fw_snap = repos_root / "ml-ecosystem-snapshots" / "src" / "ml_framework_snapshots" / "snapshots"
  if eco_fw_snap.exists():
    return eco_fw_snap

  # 7. Check user cache ~/.cache/ml_ecosystem_snapshots/
  cache_snap = Path.home() / ".cache" / "ml_ecosystem_snapshots" / "snapshots"
  if cache_snap.exists():
    return cache_snap
  cache_dir = Path.home() / ".cache" / "ml_ecosystem_snapshots"
  if cache_dir.exists():
    return cache_dir

  # 8. Check legacy sibling repositories
  candidate = repos_root / "ml-compiler-snapshots"
  if candidate.exists():
    return candidate

  return candidate
