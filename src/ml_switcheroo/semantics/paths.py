"""Path Resolution Utilities for Semantics.

Handles locating the 'semantics/' and 'snapshots/' directories
within the package or source tree.
"""

import os
from importlib.resources import files
from pathlib import Path

try:
  from ml_ecosystem_snapshots.network_env import get_custom_snapshots_paths
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
  try:
    return Path(str(files("ml_switcheroo.semantics")))
  except (OSError, KeyError, ValueError, TypeError, ModuleNotFoundError):
    pass

  # Fallback to local path if discovery fails
  return local_path


def resolve_snapshots_dir() -> Path:
  """Locate the directory containing framework snapshots and mapping overlays.

  Prioritizes the snapshot paths and user environment configurations
  before falling back to sibling ecosystem repository or user cache.

  Priority Order:
      1. `$ML_SNAPSHOTS_PATH` configured custom path.
      2. Custom snapshot paths from ecosystem utils.
      3. `$ML_ECOSYSTEM_SNAPSHOTS_DIR` environment variable.
      4. Sibling repository `../ml-ecosystem-snapshots/src/ml_ecosystem_snapshots/snapshots/`
         or `../ml-ecosystem-snapshots/snapshots/`.
      5. User cache directory `~/.cache/ml_ecosystem_snapshots/`.
      6. Canonical sibling fallback path `../ml-ecosystem-snapshots`.

  Returns:
      Path: The absolute path to the resolved 'snapshots' directory.

  """
  # 1. Check direct environment variable ML_SNAPSHOTS_PATH
  env_val = os.environ.get("ML_SNAPSHOTS_PATH")
  if env_val:
    p = Path(env_val)
    if p.exists():
      return p

  # 2. Check custom snapshot paths configured via ecosystem utilities
  for custom_dir in get_custom_snapshots_paths():
    c_path = Path(custom_dir)
    if c_path.exists():
      return c_path

  # 3. Check $ML_ECOSYSTEM_SNAPSHOTS_DIR environment variable
  env_eco = os.environ.get("ML_ECOSYSTEM_SNAPSHOTS_DIR")
  if env_eco:
    eco_path = Path(env_eco)
    if eco_path.exists():
      return eco_path

  # 4. Check sibling ml-ecosystem-snapshots or ml-framework-snapshots repository
  repos_root = resolve_semantics_dir().parent.parent.parent.parent
  for repo_name in ("ml-ecosystem-snapshots", "ml-framework-snapshots"):
    eco_snap = repos_root / repo_name / "src" / "ml_ecosystem_snapshots" / "snapshots"
    if eco_snap.exists():
      return eco_snap

    eco_snap_direct = repos_root / repo_name / "snapshots"
    if eco_snap_direct.exists():
      return eco_snap_direct

  # 5. Check user cache ~/.cache/ml_ecosystem_snapshots/
  cache_snap = Path.home() / ".cache" / "ml_ecosystem_snapshots" / "snapshots"
  if cache_snap.exists():
    return cache_snap
  cache_dir = Path.home() / ".cache" / "ml_ecosystem_snapshots"
  if cache_dir.exists():
    return cache_dir

  # 6. Return canonical sibling ml-ecosystem-snapshots path
  candidate = repos_root / "ml-ecosystem-snapshots"
  return candidate


__all__ = ["files", "get_custom_snapshots_paths", "resolve_semantics_dir", "resolve_snapshots_dir"]
