"""Optional Workspace & Distribution Discovery Bootstrap Utilities.

Isolates optional development-workspace ``sys.path`` registration from core compiler
and tracing modules. By default, packages such as ``ml_switcheroo_compiler`` and
``zero_*`` are resolved via standard Python package discovery (``importlib.util.find_spec``)
without walking parent directories or mutating ``sys.path``. Only when an explicit
``workspace_parent`` path or ``ML_SWITCHEROO_WORKSPACE_PATH`` environment variable is
configured are sibling ``<repo>/src`` directories appended to ``sys.path``.
"""

import importlib.util
import os
import sys
from pathlib import Path

_WORKSPACE_REPOS: tuple[str, ...] = (
  "ml-switcheroo-compiler",
  "ml-switcheroo-ir",
  "zero-jax",
  "zero-chex",
  "zero-optax",
  "zero-grain",
  "zero-orbax",
  "zero-flax",
  "zero-pytorch",
  "zero-keras",
  "zero-tensorflow",
  "zero-mlx",
  "zero-pax",
  "zero-zoo",
)


def is_package_installed(module_name: str) -> bool:
  """Check whether ``module_name`` is discoverable via standard ``importlib.util.find_spec``.

  Args:
      module_name: Top-level or dotted Python module name.

  Returns:
      bool: ``True`` if ``module_name`` can be resolved on the active Python import path.
  """
  try:
    return importlib.util.find_spec(module_name) is not None
  except (ImportError, ValueError, AttributeError):
    return False


def ensure_compiler_and_zero_on_path(workspace_parent: Path | None = None) -> list[str]:
  """Register configured workspace ``<repo>/src`` directories on ``sys.path`` when explicitly requested.

  Does NOT walk relative parent directories from ``__file__``. If neither ``workspace_parent``
  nor the ``ML_SWITCHEROO_WORKSPACE_PATH`` environment variable is provided, relies strictly
  on standard installed-package discovery and returns ``[]``.

  Args:
      workspace_parent: Optional explicit workspace directory containing sibling repositories.

  Returns:
      list[str]: Newly added directory paths.
  """
  effective_parent = workspace_parent
  if effective_parent is None:
    env_dir = os.environ.get("ML_SWITCHEROO_WORKSPACE_PATH", "").strip()
    if env_dir:
      effective_parent = Path(env_dir)

  if effective_parent is None:
    return []

  added: list[str] = []
  for repo_name in _WORKSPACE_REPOS:
    src_dir = effective_parent / repo_name / "src"
    src_str = str(src_dir)
    if src_dir.is_dir() and src_str not in sys.path:
      sys.path.append(src_str)
      added.append(src_str)
  return added


__all__ = ["ensure_compiler_and_zero_on_path", "is_package_installed"]
