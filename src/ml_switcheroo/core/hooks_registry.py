"""Hooks Registry."""

import importlib
import logging
from pathlib import Path
from typing import Dict, Optional, Callable, Any, List
import libcst as cst
from ml_switcheroo.core.hooks import AutoWireSpec

# Global registry
_HOOKS: Dict[str, Callable[..., cst.CSTNode]] = {}
_HOOK_METADATA: Dict[str, AutoWireSpec] = {}


_PLUGINS_LOADED = False


def register_hook(trigger: str, auto_wire: Optional[Dict[str, Any]] = None) -> Callable[[Any], Any]:
  """Docstring."""

  def decorator(func: Any) -> Any:
    """Docstring."""
    _HOOKS[trigger] = func
    if auto_wire:
      spec = AutoWireSpec.model_validate(auto_wire)
      _HOOK_METADATA[trigger] = spec
    return func

  return decorator


def get_hook(trigger: str) -> Optional[Callable[..., cst.CSTNode]]:
  """Docstring."""
  global _PLUGINS_LOADED
  if not _PLUGINS_LOADED:
    load_plugins()
    _PLUGINS_LOADED = True
  return _HOOKS.get(trigger)


def get_all_hook_metadata() -> Dict[str, AutoWireSpec]:
  """Docstring."""
  return _HOOK_METADATA


def clear_hooks() -> None:
  """Docstring."""
  global _PLUGINS_LOADED
  _HOOKS.clear()  # pragma: no cover
  _HOOK_METADATA.clear()  # pragma: no cover
  _PLUGINS_LOADED = False  # pragma: no cover


def load_plugins(plugins_dir: Optional[Path] = None, extra_dirs: Optional[List[Path]] = None) -> int:
  """Docstring."""
  if plugins_dir is None:
    # Default to the src/ml_switcheroo/plugins directory relative to this file
    plugins_dir = Path(__file__).parent.parent / "plugins"

  count = 0
  if plugins_dir and plugins_dir.exists():
    count += _import_from_dir(plugins_dir, "ml_switcheroo.plugins")
  if extra_dirs:
    for edir in extra_dirs:
      if edir.exists():
        count += _import_from_dir(edir)
  return count


def _import_from_dir(directory: Path, base_package: Optional[str] = None) -> int:
  """Docstring."""
  count = 0
  import sys

  if base_package is None:
    sys.path.insert(0, str(directory))

  for py_file in directory.glob("*.py"):
    if py_file.name.startswith("__"):
      continue
    mod_name = py_file.stem
    full_mod_name = f"{base_package}.{mod_name}" if base_package else mod_name
    try:
      if full_mod_name in sys.modules:
        importlib.reload(sys.modules[full_mod_name])
      else:
        importlib.import_module(full_mod_name)
      count += 1
    except Exception as e:  # pragma: no cover
      logging.getLogger("ml_switcheroo.hooks").warning(f"Failed to load plugin {py_file.name}: {e}")  # pragma: no cover

  if base_package is None:
    sys.path.pop(0)

  return count
