"""Hooks Registry Module.

This module provides the global registry and dynamic loading mechanisms for
translation hooks defined in plugins. It manages hook registration, metadata storage,
lazy-loading of plugins, and retrieval of hooks for AST transformations.
"""

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
  """Decorator to register a custom translation hook for a specific trigger.

  Args:
      trigger (str): The name or identifier of the framework function/operation
          that triggers this hook (e.g., 'jax.numpy.add').
      auto_wire (Optional[Dict[str, Any]]): Optional configuration dictionary for
          auto-wiring the hook, validated against `AutoWireSpec`.

  Returns:
      Callable[[Any], Any]: A decorator function that registers the target callable.

  """

  def decorator(func: Any) -> Any:
    """Registers the decorated function as a translation hook for the trigger.

    Args:
        func (Any): The hook function to be registered.

    Returns:
        Any: The registered hook function, unchanged.

    """
    _HOOKS[trigger] = func
    if auto_wire:
      spec = AutoWireSpec.model_validate(auto_wire)
      _HOOK_METADATA[trigger] = spec
    return func

  return decorator


def get_hook(trigger: str) -> Optional[Callable[..., cst.CSTNode]]:
  """Retrieves a registered translation hook for the given trigger, loading plugins if needed.

  If plugins have not been loaded yet, calling this function will automatically trigger
  the loading of all plugins.

  Args:
      trigger (str): The trigger identifier to search for.

  Returns:
      Optional[Callable[..., cst.CSTNode]]: The registered hook function if found,
          otherwise None.

  """
  global _PLUGINS_LOADED
  if not _PLUGINS_LOADED:
    load_plugins()
    _PLUGINS_LOADED = True
  return _HOOKS.get(trigger)


def get_all_hook_metadata() -> Dict[str, AutoWireSpec]:
  """Retrieves the metadata for all registered hooks.

  Returns:
      Dict[str, AutoWireSpec]: A dictionary mapping hook triggers to their corresponding
          AutoWireSpec metadata.

  """
  return _HOOK_METADATA


def clear_hooks() -> None:
  """Clears all registered hooks and their metadata from the global registry.

  This also resets the internal plugin loading status, allowing plugins to be
  reloaded on subsequent hook requests.

  """
  global _PLUGINS_LOADED
  _HOOKS.clear()
  _HOOK_METADATA.clear()
  _PLUGINS_LOADED = False


def load_plugins(plugins_dir: Optional[Path] = None, extra_dirs: Optional[List[Path]] = None) -> int:
  """Dynamically loads plugin modules from the primary and optional extra directories.

  If the primary `plugins_dir` is not specified, it defaults to the package's built-in
  'plugins' directory relative to this file.

  Args:
      plugins_dir (Optional[Path]): The primary directory to load plugins from.
          Defaults to the package's built-in 'plugins' directory if None.
      extra_dirs (Optional[List[Path]]): Additional directories to search and load
          plugins from.

  Returns:
      int: The total number of plugin modules successfully loaded or reloaded.

  """
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
  """Helper function to discover and import/reload all Python modules in a directory.

  If `base_package` is specified, it performs a package-relative import. Otherwise,
  it temporarily adds the directory to `sys.path` to allow direct top-level imports.

  Args:
      directory (Path): The filesystem directory containing target Python files.
      base_package (Optional[str]): The dot-separated base package path prefix
          to use for the imported modules.

  Returns:
      int: The count of successfully imported or reloaded plugin modules.

  """
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
    except Exception as e:
      logging.getLogger("ml_switcheroo.hooks").warning(f"Failed to load plugin {py_file.name}: {e}")

  if base_package is None:
    sys.path.pop(0)

  return count
