"""Hook Registry Module.

This module provides the global registry and dynamic loading mechanisms for
translation hooks defined in plugins. It manages hook registration, metadata storage,
lazy-loading of plugins, and retrieval of hooks for AST transformations.
"""

import importlib
import importlib.util
import logging
import sys
import typing
from pathlib import Path

import libcst as cst
from pydantic import JsonValue

from ml_switcheroo.core.hooks import AutoWireSpec, HookContext

# Global registry
HookReturn: typing.TypeAlias = cst.CSTNode | cst.RemovalSentinel | cst.FlattenSentinel[cst.BaseStatement]
_NodeT_contra = typing.TypeVar("_NodeT_contra", bound=cst.CSTNode, contravariant=True)


class HookCallableProtocol(typing.Protocol[_NodeT_contra]):
  """Protocol for registered AST hook callables."""

  def __call__(self, node: _NodeT_contra, ctx: HookContext) -> HookReturn:
    """Transform a CST node within the given hook context.

    Args:
        node: Target CST node.
        ctx: Active hook context.

    Returns:
        Transformed CST node or sentinel.
    """
    ...


HookCallable = HookCallableProtocol[cst.CSTNode]
HookInputCallable: typing.TypeAlias = (
  HookCallableProtocol[cst.Call]
  | HookCallableProtocol[cst.FunctionDef]
  | HookCallableProtocol[cst.ClassDef]
  | HookCallableProtocol[cst.For]
  | HookCallableProtocol[cst.CSTNode]
)
_HOOKT = typing.TypeVar("_HOOKT")
_HOOKS: dict[str, HookCallable] = {}
_HOOK_METADATA: dict[str, AutoWireSpec] = {}


_PLUGINS_LOADED: bool = False


def _is_hook_callable(func: object) -> typing.TypeGuard[HookCallable]:
  """Verify whether a decorated function is a valid AST hook callable.

  Args:
      func: The callable being registered.

  Returns:
      True if ``func`` is callable, narrowing it to ``HookCallable``.
  """
  return callable(func)


def register_hook(
  trigger: str,
  auto_wire: typing.Mapping[str, JsonValue] | None = None,
) -> typing.Callable[[_HOOKT], _HOOKT]:
  """Register a custom translation hook for a specific trigger.

  Args:
      trigger (str): The name or identifier of the framework function/operation
          that triggers this hook (e.g., 'jax.numpy.add').
      auto_wire (Optional[Mapping[str, JsonValue]]): Optional configuration dictionary for
          auto-wiring the hook, validated against `AutoWireSpec`.

  Returns:
      Callable[[_HOOKT], _HOOKT]: A decorator function that registers the target callable.

  """

  def decorator(func: _HOOKT) -> _HOOKT:
    """Register the decorated function as a translation hook for the trigger.

    Args:
        func (_HOOKT): The hook function to be registered.

    Returns:
        _HOOKT: The registered hook function, unchanged.

    """
    reg_func: object = func
    assert _is_hook_callable(reg_func)
    _HOOKS[trigger] = reg_func
    if auto_wire:
      spec = AutoWireSpec.model_validate(auto_wire)
      _HOOK_METADATA[trigger] = spec
    return func

  return decorator


def get_hook(trigger: str) -> HookCallable | None:
  """Retrieve a registered translation hook for the given trigger, loading plugins if needed.

  If plugins have not been loaded yet, calling this function will automatically trigger
  the loading of all plugins.

  Args:
      trigger (str): The trigger identifier to search for.

  Returns:
      Optional[HookCallable]: The registered hook function if found,
          otherwise None.

  """
  global _PLUGINS_LOADED
  if not _PLUGINS_LOADED:
    load_plugins()
    _PLUGINS_LOADED = True
  return _HOOKS.get(trigger)


def get_all_hook_metadata() -> dict[str, AutoWireSpec]:
  """Retrieve the metadata for all registered hooks.

  Returns:
      Dict[str, AutoWireSpec]: A dictionary mapping hook triggers to their corresponding
          AutoWireSpec metadata.

  """
  return _HOOK_METADATA


def clear_hooks() -> None:
  """Clear all registered hooks and their metadata from the global registry.

  This also resets the internal plugin loading status, allowing plugins to be
  reloaded on subsequent hook requests.

  """
  global _PLUGINS_LOADED
  _HOOKS.clear()
  _HOOK_METADATA.clear()
  _PLUGINS_LOADED = False


def load_plugins(plugins_dir: Path | None = None, extra_dirs: list[Path] | None = None) -> int:
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
  global _PLUGINS_LOADED
  if plugins_dir is None:
    # Default to the src/ml_switcheroo/plugins directory relative to this file
    plugins_dir = Path(__file__).parent.parent / "plugins"
    _PLUGINS_LOADED = True

  count = 0
  if plugins_dir.exists():
    count += _import_from_dir(plugins_dir, "ml_switcheroo.plugins")
  if extra_dirs:
    for edir in extra_dirs:
      if edir.exists():
        count += _import_from_dir(edir)
  return count


def _import_from_dir(directory: Path, base_package: str | None = None) -> int:
  """Support function to discover and import/reload all Python modules in a directory.

  If `base_package` is specified, it performs a package-relative import. Otherwise,
  it loads modules directly from their file location via `importlib.util.spec_from_file_location`
  without mutating `sys.path`.

  Args:
      directory (Path): The filesystem directory containing target Python files.
      base_package (Optional[str]): The dot-separated base package path prefix
          to use for the imported modules.

  Returns:
      int: The count of successfully imported or reloaded plugin modules.

  """
  count = 0

  for py_file in sorted(directory.glob("*.py")):
    if py_file.name.startswith("_") or "utils" in py_file.stem or "helpers" in py_file.stem:
      continue
    mod_name = py_file.stem
    full_mod_name = f"{base_package}.{mod_name}" if base_package else mod_name
    try:
      if base_package:
        if full_mod_name in sys.modules:
          importlib.reload(sys.modules[full_mod_name])
        else:
          importlib.import_module(full_mod_name)
      else:
        spec = importlib.util.spec_from_file_location(full_mod_name, py_file)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full_mod_name] = mod
        spec.loader.exec_module(mod)
      count += 1
    except (SyntaxError, ImportError, OSError, ValueError, TypeError, AttributeError, RuntimeError) as e:
      logging.getLogger("ml_switcheroo.hooks").warning(f"Failed to load plugin {py_file.name}: {e}")

  return count
