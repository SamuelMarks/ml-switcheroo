"""Hook Registry Module.

This module provides an instantiable ``HookRegistry`` and lazy-loading mechanisms for
translation hooks defined in plugins. It manages hook registration, metadata storage,
scoped registry binding, lazy plugin discovery, and retrieval of hooks for AST transformations.
"""

import contextvars
import importlib
import importlib.util
import logging
import sys
import typing
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import libcst as cst
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from ml_switcheroo.exceptions import PluginLoadError

if typing.TYPE_CHECKING:
  from ml_switcheroo.core.hooks import HookContext


class AutoWireSpec(BaseModel):
  """Schema for plugin self-registration metadata.

  Allows a plugin to declare the semantic operations it satisfies.

  Attributes:
      ops: Dictionary of abstract operations to inject into ``SemanticsManager``.
  """

  model_config = ConfigDict(extra="allow")

  ops: dict[str, JsonValue] | None = Field(
    default_factory=dict,
    description="Dictionary of Abstract Operations to inject into SemanticsManager.",
  )


HookReturn: typing.TypeAlias = cst.CSTNode | cst.RemovalSentinel | cst.FlattenSentinel[cst.BaseStatement]
_NodeT_contra = typing.TypeVar("_NodeT_contra", bound=cst.CSTNode, contravariant=True)


class HookCallableProtocol(typing.Protocol[_NodeT_contra]):
  """Protocol for registered AST hook callables."""

  def __call__(self, node: _NodeT_contra, ctx: "HookContext") -> HookReturn:
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


def _is_hook_callable(func: _HOOKT | HookCallable) -> typing.TypeGuard[HookCallable]:
  """Verify whether a decorated function is a valid AST hook callable.

  Args:
      func: The callable being registered.

  Returns:
      True if ``func`` is callable, narrowing it to ``HookCallable``.
  """
  return callable(func)


HookRegistrySnapshot: typing.TypeAlias = tuple[
  dict[str, HookCallable],
  dict[str, AutoWireSpec],
  bool,
]


class HookRegistry:
  """Instantiable registry encapsulating registered plugin hooks, auto-wire specs, and lazy-load state.

  Attributes:
      hooks: Mapping from trigger identifiers to registered hook callables.
      metadata: Mapping from trigger identifiers to ``AutoWireSpec`` declarations.
      plugins_loaded: Whether built-in plugins have been loaded into this registry.
  """

  def __init__(self, *, plugins_loaded: bool = False) -> None:
    """Initialize an isolated hook registry.

    Args:
        plugins_loaded: Initial flag indicating whether built-in plugin discovery is satisfied.
    """
    self.hooks: dict[str, HookCallable] = {}
    self.metadata: dict[str, AutoWireSpec] = {}
    self.plugins_loaded: bool = plugins_loaded

  def register_callable(
    self,
    trigger: str,
    func: HookCallable,
    auto_wire: typing.Mapping[str, JsonValue] | None = None,
  ) -> None:
    """Register a validated hook callable and optional auto-wire metadata under ``trigger``.

    Args:
        trigger: Identifier of the operation or hook trigger.
        func: Callable transforming a CST node within a ``HookContext``.
        auto_wire: Optional mapping validated into ``AutoWireSpec``.
    """
    self.hooks[trigger] = func
    if auto_wire:
      self.metadata[trigger] = AutoWireSpec.model_validate(auto_wire)

  def register(
    self,
    trigger: str,
    auto_wire: typing.Mapping[str, JsonValue] | None = None,
  ) -> typing.Callable[[_HOOKT], _HOOKT]:
    """Create a decorator registering a hook callable into this registry instance.

    Args:
        trigger: Identifier of the operation or hook trigger.
        auto_wire: Optional configuration dictionary validated against ``AutoWireSpec``.

    Returns:
        Decorator returning the original function unchanged.
    """

    def decorator(func: _HOOKT) -> _HOOKT:
      """Register the decorated callable in this ``HookRegistry``.

      Args:
          func: Decorated hook callable.

      Returns:
          Unchanged ``func``.

      Raises:
          PluginLoadError: If ``func`` is not callable.
      """
      reg_func: _HOOKT | HookCallable = func
      if not _is_hook_callable(reg_func):
        raise PluginLoadError(f"Registered hook for trigger '{trigger}' must be callable.")
      self.register_callable(trigger, reg_func, auto_wire=auto_wire)
      return func

    return decorator

  def ensure_loaded(self) -> None:
    """Lazily load built-in plugins into this registry if not already loaded."""
    if not self.plugins_loaded:
      self.plugins_loaded = True
      self.load_plugins()

  def get(self, trigger: str) -> HookCallable | None:
    """Look up a hook callable by ``trigger``, lazily loading built-in plugins if needed.

    Args:
        trigger: Trigger identifier to resolve.

    Returns:
        Registered ``HookCallable`` if present, otherwise ``None``.
    """
    self.ensure_loaded()
    return self.hooks.get(trigger)

  def get_all_metadata(self) -> dict[str, AutoWireSpec]:
    """Return auto-wire specifications registered in this registry.

    Returns:
        Mapping of trigger names to ``AutoWireSpec`` instances.
    """
    return self.metadata

  def clear(self, *, mark_loaded: bool = False) -> None:
    """Remove all hooks and metadata from this registry.

    Args:
        mark_loaded: Value assigned to ``self.plugins_loaded`` after clearing.
    """
    self.hooks.clear()
    self.metadata.clear()
    self.plugins_loaded = mark_loaded

  def snapshot(self) -> HookRegistrySnapshot:
    """Capture a shallow copy of this registry's hooks, metadata, and load state.

    Returns:
        Tuple of ``(hooks_copy, metadata_copy, plugins_loaded)``.
    """
    return (self.hooks.copy(), self.metadata.copy(), self.plugins_loaded)

  def restore(self, state: HookRegistrySnapshot) -> None:
    """Restore this registry from a snapshot previously returned by ``snapshot()``.

    Args:
        state: Snapshot tuple ``(hooks_copy, metadata_copy, plugins_loaded)``.
    """
    saved_hooks, saved_meta, saved_loaded = state
    self.hooks.clear()
    self.hooks.update(saved_hooks)
    self.metadata.clear()
    self.metadata.update(saved_meta)
    self.plugins_loaded = saved_loaded

  def load_plugins(
    self,
    plugins_dir: Path | None = None,
    extra_dirs: typing.Sequence[str | Path] | None = None,
  ) -> int:
    """Discover and load plugin modules into this registry instance.

    Args:
        plugins_dir: Primary built-in plugins directory (defaults to ``ml_switcheroo/plugins``).
        extra_dirs: Optional additional plugin directories to import.

    Returns:
        Total count of plugin modules imported or reloaded.
    """
    if plugins_dir is None:
      plugins_dir = Path(__file__).parent.parent / "plugins"
      self.plugins_loaded = True

    count = 0
    with bind_hook_registry(self):
      if plugins_dir.exists():
        count += _import_from_dir(plugins_dir, "ml_switcheroo.plugins")
      if extra_dirs:
        for edir in extra_dirs:
          edir_path = Path(edir)
          if edir_path.exists():
            count += _import_from_dir(edir_path)
    return count


_ACTIVE_HOOK_REGISTRY: contextvars.ContextVar[HookRegistry | None] = contextvars.ContextVar(
  "_ACTIVE_HOOK_REGISTRY",
  default=None,
)


@lru_cache(maxsize=1)
def get_default_hook_registry() -> HookRegistry:
  """Return the lazily instantiated default ``HookRegistry``.

  Returns:
      Default ``HookRegistry`` instance.
  """
  return HookRegistry()


def _current_hook_registry() -> HookRegistry:
  """Return the scoped active ``HookRegistry``, falling back to ``get_default_hook_registry()``.

  Returns:
      Active ``HookRegistry`` instance.
  """
  scoped = _ACTIVE_HOOK_REGISTRY.get()
  return scoped if scoped is not None else get_default_hook_registry()


@contextmanager
def bind_hook_registry(registry: HookRegistry) -> typing.Iterator[HookRegistry]:
  """Temporarily bind ``registry`` as the active hook registry for the current execution context.

  Args:
      registry: ``HookRegistry`` to activate within the ``with`` block.

  Yields:
      The bound ``HookRegistry`` instance.
  """
  token = _ACTIVE_HOOK_REGISTRY.set(registry)
  try:
    yield registry
  finally:
    _ACTIVE_HOOK_REGISTRY.reset(token)


def register_hook(
  trigger: str,
  auto_wire: typing.Mapping[str, JsonValue] | None = None,
) -> typing.Callable[[_HOOKT], _HOOKT]:
  """Register a custom translation hook for ``trigger`` in the active ``HookRegistry``.

  Args:
      trigger: Identifier of the framework function or plugin operation (e.g., ``'jax.numpy.add'``).
      auto_wire: Optional configuration dictionary validated against ``AutoWireSpec``.

  Returns:
      Decorator registering the target callable in the active ``HookRegistry``.
  """

  def decorator(func: _HOOKT) -> _HOOKT:
    """Register the decorated function as a translation hook for ``trigger``.

    Args:
        func: Hook function to be registered.

    Returns:
        Unchanged hook function.

    Raises:
        PluginLoadError: If ``func`` is not callable.
    """
    reg_func: _HOOKT | HookCallable = func
    if not _is_hook_callable(reg_func):
      raise PluginLoadError(f"Registered hook for trigger '{trigger}' must be callable.")
    _current_hook_registry().register_callable(trigger, reg_func, auto_wire=auto_wire)
    return func

  return decorator


def get_hook(trigger: str) -> HookCallable | None:
  """Retrieve a registered translation hook for ``trigger`` from the active ``HookRegistry``.

  Lazily loads built-in plugins on first lookup if not yet loaded.

  Args:
      trigger: Trigger identifier to look up.

  Returns:
      Registered ``HookCallable`` if found, otherwise ``None``.
  """
  return _current_hook_registry().get(trigger)


def get_all_hook_metadata() -> dict[str, AutoWireSpec]:
  """Retrieve auto-wire metadata for all hooks in the active ``HookRegistry``.

  Returns:
      Dictionary mapping hook triggers to ``AutoWireSpec`` metadata.
  """
  return _current_hook_registry().get_all_metadata()


def clear_hooks(*, mark_loaded: bool = False) -> None:
  """Clear all registered hooks and metadata in the active ``HookRegistry``.

  Args:
      mark_loaded: Whether to mark plugins as already loaded after clearing.
  """
  _current_hook_registry().clear(mark_loaded=mark_loaded)


def set_plugins_loaded(loaded: bool = True) -> None:
  """Set ``plugins_loaded`` on the active ``HookRegistry``.

  Args:
      loaded: Desired plugin loaded state.
  """
  _current_hook_registry().plugins_loaded = loaded


def load_plugins(
  plugins_dir: Path | None = None,
  extra_dirs: typing.Sequence[str | Path] | None = None,
) -> int:
  """Dynamically load plugin modules into the active ``HookRegistry``.

  Args:
      plugins_dir: Primary directory to load plugins from (defaults to built-in ``plugins``).
      extra_dirs: Additional directories to search and load plugins from.

  Returns:
      Total number of plugin modules successfully loaded or reloaded.
  """
  return _current_hook_registry().load_plugins(plugins_dir=plugins_dir, extra_dirs=extra_dirs)


def _import_from_dir(directory: Path, base_package: str | None = None) -> int:
  """Discover and import/reload all non-helper Python modules in ``directory``.

  Args:
      directory: Filesystem directory containing candidate plugin ``.py`` files.
      base_package: Optional dot-separated package prefix for package-relative imports.

  Returns:
      Count of successfully imported or reloaded plugin modules.
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
        if spec is None or spec.loader is None:
          raise PluginLoadError(f"Unable to build module spec for plugin '{py_file}'.")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full_mod_name] = mod
        spec.loader.exec_module(mod)
      count += 1
    except (
      PluginLoadError,
      SyntaxError,
      ImportError,
      OSError,
      ValueError,
      TypeError,
      AttributeError,
      RuntimeError,
    ) as e:
      logging.getLogger("ml_switcheroo.hooks").warning(f"Failed to load plugin {py_file.name}: {e}")

  return count


__all__ = [
  "AutoWireSpec",
  "HookCallable",
  "HookCallableProtocol",
  "HookInputCallable",
  "HookRegistry",
  "HookRegistrySnapshot",
  "HookReturn",
  "bind_hook_registry",
  "clear_hooks",
  "get_all_hook_metadata",
  "get_default_hook_registry",
  "get_hook",
  "load_plugins",
  "register_hook",
  "set_plugins_loaded",
]
