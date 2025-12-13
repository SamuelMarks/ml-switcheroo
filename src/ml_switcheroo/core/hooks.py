"""
Plugin Registry, Hook Context, and Dynamic Loader.

This module provides the infrastructure for extending ml-switcheroo via plugins.
It enables developers to intercept and modify the Abstract Syntax Tree (AST)
during the conversion process using a hook-based system.

How to write a Plugin
---------------------

A plugin is a simple Python function decorated with ``@register_hook``. It receives
the current AST node (typically ``cst.Call``, but can be others like ``cst.For``)
and a context object.

Usage
-----
.. code-block:: python

    @register_hook("my_custom_trigger")
    def transform_node(node: cst.CSTNode, ctx: HookContext) -> cst.CSTNode:
        # Logic
        return node
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Any, Type, TypeVar, List, Union
import libcst as cst
from pydantic import BaseModel

# We import RuntimeConfig for type hinting
from ml_switcheroo.config import RuntimeConfig

# Validation that SemanticsManager is importable for type hinting
SemanticsManagerType = Any
T = TypeVar("T", bound=BaseModel)

# Callbacks for plugin side-effects
ArgInjectorType = Callable[[str, Optional[str]], None]
PreambleInjectorType = Callable[[str], None]


class HookContext:
  """
  Context object passed to every plugin hook during transcoding.

  Provides read-only access to global state and write access
  to specific injection points (signature args, function body preambles).
  """

  def __init__(
    self,
    semantics: SemanticsManagerType,
    config: RuntimeConfig,
    arg_injector: Optional[ArgInjectorType] = None,
    preamble_injector: Optional[PreambleInjectorType] = None,
  ):
    self.semantics = semantics
    self._runtime_config = config
    self._arg_injector = arg_injector
    self._preamble_injector = preamble_injector

    self.source_fw = config.source_framework
    self.target_fw = config.target_framework
    self.metadata: Dict[str, Any] = {}

  def inject_signature_arg(self, name: str, annotation: Optional[str] = None) -> None:
    """Requests injection of an argument into the current function signature."""
    if self._arg_injector:
      self._arg_injector(name, annotation)

  def inject_preamble(self, code_str: str) -> None:
    """Requests injection of a statement at the beginning of the function body."""
    if self._preamble_injector:
      self._preamble_injector(code_str)

  def raw_config(self, key: str, default: Any = None) -> Any:
    """Retrieve a raw value from the unstructured plugin settings dict."""
    return self._runtime_config.plugin_settings.get(key, default)

  def config(self, key: str, default: Any = None) -> Any:
    """Legacy alias for raw_config."""
    return self.raw_config(key, default)

  def validate_settings(self, model: Type[T]) -> T:
    """Validates global config against a Plugin-specific Pydantic schema."""
    relevant_keys = model.model_fields.keys()
    subset = {k: v for k, v in self._runtime_config.plugin_settings.items() if k in relevant_keys}
    return model.model_validate(subset)

  def lookup_api(self, op_name: str) -> Optional[str]:
    """Resolves target framework's API string for a given standard operation."""
    if not self.semantics:
      return None

    known_apis = self.semantics.get_known_apis()
    details = known_apis.get(op_name)
    if not details:
      return None

    variants = details.get("variants", {})
    target_variant = variants.get(self.target_fw)

    if not target_variant:
      return None

    return target_variant.get("api")

  def lookup_signature(self, op_name: str) -> List[str]:
    """Retrieves standard argument list for a given operation."""
    if not self.semantics:
      return []
    known_apis = self.semantics.get_known_apis()
    details = known_apis.get(op_name)
    if not details:
      return []
    std_args = details.get("std_args", [])
    cleaned_args = []
    for item in std_args:
      if isinstance(item, (list, tuple)):
        cleaned_args.append(item[0])
      else:
        cleaned_args.append(item)
    return cleaned_args


# Updated Type alias to allow arbitrary CSTNodes (e.g. For, Call)
HookFunction = Callable[[Any, HookContext], Any]

# Global Registry
_HOOKS: Dict[str, HookFunction] = {}
_PLUGINS_LOADED = False


def register_hook(trigger: str) -> Callable[[HookFunction], HookFunction]:
  """
  Decorator to register a function as a plugin hook.

  Args:
      trigger: The unique identifier. Can be an operation ID or a
               reserved system event like "transform_for_loop".
  """

  def decorator(func: HookFunction) -> HookFunction:
    _HOOKS[trigger] = func
    return func

  return decorator


def get_hook(trigger: str) -> Optional[HookFunction]:
  """
  Retrieves a registered hook function by its trigger name.
  """
  if not _PLUGINS_LOADED:
    load_plugins()
  return _HOOKS.get(trigger)


def clear_hooks() -> None:
  """Resets the internal hook registry."""
  global _PLUGINS_LOADED
  _HOOKS.clear()
  _PLUGINS_LOADED = False


def load_plugins(plugins_dir: Optional[Path] = None, extra_dirs: Optional[List[Path]] = None) -> int:
  """
  Dynamically imports all modules in the plugins directory.
  """
  global _PLUGINS_LOADED
  total_loaded = 0

  if not _PLUGINS_LOADED:
    target_dir = plugins_dir
    if target_dir is None:
      current_dir = Path(__file__).resolve().parent
      target_dir = current_dir.parent / "plugins"

    # Fallback to package lookup
    if not target_dir.exists():
      try:
        import ml_switcheroo.plugins

        if ml_switcheroo.plugins.__file__:
          target_dir = Path(ml_switcheroo.plugins.__file__).parent
      except (ImportError, AttributeError):
        pass

    if target_dir and target_dir.exists():
      total_loaded += _import_from_dir(target_dir, base_package="ml_switcheroo.plugins")
      _PLUGINS_LOADED = True

  if extra_dirs:
    for ex_dir in extra_dirs:
      if ex_dir.exists() and ex_dir.is_dir():
        total_loaded += _import_from_dir(ex_dir, base_package=None)

  return total_loaded


def _import_from_dir(directory: Path, base_package: Optional[str] = None) -> int:
  """Helper to iterate and import python files from a directory."""
  count = 0
  for item in directory.glob("*.py"):
    if item.name == "__init__.py":
      continue
    module_name = item.stem
    if base_package:
      try:
        if base_package in sys.modules or (
          sys.modules.get("ml_switcheroo") and hasattr(sys.modules["ml_switcheroo"], "plugins")
        ):
          importlib.import_module(f"{base_package}.{module_name}")
          count += 1
          continue
      except Exception:
        pass
    try:
      unique_name = f"switcheroo_plugin_{module_name}_{item.stat().st_ino}"
      spec = importlib.util.spec_from_file_location(unique_name, item)
      if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = mod
        spec.loader.exec_module(mod)
        count += 1
    except Exception as e:
      print(f"Failed to load plugin {item.name}: {e}")
  return count
