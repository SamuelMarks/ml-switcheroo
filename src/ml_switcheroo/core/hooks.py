"""
Plugin Registry, Hook Context, and Dynamic Loader.

This module provides the infrastructure for extending ml-switcheroo via plugins.
It enables developers to intercept and modify the Abstract Syntax Tree (AST)
during the conversion process using a hook-based system.

Refactor:
    - Added `auto_wire` support to the `register_hook` decorator.
    - Plugins can now declare their own Semantic definitions, eliminating
      the need to edit `standards_internal.py` or JSON files for self-contained features.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Any, Type, TypeVar, List, Union
from pydantic import BaseModel, Field, ConfigDict

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
    """
    Initializes the hook context.

    Args:
        semantics: Reference to the SemanticsManager.
        config: Runtime configuration (strict mode, selected frameworks).
        arg_injector: Callback to inject arguments into function signature.
        preamble_injector: Callback to inject code at top of function.
    """
    self.semantics = semantics
    self._runtime_config = config
    self._arg_injector = arg_injector
    self._preamble_injector = preamble_injector

    self.source_fw = config.source_framework
    self.target_fw = config.target_framework
    self.metadata: Dict[str, Any] = {}
    self.current_op_id: Optional[str] = None

  def inject_signature_arg(self, name: str, annotation: Optional[str] = None) -> None:
    """Requests injection of argument into the current function signature."""
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


class AutoWireSpec(BaseModel):
  """
  Schema for plugin self-registration metadata.
  Allows a plugin to define the Semantic Operation it satisfies.
  """

  model_config = ConfigDict(extra="allow")

  ops: Dict[str, Dict[str, Any]] = Field(
    default_factory=dict,
    description="Dictionary of Abstract Operations to inject into SemanticsManager.",
  )


# Updated Type alias to allow arbitrary CSTNodes (e.g. For, Call)
HookFunction = Callable[[Any, HookContext], Any]

# Global Registries
_HOOKS: Dict[str, HookFunction] = {}
_HOOK_METADATA: Dict[str, AutoWireSpec] = {}
_PLUGINS_LOADED = False


def register_hook(trigger: str, auto_wire: Optional[Dict[str, Any]] = None) -> Callable[[HookFunction], HookFunction]:
  """
  Decorator to register a function as a plugin hook.

  Args:
      trigger: The unique identifier. Can be an operation ID or a
               reserved system event like "transform_for_loop".
      auto_wire: Optional dictionary defining the Semantic Spec for this plugin.
                 If provided, the SemanticsManager will automatically load
                 these definitions, eliminating the need for JSON usage.
                 Format matches `semantics/*.json` schema (e.g. `{"ops": {"MyOp": ...}}`).
  """

  def decorator(func: HookFunction) -> HookFunction:
    _HOOKS[trigger] = func
    if auto_wire:
      try:
        spec = AutoWireSpec.model_validate(auto_wire)
        _HOOK_METADATA[trigger] = spec
      except Exception as e:
        print(f"⚠️  Invalid auto_wire spec for hook '{trigger}': {e}")
    return func

  return decorator


def get_hook(trigger: str) -> Optional[HookFunction]:
  """
  Retrieves a registered hook function by its trigger name.
  Lazily loads plugins from the default directory if registry is empty.
  """
  if not _PLUGINS_LOADED:
    load_plugins()
  return _HOOKS.get(trigger)


def get_all_hook_metadata() -> Dict[str, AutoWireSpec]:
  """
  Returns the metadata for all registered hooks.
  Used by SemanticsManager to hydrate the Knowledge Base.
  """
  if not _PLUGINS_LOADED:
    load_plugins()
  return _HOOK_METADATA


def clear_hooks() -> None:
  """Resets the internal hook registry. Primarily for testing."""
  global _PLUGINS_LOADED
  _HOOKS.clear()
  _HOOK_METADATA.clear()
  _PLUGINS_LOADED = False


def load_plugins(plugins_dir: Optional[Path] = None, extra_dirs: Optional[List[Path]] = None) -> int:
  """
  Dynamically imports plugins.

  Args:
      plugins_dir: Overrides default package directory.
                   If provided, this directory is scanned for .py files.
                   If NOT provided, the internal `ml_switcheroo.plugins` package is loaded.
      extra_dirs: Additional directories to scan (e.g. user extensions).

  Returns:
      int: Number of modules loaded.
  """
  global _PLUGINS_LOADED
  total_loaded = 0

  # 1. Load Defaults (Internal Package) if no specific override
  if not _PLUGINS_LOADED and plugins_dir is None:
    try:
      import ml_switcheroo.plugins  # noqa: F401

      _PLUGINS_LOADED = True
      # We just count existing hooks as a proxy for "loaded"
      total_loaded += len(_HOOKS)
    except Exception as e:
      print(f"⚠️  Failed to auto-load default plugins: {e}")

  # 2. Load explicit override directory if provided
  if plugins_dir and plugins_dir.exists() and plugins_dir.is_dir():
    total_loaded += _import_from_dir(plugins_dir, base_package=None)
    _PLUGINS_LOADED = True

  # 3. Load Extra Dirs (User Extensions)
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

    # Try package based import logic first
    if base_package:
      try:
        importlib.import_module(f"{base_package}.{module_name}")
        count += 1
        continue
      except Exception:
        pass

    # Fallback: Load file path directly (for external dirs)
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
