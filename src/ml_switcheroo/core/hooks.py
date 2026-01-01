"""
Plugin Binding Infrastructure.

This module provides the infrastructure for extending ml-switcheroo via plugins.
It enables developers to intercept and modify the Abstract Syntax Tree (AST)
during the conversion process using a hook-based system.

Refactor:
    - Added `auto_wire` support to the `register_hook` decorator.
    - Plugins can now declare their own Semantic definitions.
    - HookContext now exposes `plugin_traits` and `current_variant` for data-driven logic.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Any, Type, TypeVar, List, Union
from pydantic import BaseModel, Field, ConfigDict

# We import RuntimeConfig for type hinting
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.schema import PluginTraits, Variant

# Validation that SemanticsManager is importable for type hinting
SemanticsManagerType = Any
T = TypeVar("T", bound=BaseModel)

# Callbacks for plugin side-effects
ArgInjectorType = Callable[[str, Optional[str]], None]
PreambleInjectorType = Callable[[str], None]


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
class HookContext:
  """
  Context object passed to every plugin hook during transcoding.

  Provides read-only access to global state and write access
  to specific injection points (signature args, function body preambles).
  Now exposes `plugin_traits` and `current_variant` for data-driven decisions.
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

    # Plugin State
    self.metadata: Dict[str, Any] = {}
    self.current_op_id: Optional[str] = None

  @property
  def plugin_traits(self) -> PluginTraits:
    """
    Returns the capabilities of the current Target Framework.
    This allows plugins to check functionality (e.g. has_numpy_compatible_arrays)
    rather than checking the framework name string.

    Returns:
        PluginTraits: The capability flags for the target framework.
    """
    if not self.semantics:
      return PluginTraits()

    conf = self.semantics.get_framework_config(self.target_fw)
    if not conf:
      return PluginTraits()

    # Handle dict or Pydantic object
    traits = conf.get("plugin_traits")
    if not traits:
      return PluginTraits()

    if isinstance(traits, dict):
      return PluginTraits.model_validate(traits)
    if isinstance(traits, PluginTraits):
      return traits

    return PluginTraits()

  @property
  def current_variant(self) -> Optional[Variant]:
    """
    Returns the Variant definition for the current operation/target.
    Allows plugins to read extra metadata defined in the JSON (e.g. pack_to_tuple).

    Returns:
        Optional[Variant]: The variant definition if resolved, else None.
    """
    if not self.semantics or not self.current_op_id:
      return None

    # Access definition
    # Use low-level retrieval to avoid recursion
    data = self.semantics.resolve_variant(self.current_op_id, self.target_fw)
    if not data:
      return None

    return Variant.model_validate(data)

  def inject_signature_arg(self, name: str, annotation: Optional[str] = None) -> None:
    """
    Requests injection of argument into the current function signature.

    Args:
        name (str): The name of the argument to inject.
        annotation (Optional[str]): Type hint string for the argument.
    """
    if self._arg_injector:
      self._arg_injector(name, annotation)

  def inject_preamble(self, code_str: str) -> None:
    """
    Requests injection of a statement at the beginning of the function body.

    Args:
        code_str (str): Python source code string to inject.
    """
    if self._preamble_injector:
      self._preamble_injector(code_str)

  def raw_config(self, key: str, default: Any = None) -> Any:
    """
    Retrieve a raw value from the unstructured plugin settings dict.

    Args:
        key (str): Configuration key.
        default (Any): Default value if key is not found.

    Returns:
        Any: The configuration value.
    """
    return self._runtime_config.plugin_settings.get(key, default)

  def config(self, key: str, default: Any = None) -> Any:
    """
    Legacy alias for raw_config.

    Args:
        key (str): Configuration key.
        default (Any): Default value.

    Returns:
        Any: The configuration value.
    """
    return self.raw_config(key, default)

  def validate_settings(self, model: Type[T]) -> T:
    """
    Validates global config against a Plugin-specific Pydantic schema.

    Args:
        model (Type[T]): Pydantic model definition.

    Returns:
        T: Validated configuration object.
    """
    relevant = model.model_fields.keys()
    subset = {k: v for k, v in self._runtime_config.plugin_settings.items() if k in relevant}
    return model.model_validate(subset)

  def lookup_api(self, op_name: str) -> Optional[str]:
    """
    Resolves target framework's API string for a given standard operation.

    Args:
        op_name (str): Standard operation ID.

    Returns:
        Optional[str]: The target API string, or None if not found.
    """
    if not self.semantics:
      return None

    # Use the inheritance-aware resolve_variant method
    # instead of direct dict access to support child frameworks (e.g. flax_nnx -> jax mapping)
    target_variant = self.semantics.resolve_variant(op_name, self.target_fw)

    if not target_variant:
      return None

    return target_variant.get("api")

  def lookup_signature(self, op_name: str) -> List[str]:
    """
    Retrieves standard argument list for a given operation.

    Args:
        op_name (str): Standard operation ID.

    Returns:
        List[str]: List of argument names.
    """
    if not self.semantics:
      return []
    # get_definition_by_id checks main data store
    details = self.semantics.get_definition_by_id(op_name)
    if not details:
      return []
    std_args = details.get("std_args", [])
    cleaned_args = []
    for item in std_args:
      if isinstance(item, (list, tuple)):
        cleaned_args.append(item[0])
      elif isinstance(item, dict):
        # Handle ParameterDef dict or object
        name = item.get("name")
        if name:
          cleaned_args.append(name)
      else:
        cleaned_args.append(item)
    return cleaned_args


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

  Returns:
      Callable: The decorator wrapper.
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

  Args:
      trigger (str): Hook identifier key.

  Returns:
      Optional[HookFunction]: The registered function or None.
  """
  if not _PLUGINS_LOADED:
    load_plugins()
  return _HOOKS.get(trigger)


def get_all_hook_metadata() -> Dict[str, AutoWireSpec]:
  """
  Returns the metadata for all registered hooks.
  Used by SemanticsManager to hydrate the Knowledge Base.

  Returns:
      Dict[str, AutoWireSpec]: Metadata for autowired plugins.
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
  if not _PLUGINS_LOADED and not plugins_dir:
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
  """
  Helper to iterate and import python files from a directory.

  Args:
      directory (Path): The directory to scan.
      base_package (Optional[str]): Base python package name if loading via importlib.

  Returns:
      int: Count of successfully imported modules.
  """
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
