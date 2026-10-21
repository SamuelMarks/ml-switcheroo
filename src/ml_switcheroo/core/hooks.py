"""Plugin Binding Infrastructure.

This module provides the infrastructure for extending ml-switcheroo via plugins.
It enables developers to intercept and modify the Abstract Syntax Tree (AST)
during the conversion process using a hook-based system.

Refactor:
    - Added `auto_wire` support to the `register_hook` decorator.
    - Plugins can now declare their own Semantic definitions.
    - HookContext now exposes `plugin_traits` and `current_variant` for data-driven logic.
    - **New**: `resolve_type` method to query Symbol Table.
"""

from __future__ import annotations  # noqa: E402

import sys  # noqa: E402
import types  # noqa: E402
import typing  # noqa: E402
from pydantic import BaseModel, Field, ConfigDict, JsonValue  # noqa: E402
import libcst as cst  # noqa: E402

# We import RuntimeConfig for type hinting
from ml_switcheroo.core.dsl import FrameworkVariant  # noqa: E402
from ml_switcheroo.semantics.schema import PluginTraits  # noqa: E402

# Lazy import to avoid circular dependency
if typing.TYPE_CHECKING:
  from ml_switcheroo.analysis.symbol_table import SymbolTable as SymbolTableType
  from ml_switcheroo.config import RuntimeConfig as RuntimeConfigType
  from ml_switcheroo.semantics.manager import SemanticsManager as SemanticsManagerType
else:
  SymbolTableType = type
  RuntimeConfigType = type
  SemanticsManagerType = type

T = typing.TypeVar("T", bound=BaseModel)

# Callbacks for plugin side-effects
ArgInjectorType = typing.Callable[[str, str | None], None]
PreambleInjectorType = typing.Callable[[str], None]


class AutoWireSpec(BaseModel):
  """Schema for plugin self-registration metadata.

  Allows a plugin to define the Semantic Operation it satisfies.
  """

  model_config = ConfigDict(extra="allow")

  ops: dict[str, JsonValue] | None = Field(
    default_factory=lambda: {},
    description="Dictionary of Abstract Operations to inject into SemanticsManager.",
  )


# Updated Type alias to allow arbitrary CSTNodes (e.g. For, Call)
class HookContext:
  """Context object passed to every plugin hook during transcoding.

  Provides read-only access to global state and write access
  to specific injection points (signature args, function body preambles).
  Now exposes `plugin_traits` and `current_variant` for data-driven decisions.
  """

  def __init__(
    self,
    semantics: SemanticsManagerType | None,
    config: RuntimeConfigType,
    arg_injector: ArgInjectorType | None = None,
    preamble_injector: PreambleInjectorType | None = None,
    symbol_table: SymbolTableType | None = None,
  ):
    """Initialize the hook context.

    Args:
        semantics: Reference to the SemanticsManager.
        config: Runtime configuration (strict mode, selected frameworks).
        arg_injector: Callback to inject arguments into function signature.
        preamble_injector: Callback to inject code at top of function.
        symbol_table: Pre-calculated Symbol Table for type resolution.

    """
    self.semantics: SemanticsManagerType | None = semantics
    self._runtime_config = config
    self._arg_injector = arg_injector
    self._preamble_injector = preamble_injector
    self._symbol_table = symbol_table

    self.source_framework = config.effective_source
    self.target_framework = config.effective_target

    # Plugin State
    self.metadata: dict[str, JsonValue] = {}
    self.current_op_id: str | None = None

  def resolve_type(self, node: cst.CSTNode) -> str | None:
    """Query the Symbol Table for the inferred type of a node.

    Args:
        node: The LibCST node to inspect.

    Returns:
        str: "Tensor" if it's a tensor, "Module" if module, or None.

    """
    if not self._symbol_table:
      return None

    sym = self._symbol_table.get_type(node)
    if not sym:
      return None

    # Return simple string type indicator
    if "Tensor" in sym.name:
      return "Tensor"
    if "Module" in sym.name:
      return "Module"
    return str(sym.name)

  @property
  def plugin_traits(self) -> PluginTraits:
    """Return the capabilities of the current Target Framework.

    This allows plugins to check functionality (e.g. has_numpy_compatible_arrays)
    rather than checking the framework name string.

    Returns:
        PluginTraits: The capability flags for the target framework.

    """
    if self.semantics is None:
      return PluginTraits()

    conf = self.semantics.get_framework_config(self.target_framework)
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
  def current_variant(self) -> FrameworkVariant | None:
    """Return the Variant definition for the current operation/target.

    Allows plugins to read extra metadata defined in the JSON (e.g. pack_to_tuple).

    Returns:
        Optional[FrameworkVariant]: The variant definition if resolved, else None.

    """
    if self.semantics is None or not self.current_op_id:
      return None

    # Access definition
    # Use low-level retrieval to avoid recursion
    data = self.semantics.resolve_variant(self.current_op_id, self.target_framework)
    if not data:
      return None

    return FrameworkVariant.model_validate(data)

  def inject_signature_arg(self, name: str, annotation: str | None = None) -> None:
    """Request injection of argument into the current function signature.

    Args:
        name (str): The name of the argument to inject.
        annotation (Optional[str]): Type hint string for the argument.

    """
    if self._arg_injector:
      self._arg_injector(name, annotation)

  def inject_preamble(self, code_str: str) -> None:
    """Request injection of a statement at the beginning of the function body.

    Args:
        code_str (str): Python source code string to inject.

    """
    if self._preamble_injector:
      self._preamble_injector(code_str)

  def raw_config(self, key: str, default: JsonValue = None) -> JsonValue:
    """Retrieve a raw value from the unstructured plugin settings dict.

    Args:
        key (str): Configuration key.
        default: Default value if key is not found.

    Returns:
        The configuration value.

    """
    res: JsonValue = self._runtime_config.plugin_settings.get(key, default)
    return res

  def validate_settings(self, model: type[T]) -> T:
    """Validate global config against a Plugin-specific Pydantic schema.

    Args:
        model (Type[T]): Pydantic model definition.

    Returns:
        T: Validated configuration object.

    """
    relevant = model.model_fields.keys()
    subset = {k: v for k, v in self._runtime_config.plugin_settings.items() if k in relevant}
    return model.model_validate(subset)

  def lookup_api(self, op_name: str) -> str | None:
    """Resolve target framework's API string for a given standard operation.

    Args:
        op_name (str): Standard operation ID.

    Returns:
        Optional[str]: The target API string, or None if not found.

    """
    if self.semantics is None:
      return None

    # Use the inheritance-aware resolve_variant method
    # instead of direct dict access to support child frameworks (e.g. flax_nnx -> jax mapping)
    target_variant = self.semantics.resolve_variant(op_name, self.target_framework)

    if not target_variant:
      return None

    api_val = target_variant.get("api")
    return str(api_val) if api_val is not None else None

  def lookup_signature(self, op_name: str) -> list[str]:
    """Retrieve standard argument list for a given operation.

    Args:
        op_name (str): Standard operation ID.

    Returns:
        List[str]: List of argument names.

    """
    if self.semantics is None:
      return []
    # get_definition_by_id checks main data store
    details = self.semantics.get_definition_by_id(op_name)
    if not details:
      return []
    std_args = details.get("std_args", [])
    if not isinstance(std_args, list):
      return []
    cleaned_args: list[str] = []
    for item in std_args:
      if isinstance(item, (list, tuple)) and item:
        cleaned_args.append(str(item[0]))
      elif isinstance(item, dict):
        # Handle ParameterDef dict or object
        name = item.get("name")
        if isinstance(name, str) and name:
          cleaned_args.append(name)
      else:
        cleaned_args.append(str(item))
    return cleaned_args


import ml_switcheroo.core.hooks_registry as _hr  # noqa: E402
from ml_switcheroo.core.hooks_registry import (  # noqa: E402
  register_hook,
  get_hook,
  get_all_hook_metadata,
  clear_hooks,
  load_plugins,
  _HOOKS,
  _HOOK_METADATA,
  HookCallable,
)

HookFunction = HookCallable

_PLUGINS_LOADED: bool = False


class _HooksModule(types.ModuleType):
  """Dynamic module type ensuring synchronization with hooks_registry."""

  @property
  def _PLUGINS_LOADED(self) -> bool:
    """Indicate whether plugins have been loaded in the hook registry.

    Returns:
        bool: True if plugins have been loaded, False otherwise.
    """
    return _hr._PLUGINS_LOADED

  @_PLUGINS_LOADED.setter
  def _PLUGINS_LOADED(self, value: bool) -> None:
    """Set the plugin loading state in the hook registry.

    Args:
        value: The new plugin loaded state.
    """
    _hr._PLUGINS_LOADED = value


sys.modules[__name__].__class__ = _HooksModule

__all__ = [
  "HookContext",
  "AutoWireSpec",
  "PluginTraits",
  "register_hook",
  "get_hook",
  "get_all_hook_metadata",
  "clear_hooks",
  "load_plugins",
  "_HOOKS",
  "_HOOK_METADATA",
  "_PLUGINS_LOADED",
]
