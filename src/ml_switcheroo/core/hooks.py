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

from typing import Callable, Dict, Optional, Any, Type, TypeVar, List
from pydantic import BaseModel, Field, ConfigDict

# We import RuntimeConfig for type hinting
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.dsl import FrameworkVariant
from ml_switcheroo.semantics.schema import PluginTraits

# Lazy import to avoid circular dependency
# TYPE_CHECKING block logic or Any is sufficient for runtime
SymbolTableType = Any
SemanticsManagerType = Any
T = TypeVar("T", bound=BaseModel)

# Callbacks for plugin side-effects
ArgInjectorType = Callable[[str, Optional[str]], None]
PreambleInjectorType = Callable[[str], None]


class AutoWireSpec(BaseModel):
  """Schema for plugin self-registration metadata.


  Allows a plugin to define the Semantic Operation it satisfies.
  """

  model_config = ConfigDict(extra="allow")

  ops: Dict[str, Dict[str, Any]] = Field(
    default_factory=dict,
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
    semantics: SemanticsManagerType,
    config: RuntimeConfig,
    arg_injector: Optional[ArgInjectorType] = None,
    preamble_injector: Optional[PreambleInjectorType] = None,
    symbol_table: Optional[SymbolTableType] = None,
  ):
    """Initializes the hook context.

    Args:
        semantics: Reference to the SemanticsManager.
        config: Runtime configuration (strict mode, selected frameworks).
        arg_injector: Callback to inject arguments into function signature.
        preamble_injector: Callback to inject code at top of function.
        symbol_table: Pre-calculated Symbol Table for type resolution.

    """
    self.semantics = semantics
    self._runtime_config = config
    self._arg_injector = arg_injector
    self._preamble_injector = preamble_injector
    self._symbol_table = symbol_table

    self.source_fw = config.effective_source
    self.target_fw = config.effective_target

    # Plugin State
    self.metadata: Dict[str, Any] = {}
    self.current_op_id: Optional[str] = None

  def resolve_type(self, node: Any) -> Optional[str]:
    """Queries the Symbol Table for the inferred type of a node.

    Args:
        node: The LibCST node to inspect.

    Returns:
        str: "Tensor" if it's a tensor, "Module" if module, or None.

    """
    if not self._symbol_table:
      return None

    sym = self._symbol_table.get_type(node)  # pragma: no cover
    if not sym:  # pragma: no cover
      return None  # pragma: no cover
    # pragma: no cover
    # Return simple string type indicator  # pragma: no cover
    if "Tensor" in sym.name:  # pragma: no cover
      return "Tensor"  # pragma: no cover
    if "Module" in sym.name:  # pragma: no cover
      return "Module"  # pragma: no cover
    return sym.name  # pragma: no cover

  @property
  def plugin_traits(self) -> PluginTraits:
    """Returns the capabilities of the current Target Framework.

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
  def current_variant(self) -> Optional[FrameworkVariant]:
    """Returns the Variant definition for the current operation/target.

    Allows plugins to read extra metadata defined in the JSON (e.g. pack_to_tuple).

    Returns:
        Optional[FrameworkVariant]: The variant definition if resolved, else None.

    """
    if not self.semantics or not self.current_op_id:
      return None  # pragma: no cover

    # Access definition
    # Use low-level retrieval to avoid recursion
    data = self.semantics.resolve_variant(self.current_op_id, self.target_fw)
    if not data:
      return None

    return FrameworkVariant.model_validate(data)

  def inject_signature_arg(self, name: str, annotation: Optional[str] = None) -> None:
    """Requests injection of argument into the current function signature.

    Args:
        name (str): The name of the argument to inject.
        annotation (Optional[str]): Type hint string for the argument.

    """
    if self._arg_injector:
      self._arg_injector(name, annotation)

  def inject_preamble(self, code_str: str) -> None:
    """Requests injection of a statement at the beginning of the function body.

    Args:
        code_str (str): Python source code string to inject.

    """
    if self._preamble_injector:
      self._preamble_injector(code_str)

  def raw_config(self, key: str, default: Any = None) -> Any:
    """Retrieve a raw value from the unstructured plugin settings dict.

    Args:
        key (str): Configuration key.
        default (Any): Default value if key is not found.

    Returns:
        Any: The configuration value.

    """
    return self._runtime_config.plugin_settings.get(key, default)

  def validate_settings(self, model: Type[T]) -> T:
    """Validates global config against a Plugin-specific Pydantic schema.

    Args:
        model (Type[T]): Pydantic model definition.

    Returns:
        T: Validated configuration object.

    """
    relevant = model.model_fields.keys()
    subset = {k: v for k, v in self._runtime_config.plugin_settings.items() if k in relevant}
    return model.model_validate(subset)

  def lookup_api(self, op_name: str) -> Optional[str]:
    """Resolves target framework's API string for a given standard operation.

    Args:
        op_name (str): Standard operation ID.

    Returns:
        Optional[str]: The target API string, or None if not found.

    """
    if not self.semantics:
      return None  # pragma: no cover

    # Use the inheritance-aware resolve_variant method
    # instead of direct dict access to support child frameworks (e.g. flax_nnx -> jax mapping)
    target_variant = self.semantics.resolve_variant(op_name, self.target_fw)

    if not target_variant:
      return None

    return target_variant.get("api")

  def lookup_signature(self, op_name: str) -> List[str]:
    """Retrieves standard argument list for a given operation.

    Args:
        op_name (str): Standard operation ID.

    Returns:
        List[str]: List of argument names.

    """
    if not self.semantics:
      return []  # pragma: no cover
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
        name = item.get("name")  # pragma: no cover
        if name:  # pragma: no cover
          cleaned_args.append(name)  # pragma: no cover
      else:
        cleaned_args.append(item)
    return cleaned_args


from ml_switcheroo.core.hooks_registry import (  # noqa: E402
  register_hook,
  get_hook,
  get_all_hook_metadata,
  clear_hooks,
  load_plugins,
  _HOOKS,
  _HOOK_METADATA,
)

HookFunction = Callable[[Any, HookContext], Any]

_PLUGINS_LOADED = False
__all__ = [
  "HookContext",
  "AutoWireSpec",
  "register_hook",
  "get_hook",
  "get_all_hook_metadata",
  "clear_hooks",
  "load_plugins",
  "_HOOKS",
  "_HOOK_METADATA",
]
