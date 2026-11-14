"""Plugin Binding Infrastructure.

This module provides the ``HookContext`` passed to plugin hooks during AST
transpilation and re-exports ``HookRegistry`` registration and lookup helpers.
"""

from __future__ import annotations

import typing

import libcst as cst
from pydantic import BaseModel, JsonValue

from ml_switcheroo.analysis.symbol_table import SymbolTable as SymbolTableType
from ml_switcheroo.config import RuntimeConfig as RuntimeConfigType
from ml_switcheroo.core.dsl import FrameworkVariant, ParameterDef
from ml_switcheroo.core.hooks_registry import (
  AutoWireSpec,
  HookCallable,
  HookRegistry,
  bind_hook_registry,
  clear_hooks,
  get_all_hook_metadata,
  get_default_hook_registry,
  get_hook,
  load_plugins,
  register_hook,
  set_plugins_loaded,
)
from ml_switcheroo.semantics.manager import SemanticsManager as SemanticsManagerType
from ml_switcheroo.semantics.resolution import (
  query_framework_traits,
  query_operation,
  query_variant,
)
from ml_switcheroo.semantics.schema import PluginTraits

T = typing.TypeVar("T", bound=BaseModel)

ArgInjectorType = typing.Callable[[str, str | None], None]
PreambleInjectorType = typing.Callable[[str], None]
PluginMetadataValue: typing.TypeAlias = (
  JsonValue
  | cst.CSTNode
  | dict[str, "PluginMetadataValue"]
  | dict[str, dict[str, cst.BaseExpression]]
  | dict[str, cst.BaseExpression]
)

HookFunction = HookCallable


class HookContext:
  """Context object passed to every plugin hook during transcoding.

  Provides read-only access to semantic/config state and controlled write access
  to function signature argument and preamble injection points.

  Attributes:
      semantics: Reference to the ``SemanticsManager`` (or ``None``).
      hook_registry: Active ``HookRegistry`` associated with this conversion run.
      source_framework: Effective source framework identifier.
      target_framework: Effective target framework identifier.
      metadata: Per-conversion mutable plugin state dictionary.
      current_op_id: Currently active abstract operation identifier, if any.
  """

  semantics: SemanticsManagerType | None = None
  current_op_id: str | None = None
  source_framework: str = ""
  target_framework: str = ""

  def __init__(
    self,
    semantics: SemanticsManagerType | None,
    config: RuntimeConfigType,
    arg_injector: ArgInjectorType | None = None,
    preamble_injector: PreambleInjectorType | None = None,
    symbol_table: SymbolTableType | None = None,
    hook_registry: HookRegistry | None = None,
  ) -> None:
    """Initialize the hook context.

    Args:
        semantics: Reference to the ``SemanticsManager``.
        config: Runtime configuration (strict mode, selected frameworks).
        arg_injector: Callback to inject arguments into function signature.
        preamble_injector: Callback to inject code at top of function.
        symbol_table: Pre-calculated Symbol Table for type resolution.
        hook_registry: Optional explicit ``HookRegistry`` instance.
    """
    self.semantics: SemanticsManagerType | None = semantics
    self._runtime_config = config
    self._arg_injector = arg_injector
    self._preamble_injector = preamble_injector
    self._symbol_table = symbol_table
    self.hook_registry: HookRegistry = hook_registry if hook_registry is not None else get_default_hook_registry()

    self.source_framework = config.effective_source
    self.target_framework = config.effective_target

    self.metadata: dict[str, PluginMetadataValue] = {}
    self.current_op_id: str | None = None

  def resolve_type(self, node: cst.CSTNode) -> str | None:
    """Query the Symbol Table for the inferred type of a node.

    Args:
        node: The LibCST node to inspect.

    Returns:
        ``"Tensor"`` if tensor, ``"Module"`` if module, symbol name string, or ``None``.
    """
    if not self._symbol_table:
      return None

    sym = self._symbol_table.get_type(node)
    if not sym:
      return None

    if "Tensor" in sym.name:
      return "Tensor"
    if "Module" in sym.name:
      return "Module"
    return str(sym.name)

  @property
  def plugin_traits(self) -> PluginTraits:
    """Return the capabilities of the current Target Framework.

    Returns:
        Capability flags for the target framework.
    """
    if self.semantics is None:
      return PluginTraits()

    fw_traits = query_framework_traits(self.semantics, self.target_framework)
    return fw_traits.plugin_traits if fw_traits.plugin_traits is not None else PluginTraits()

  @property
  def current_variant(self) -> FrameworkVariant | None:
    """Return the ``FrameworkVariant`` definition for the current operation and target.

    Returns:
        The variant definition if resolved, else ``None``.
    """
    if self.semantics is None or not self.current_op_id:
      return None

    return query_variant(self.semantics, self.current_op_id, self.target_framework)

  def inject_signature_arg(self, name: str, annotation: str | None = None) -> None:
    """Request injection of an argument into the current function signature.

    Args:
        name: Name of the argument to inject.
        annotation: Optional type hint string for the argument.
    """
    if self._arg_injector:
      self._arg_injector(name, annotation)

  def inject_preamble(self, code_str: str) -> None:
    """Request injection of a statement at the beginning of the function body.

    Args:
        code_str: Python source code string to inject.
    """
    if self._preamble_injector:
      self._preamble_injector(code_str)

  def raw_config(self, key: str, default: JsonValue = None) -> JsonValue:
    """Retrieve a raw value from the unstructured plugin settings dict.

    Args:
        key: Configuration key.
        default: Default value if key is not found.

    Returns:
        The configuration value.
    """
    res: JsonValue = self._runtime_config.plugin_settings.get(key, default)
    return res

  def validate_settings(self, model: type[T]) -> T:
    """Validate global plugin settings against a plugin-specific Pydantic schema.

    Args:
        model: Pydantic model definition.

    Returns:
        Validated configuration object.
    """
    relevant = model.model_fields.keys()
    subset = {k: v for k, v in self._runtime_config.plugin_settings.items() if k in relevant}
    return model.model_validate(subset)

  def lookup_api(self, op_name: str) -> str | None:
    """Resolve target framework's API string for a given standard operation.

    Args:
        op_name: Standard operation ID.

    Returns:
        Target API string, or ``None`` if not found.
    """
    if self.semantics is None:
      return None

    typed_variant = query_variant(self.semantics, op_name, self.target_framework)
    return typed_variant.api if typed_variant is not None else None

  def lookup_signature(self, op_name: str) -> list[str]:
    """Retrieve standard argument list for a given operation.

    Args:
        op_name: Standard operation ID.

    Returns:
        List of argument names.
    """
    if self.semantics is None:
      return []
    op_model = query_operation(self.semantics, op_name)
    if op_model is not None:
      return [item.name if isinstance(item, ParameterDef) else item for item in op_model.std_args]

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
        name = item.get("name")
        if isinstance(name, str) and name:
          cleaned_args.append(name)
      else:
        cleaned_args.append(str(item))
    return cleaned_args


__all__ = [
  "AutoWireSpec",
  "FrameworkVariant",
  "HookCallable",
  "HookContext",
  "HookFunction",
  "HookRegistry",
  "PluginMetadataValue",
  "PluginTraits",
  "bind_hook_registry",
  "clear_hooks",
  "get_all_hook_metadata",
  "get_default_hook_registry",
  "get_hook",
  "load_plugins",
  "register_hook",
  "set_plugins_loaded",
]
