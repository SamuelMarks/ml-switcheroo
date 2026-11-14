"""Rewriter Context Module.

This module provides the `RewriterContext` container, which holds the shared state
for the transpilation pipeline. It decouples state management (Symbol Tables,
Scopes, Configuration) from the logic transformers (RewriterStages), enabling
a composition-based architecture.
"""

import typing

from pydantic import BaseModel, JsonValue

from ml_switcheroo.analysis.symbol_table import SymbolTable
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext, HookRegistry, get_default_hook_registry
from ml_switcheroo.core.rewriter.types import SignatureContext
from ml_switcheroo.core.runtime_models import RuntimeTraceReport
from ml_switcheroo.exceptions import ConfigurationError
from ml_switcheroo.semantics.resolution import query_framework_traits
from ml_switcheroo.semantics.schema import PluginTraits

if typing.TYPE_CHECKING:
  from ml_switcheroo.semantics.manager import SemanticsManager


@typing.runtime_checkable
class _ModelDumpProtocol(typing.Protocol):
  """Protocol for objects providing a `model_dump()` method."""

  def model_dump(self) -> dict[str, JsonValue]:
    """Dump object fields to a dictionary.

    Returns:
        Dictionary of field values.
    """


def _has_model_dump(
  obj: BaseModel | _ModelDumpProtocol | JsonValue,
) -> typing.TypeGuard[_ModelDumpProtocol]:
  """Check if `obj` provides a callable `model_dump` method.

  Args:
      obj: Candidate object.

  Returns:
      True if `obj` has `model_dump`.
  """
  if obj is None or isinstance(obj, (dict, list, str, int, float, bool)):
    return False
  return isinstance(obj, _ModelDumpProtocol)


class SemanticsProtocol(typing.Protocol):
  """Minimal structural protocol for semantics managers in RewriterContext."""


def _is_semantics_mgr(
  obj: "SemanticsManager | SemanticsProtocol | None",
) -> typing.TypeGuard["SemanticsManager"]:
  """Narrow `obj` to `SemanticsManager`.

  Args:
      obj: Semantics manager or mock object.

  Returns:
      True when `obj` is non-None.
  """
  return obj is not None


class RewriterContext:
  """Shared state container for the rewriting pipeline.

  Encapsulates all mutable state required during the AST traversal, allowing
  multiple `RewriterStage` passes to operate on a consistent context.
  """

  def __init__(
    self,
    semantics: "SemanticsManager | SemanticsProtocol | None",
    config: RuntimeConfig | None = None,
    symbol_table: SymbolTable | None = None,
    arg_injector: typing.Callable[[str, str | None], None] | None = None,
    preamble_injector: typing.Callable[[str], None] | None = None,
    runtime_trace: RuntimeTraceReport | None = None,
    hook_registry: HookRegistry | None = None,
  ):
    """Initialize the context.

    Args:
        semantics: The Semantic Knowledge Base manager.
        config: The runtime configuration for the conversion.
        symbol_table: Pre-computed symbol table for type resolution.
        arg_injector: Callback to inject arguments into the current function scope.
        preamble_injector: Callback to inject code blocks into the current scope.
        runtime_trace: Optional on-device forward and backward execution telemetry report.
        hook_registry: Optional isolated ``HookRegistry`` instance for plugin lookup.

    Raises:
        ConfigurationError: If ``semantics`` is ``None``.
    """
    if config is None:
      config = RuntimeConfig()
    if not _is_semantics_mgr(semantics):
      raise ConfigurationError("RewriterContext requires a non-None SemanticsManager instance.")
    self.semantics: "SemanticsManager" = semantics
    self.config = config
    self.symbol_table = symbol_table
    self.runtime_trace = runtime_trace
    self.hook_registry: HookRegistry = hook_registry if hook_registry is not None else get_default_hook_registry()

    # -- Core State --
    # Tracks variable scopes (e.g. stateful variables in classes)
    self.scope_stack: list[set[str]] = [set()]

    # Tracks function signatures being visited (for argument injection)
    self.signature_stack: list[SignatureContext] = []

    # Maps import aliases to canonical names (e.g., 't' -> 'torch')
    self.alias_map: dict[str, str] = {}

    # Accumulates error/warning messages for the current statement
    self.current_stmt_errors: list[str] = []
    self.current_stmt_warnings: list[str] = []

    # Module-level code injection buffer
    self.module_preamble: list[str] = []
    self._satisfied_preamble_injections: set[str] = set()

    # Flags
    self.in_module_class: bool = False

    # -- Helpers --
    # Provide default injectors that mutate context state if caller didn't provide overrides
    # This ensures plugins calling ctx.inject_* work out-of-the-box in standard pipeline/rewriter.
    final_arg_injector = arg_injector if arg_injector else self._default_arg_injector
    final_pre_injector = preamble_injector if preamble_injector else self._default_preamble_injector

    self.hook_context = HookContext(
      semantics=self.semantics,
      config=config,
      arg_injector=final_arg_injector,
      preamble_injector=final_pre_injector,
      symbol_table=symbol_table,
      hook_registry=self.hook_registry,
    )

    # Pre-populate alias map with source defaults
    self._hydrate_source_aliases()

  def _default_arg_injector(self, name: str, annotation: str | None) -> None:
    """Supply callback: Appends to the current signature context.

    Args:
        name: Name of the argument to inject.
        annotation: Optional type annotation for the argument.

    Returns:
        None
    """
    if self.signature_stack:
      # Avoid duplicates in list
      current_ctx = self.signature_stack[-1]
      existing = {n for n, _ in current_ctx.injected_args}
      if name not in existing:
        current_ctx.injected_args.append((name, annotation))

  def _default_preamble_injector(self, code: str) -> None:
    """Supply callback: Injects to function body or module header.

    Args:
        code: The block of code to inject.

    Returns:
        None
    """
    is_import = code.lstrip().startswith("import ") or code.lstrip().startswith("from ")
    if self.signature_stack and not is_import:
      # Inside a function: queue for body injection
      # Dedup based on presence in current stack frame
      current_ctx = self.signature_stack[-1]
      if code not in current_ctx.preamble_stmts:
        current_ctx.preamble_stmts.append(code)
    else:
      # Global scope: queue for module header
      # Check against cache to prevent duplicate injection
      if code not in self._satisfied_preamble_injections:
        self.module_preamble.append(code)
        self._satisfied_preamble_injections.add(code)

  @property
  def source_framework(self) -> str:
    """Return effective source framework string.

    Returns:
        The source framework identifier as a string.
    """
    return str(self.config.effective_source)

  @property
  def target_framework(self) -> str:
    """Return effective target framework string.

    Returns:
        The target framework identifier as a string.
    """
    return str(self.config.effective_target)

  @property
  def plugin_traits(self) -> PluginTraits:
    """Return plugin traits for target framework.

    Returns:
        PluginTraits for the target framework.
    """
    return self.hook_context.plugin_traits

  def _hydrate_source_aliases(self) -> None:
    """Load default aliases for the source framework from semantics config.

    Returns:
        None
    """
    try:
      fw_traits = query_framework_traits(self.semantics, self.source_framework)
      if fw_traits.alias is not None and fw_traits.alias.name:
        self.alias_map[fw_traits.alias.name] = fw_traits.alias.name
        return

      fw_conf = self.semantics.get_framework_config(self.source_framework)
      if not fw_conf:
        return

      alias_raw = fw_conf.get("alias")
      alias_info = alias_raw.model_dump() if _has_model_dump(alias_raw) else alias_raw

      if isinstance(alias_info, dict):
        name = alias_info.get("name")
        if isinstance(name, str) and name:
          self.alias_map[name] = name
    except (KeyError, TypeError, AttributeError, ValueError, RuntimeError) as exc:
      import logging

      logging.getLogger(__name__).debug("Startup alias hydration skipped: %s", exc)


__all__ = ["RewriterContext", "SignatureContext"]
