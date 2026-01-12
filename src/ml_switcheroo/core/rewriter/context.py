"""
Rewriter Context Module.

This module provides the `RewriterContext` container, which holds the shared state
for the transpilation pipeline. It decouples state management (Symbol Tables,
Scopes, Configuration) from the logic transformers (RewriterStages), enabling
a composition-based architecture.
"""

from typing import Dict, List, Optional, Set, Callable, Union

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.analysis.symbol_table import SymbolTable
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.core.rewriter.types import SignatureContext


class RewriterContext:
  """
  Shared state container for the rewriting pipeline.

  Encapsulates all mutable state required during the AST traversal, allowing
  multiple `RewriterStage` passes to operate on a consistent context.
  """

  def __init__(
    self,
    semantics: SemanticsManager,
    config: RuntimeConfig,
    symbol_table: Optional[SymbolTable] = None,
    arg_injector: Optional[Callable[[str, Optional[str]], None]] = None,
    preamble_injector: Optional[Callable[[str], None]] = None,
  ):
    """
    Initializes the context.

    Args:
        semantics: The Semantic Knowledge Base manager.
        config: The runtime configuration for the conversion.
        symbol_table: Pre-computed symbol table for type resolution.
        arg_injector: Callback to inject arguments into the current function scope.
        preamble_injector: Callback to inject code blocks into the current scope.
    """
    self.semantics = semantics
    self.config = config
    self.symbol_table = symbol_table

    # -- Core State --
    # Tracks variable scopes (e.g. stateful variables in classes)
    self.scope_stack: List[Set[str]] = [set()]

    # Tracks function signatures being visited (for argument injection)
    self.signature_stack: List[SignatureContext] = []

    # Maps import aliases to canonical names (e.g., 't' -> 'torch')
    self.alias_map: Dict[str, str] = {}

    # Accumulates error/warning messages for the current statement
    self.current_stmt_errors: List[str] = []
    self.current_stmt_warnings: List[str] = []

    # Module-level code injection buffer
    self.module_preamble: List[str] = []
    self._satisfied_preamble_injections: Set[str] = set()

    # Flags
    self.in_module_class: bool = False

    # -- Helpers --
    # Provide default injectors that mutate context state if caller didn't provide overrides
    # This ensures plugins calling ctx.inject_* work out-of-the-box in standard pipeline/rewriter.
    final_arg_injector = arg_injector if arg_injector else self._default_arg_injector
    final_pre_injector = preamble_injector if preamble_injector else self._default_preamble_injector

    self.hook_context = HookContext(
      semantics=semantics,
      config=config,
      arg_injector=final_arg_injector,
      preamble_injector=final_pre_injector,
      symbol_table=symbol_table,
    )

    # Pre-populate alias map with source defaults
    self._hydrate_source_aliases()

  def _default_arg_injector(self, name: str, annotation: Optional[str]) -> None:
    """Default callback: Appends to the current signature context."""
    if self.signature_stack:
      # Avoid duplicates in list
      current_ctx = self.signature_stack[-1]
      existing = {n for n, _ in current_ctx.injected_args}
      if name not in existing:
        current_ctx.injected_args.append((name, annotation))

  def _default_preamble_injector(self, code: str) -> None:
    """Default callback: Injects to function body or module header."""
    if self.signature_stack:
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
  def source_fw(self) -> str:
    """Returns effective source framework string."""
    return str(self.config.effective_source)

  @property
  def target_fw(self) -> str:
    """Returns effective target framework string."""
    return str(self.config.effective_target)

  def _hydrate_source_aliases(self) -> None:
    """Loads default aliases for the source framework from semantics config."""
    try:
      fw_conf = self.semantics.get_framework_config(self.source_fw)
      if not fw_conf:
        return

      alias_info = fw_conf.get("alias")
      # Handle Pydantic model dump or dict
      if hasattr(alias_info, "model_dump"):
        alias_info = alias_info.model_dump()

      if isinstance(alias_info, dict):
        name = alias_info.get("name")
        if name:
          self.alias_map[name] = name
    except Exception:
      # Ignore startup hydration errors if config is partial
      pass
