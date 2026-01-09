"""
Rewriter Package.

This package provides the core AST transformation logic defined as a composition
of specialized stages (Structure, API, Auxiliary).

Classes:
- `RewriterPipeline`: The modern context-aware orchestrator.
- `PivotRewriter`: The backward-compatible class aggregating all stages for existing tests.
"""

from typing import Optional, Union

from ml_switcheroo.analysis.symbol_table import SymbolTable
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.rewriter.context import RewriterContext

# Stages
from ml_switcheroo.core.rewriter.calls.mixer import ApiStage, CallMixin
from ml_switcheroo.core.rewriter.structure import StructureStage, StructureMixin
from ml_switcheroo.core.rewriter.control_flow import AuxiliaryStage
from ml_switcheroo.core.rewriter.base import BaseRewriter

# Legacy Mixins exposed for potential external subclassing (Deprecated)
from ml_switcheroo.core.rewriter.attributes import AttributeMixin
from ml_switcheroo.core.rewriter.normalization import NormalizationMixin
from ml_switcheroo.core.rewriter.decorators import DecoratorMixin
from ml_switcheroo.core.rewriter.control_flow import ControlFlowMixin


class PivotRewriter(
  AuxiliaryStage,
  ApiStage,
  StructureStage,
  BaseRewriter,
):
  """
  The main AST transformer for ml-switcheroo.

  This class aggregates the logic from:
  1.  `AuxiliaryStage`: Decorators and Control Flow.
  2.  `ApiStage`: Function Calls, Attributes, Assignments.
  3.  `StructureStage`: Class Inheritance, Function Signatures, Type Hints.
  4.  `BaseRewriter`: Common helpers, Import/Alias tracking, Error Handling.

  It ensures backward compatibility with the legacy initialization signature:
  `PivotRewriter(semantics, config, symbol_table)`.
  """

  def __init__(
    self,
    semantics: Union[SemanticsManager, RewriterContext],
    config: Optional[RuntimeConfig] = None,
    symbol_table: Optional[SymbolTable] = None,
  ):
    """
    Initializes the rewriter.

    Adapts legacy arguments into a `RewriterContext` if necessary.
    """
    # Handle Polymorphic input (Context or Components)
    if isinstance(semantics, RewriterContext):
      context = semantics
    else:
      if config is None:
        # Safe config default if omitted (though typically required)
        config = RuntimeConfig(source_framework="torch", target_framework="jax")
        # raise ValueError("Config required for legacy PivotRewriter initialization")

      # Construct context. We bind internal methods as callbacks here.
      # However, since 'self' isn't fully initialized until super(), we pass None for now
      # and rely on BaseRewriter to bind them, OR bind simple lambdas that delegate to self later.
      # But the RewriterContext holds the hook context which needs the callbacks.
      # A cleaner way: Create context, then set callbacks after super init?
      # HookContext is immutable regarding callbacks.
      # We define wrapper functions.

      context = RewriterContext(
        semantics=semantics,
        config=config,
        symbol_table=symbol_table,
        # We defer binding; BaseRewriter.__init__ will wire up if it handles it,
        # but BaseRewriter creates its OWN context if not provided.
        # Here we provide it. So we must wire it manually.
        arg_injector=lambda n, a: self._callback_inject_arg(n, a),
        preamble_injector=lambda c: self._callback_inject_preamble(c),
      )

    # Initialize all bases with the shared context.
    # super() call targets MRO (AuxiliaryStage).
    # AuxiliaryStage -> RewriterStage -> init(context)
    super().__init__(context)


# Alias for clarity in new code
RewriterPipeline = PivotRewriter

__all__ = [
  "PivotRewriter",
  "RewriterPipeline",
  "RewriterContext",
  "ApiStage",
  "StructureStage",
  "AuxiliaryStage",
  "BaseRewriter",
  "CallMixin",
  "StructureMixin",
  "AttributeMixin",
  "NormalizationMixin",
  "DecoratorMixin",
  "ControlFlowMixin",
]
