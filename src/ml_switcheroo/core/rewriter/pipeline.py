"""Orchestration logic for executing sequential rewriter passes.

This module provides the ``RewriterPipeline``, which manages the sequential
execution of multiple ``RewriterPass`` instances over a shared Context.
"""

import libcst as cst

from ml_switcheroo.core.hooks import bind_hook_registry, get_default_hook_registry
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.interface import RewriterPass


class RewriterPipeline:
  """Manage a sequence of rewriting passes and executes them in order.

  Attributes:
      passes: Ordered sequence of ``RewriterPass`` stages executed by ``run``.
  """

  def __init__(self, passes: list[RewriterPass]) -> None:
    """Initialize the pipeline with a list of passes.

    Args:
        passes: Sequenced list of passes to execute.
    """
    self.passes: list[RewriterPass] = passes

  def run(self, module: cst.Module, context: RewriterContext) -> cst.Module:
    """Execute all registered passes sequentially on the module.

    Args:
        module: The source AST to transform.
        context: The shared execution state containing semantics, config, and hook registry.

    Returns:
        The fully transformed AST.
    """
    try:
      active_reg = context.hook_registry
    except AttributeError:
      active_reg = get_default_hook_registry()

    with bind_hook_registry(active_reg):
      current_module = module
      for pass_instance in self.passes:
        current_module = pass_instance.transform(current_module, context)

      return current_module
