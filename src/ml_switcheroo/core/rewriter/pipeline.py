"""
Orchestration logic for executing sequential rewriter passes.

This module provides the ``RewriterPipeline``, which manages the sequential
execution of multiple ``RewriterPass`` instances over a shared Context.
"""

from typing import List
import libcst as cst
from ml_switcheroo.core.rewriter.interface import RewriterPass
from ml_switcheroo.core.rewriter.context import RewriterContext


class RewriterPipeline:
  """
  Manages a sequence of rewriting passes and executes them in order.
  """

  def __init__(self, passes: List[RewriterPass]) -> None:
    """
    Initializes the pipeline with a list of passes.

    Args:
        passes: Sequenced list of passes to execute.
    """
    self.passes = passes

  def run(self, module: cst.Module, context: RewriterContext) -> cst.Module:
    """
    Executes all registered passes sequentially on the module.

    Args:
        module: The source AST to transform.
        context: The shared execution state containing semantics and config.

    Returns:
        The fully transformed AST.
    """
    current_module = module
    for pass_instance in self.passes:
      current_module = pass_instance.transform(current_module, context)

    return current_module
