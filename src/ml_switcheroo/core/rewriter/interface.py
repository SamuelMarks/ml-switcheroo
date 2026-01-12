"""
Interface definition for Rewriter Passes.

This module defines the abstract base class that all transformation passes
must implement to be compatible with the ``RewriterPipeline``.
"""

from abc import ABC, abstractmethod
import libcst as cst
from ml_switcheroo.core.rewriter.context import RewriterContext


class RewriterPass(ABC):
  """
  Abstract contract for a transformation pass in the rewriting pipeline.

  Passes encapsulate discrete transformation logic (e.g. Structural Rewriting,
  API Remapping) and are executed sequentially by the pipeline.
  """

  @abstractmethod
  def transform(self, module: cst.Module, context: RewriterContext) -> cst.Module:
    """
    Executes the transformation logic on the given CST module.

    Args:
        module: The input LibCST module.
        context: The shared rewriter context containing configuration and state.

    Returns:
        The transformed LibCST module.
    """
    pass
