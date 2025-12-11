"""
Rewriter Package.

This package provides the `PivotRewriter` class, composed of several mixins
to handle specific aspects of the AST transformation:
- Structure: Class and Function definitions.
- Calls: Function invocations.
- Attributes: Attribute access.
- Normalization: Argument mapping.
- ControlFlow: Loop and branching logic.
- Decorators: Decorator handling.
"""

from ml_switcheroo.core.rewriter.calls import CallMixin
from ml_switcheroo.core.rewriter.normalization import NormalizationMixin
from ml_switcheroo.core.rewriter.attributes import AttributeMixin
from ml_switcheroo.core.rewriter.structure import StructureMixin
from ml_switcheroo.core.rewriter.decorators import DecoratorMixin
from ml_switcheroo.core.rewriter.control_flow import ControlFlowMixin
from ml_switcheroo.core.rewriter.base import BaseRewriter


class PivotRewriter(
  ControlFlowMixin,
  DecoratorMixin,
  CallMixin,
  NormalizationMixin,
  AttributeMixin,
  StructureMixin,
  BaseRewriter,
):
  """
  The main AST transformer for ml-switcheroo.

  Inherits functionality from component identifiers (Mixins) and the base
  transformer logic. This class is the entry point for the ASTEngine.
  """

  pass
