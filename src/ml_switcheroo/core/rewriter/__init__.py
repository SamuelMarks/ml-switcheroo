"""
Rewriter Package.

This package provides the `PivotRewriter` class, composed of several mixins
to handle specific aspects of the AST transformation:
- Structure: Class and Function definitions (including NNX bidirectional support).
- Calls: Function invocations and operator overloading.
- Attributes: Attribute access and assignment tracking.
- Normalization: Argument mapping and infix rewriting.
- Decorators: Decorator replacement and removal.
"""

from ml_switcheroo.core.rewriter.calls import CallMixin
from ml_switcheroo.core.rewriter.normalization import NormalizationMixin
from ml_switcheroo.core.rewriter.attributes import AttributeMixin
from ml_switcheroo.core.rewriter.structure import StructureMixin
from ml_switcheroo.core.rewriter.decorators import DecoratorMixin
from ml_switcheroo.core.rewriter.base import BaseRewriter


class PivotRewriter(
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
