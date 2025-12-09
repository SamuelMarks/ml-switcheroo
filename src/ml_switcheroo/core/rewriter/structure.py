"""
Structure Rewriter Aggregator.

Combines class, function, and type annotation rewriting logic.
Splitting into sub-modules keeps file sizes manageable.
"""

from ml_switcheroo.core.rewriter.structure_class import ClassStructureMixin
from ml_switcheroo.core.rewriter.structure_func import FuncStructureMixin
from ml_switcheroo.core.rewriter.structure_types import TypeStructureMixin


class StructureMixin(ClassStructureMixin, FuncStructureMixin, TypeStructureMixin):
  """
  Composite mixin for all structural rewriting tasks.
  Inherits from granular mixins to assemble the full feature set.
  """
