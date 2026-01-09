"""
Type Hint Rewriting Logic.

Handles transformation of:
1. Function Parameter Annotations.
2. Return Type Annotations.
3. Annotated Assignments (Variable Declarations).
"""

from typing import Optional, TYPE_CHECKING
import libcst as cst

if TYPE_CHECKING:
  from ml_switcheroo.core.rewriter.structure import StructureStage


class TypeStructureMixin:
  """
  Mixin for transforming Type Annotations.

  Attributes:
      _in_annotation (bool): Tracks if the visitor is inside an annotation.
  """

  _in_annotation: bool = False

  def visit_Annotation(self: "StructureStage", node: cst.Annotation) -> Optional[bool]:
    """
    Enters a type annotation node.
    """
    self._in_annotation = True
    return True

  def leave_Annotation(
    self: "StructureStage", original_node: cst.Annotation, updated_node: cst.Annotation
  ) -> cst.Annotation:
    """
    Leaves a type annotation node.
    """
    self._in_annotation = False
    return updated_node

  def leave_Name(self: "StructureStage", original_node: cst.Name, updated_node: cst.Name) -> cst.BaseExpression:
    """
    Rewrites Names found within Type Annotations using alias resolution and mapping lookup.
    """
    if getattr(self, "_in_annotation", False):
      full_name = self._get_qualified_name(original_node)
      if full_name:
        mapping = self._get_mapping(full_name, silent=True)
        if mapping and "api" in mapping:
          return self._create_name_node(mapping["api"])

    return updated_node
