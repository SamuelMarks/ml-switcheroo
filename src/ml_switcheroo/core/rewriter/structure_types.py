"""
Type Hint Rewriting Logic.

Handles transformation of:
1. Function Parameter Annotations.
2. Return Type Annotations.
3. Annotated Assignments (Variable Declarations).
"""

from typing import Optional
import libcst as cst

from ml_switcheroo.core.rewriter.base import BaseRewriter


class TypeStructureMixin(BaseRewriter):
  """
  Mixin for transforming Type Annotations.

  Attributes:
      _in_annotation (bool): Tracks if the visitor is currently inside a type annotation.
  """

  def visit_Annotation(self, node: cst.Annotation) -> Optional[bool]:
    """
    Enters a type annotation node (e.g., `: torch.Tensor` or `-> int`).
    Sets a flag to allow `leave_Name` to rewrite type names.
    """
    self._in_annotation = True
    return True

  def leave_Annotation(self, original_node: cst.Annotation, updated_node: cst.Annotation) -> cst.Annotation:
    """
    Leaves a type annotation node. Resets the annotation flag.
    """
    self._in_annotation = False
    return updated_node

  def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.BaseExpression:
    """
    Rewrites Names found within Type Annotations.

    If we are inside an annotation context (e.g., `x: Tensor`), we resolve
    this name via aliases (e.g., `Tensor` -> `torch.Tensor`), look it up
    in the semantics, and rewrite it if a mapping exists (e.g., -> `jax.Array`).

    Note: Calls and Attributes (e.g. `torch.Tensor`) are handled by other mixins
    or `leave_Attribute` regardless of context, but bare `Name` nodes in code
    are usually variables we don't want to touch. This method is scoped strictly
    to annotations to be safe.
    """
    if getattr(self, "_in_annotation", False):
      full_name = self._get_qualified_name(original_node)
      if full_name:
        mapping = self._get_mapping(full_name)
        if mapping and "api" in mapping:
          return self._create_name_node(mapping["api"])

    return updated_node
