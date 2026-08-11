"""MLIR Emitter Declaration Mixin.

This module provides the MlirEmitterDeclMixin class, which is a mixin
offering capabilities to convert Python class and function definitions
into MLIR operation nodes (such as sw.module and sw.func).
"""

import libcst as cst
from typing import TYPE_CHECKING, Any
from ml_switcheroo.core.mlir.cst import OperationNode, RegionNode, AttributeNode, TypeNode


class MlirEmitterDeclMixin:
  """Mixin class providing MLIR emission capabilities for Python declarations.

  This mixin contains methods for converting Python high-level AST constructs
  like class definitions and function definitions into corresponding MLIR
  structural constructs (operations, regions, and blocks).
  """

  if TYPE_CHECKING:
    ctx: Any

    def _flatten_attr(self, attr: Any) -> Any:
      """Flatten a Name or Attribute chain into a string.

      Args:
          self: The mixin instance.
          attr: A LibCST Name or Attribute node representing an identifier chain.

      Returns:
          A dotted string representation of the chain (e.g. "self.layer"),
          or None if the attribute chain cannot be flattened.
      """
      return None

    def _emit_block(self, block: Any, label: str = "^bb0") -> Any:
      """Convert a sequence of statements or a suite into an MLIR Block.

      Args:
          self: The mixin instance.
          block: A CST Suite or list of statement nodes.
          label: Optional label for the generated block (defaults to "^bb0").

      Returns:
          A BlockNode representing the populated MLIR Block.
      """
      return None

    def _annotation_to_string(self, ann: Any) -> str:
      """Convert a type annotation node to its string representation.

      Args:
          self: The mixin instance.
          ann: The type annotation CST node (typically a Name or Attribute).

      Returns:
          The string representation of the type annotation, or "Any" if not flattenable.
      """
      return "Any"

  def _emit_class_def(self, node: cst.ClassDef) -> OperationNode:
    """Convert a Python class definition to `sw.module`.

    Args:
        node: The ClassDef node.

    Returns:
        An `sw.module` OperationNode containing the class body region.

    """
    self.ctx.enter_scope()
    name_obj = AttributeNode(name="sym_name", value=f'"{node.name.value}"')

    attributes = [name_obj]

    # Capture Bases (Inheritance)
    if node.bases:
      base_names = []
      for b in node.bases:
        flat_name = self._flatten_attr(b.value)
        if flat_name:
          base_names.append(f'"{flat_name}"')

      if base_names:
        attributes.append(AttributeNode(name="bases", value=base_names))

    region = RegionNode(blocks=[self._emit_block(node.body)])
    op = OperationNode(name="sw.module", attributes=attributes, regions=[region])
    self.ctx.exit_scope()
    return op

  def _emit_func_def(self, node: cst.FunctionDef) -> OperationNode:
    """Convert a Python function definition to `sw.func`.

    Args:
        node: The FunctionDef node.

    Returns:
        An `sw.func` OperationNode with arguments mapped to block arguments.

    """
    self.ctx.enter_scope()
    func_name = node.name.value
    block_args = []

    for param in node.params.params:
      if isinstance(param.name, cst.Name):
        p_name = param.name.value
        val = self.ctx.allocate_ssa(prefix=f"%{p_name}")
        self.ctx.declare(p_name, val)
        t_str = "!sw.unknown"
        if param.annotation:
          t_str = f'!sw.type<"{self._annotation_to_string(param.annotation.annotation)}">'
        block_args.append((val, TypeNode(body=t_str)))

    body_block = self._emit_block(node.body, label="^entry")
    body_block.arguments = block_args
    op = OperationNode(
      name="sw.func",
      attributes=[AttributeNode(name="sym_name", value=f'"{func_name}"')],
      regions=[RegionNode(blocks=[body_block])],
    )
    self.ctx.exit_scope()
    return op
