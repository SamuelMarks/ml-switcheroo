"""MLIR Emitter Declaration Mixin."""

import libcst as cst
from typing import TYPE_CHECKING, Any
from ml_switcheroo.core.mlir.nodes import OperationNode, RegionNode, AttributeNode, TypeNode


class MlirEmitterDeclMixin:
  """Docstring."""

  if TYPE_CHECKING:
    ctx: Any

    def _flatten_attr(self, attr: Any) -> Any:
      """Docstring."""
      ...

    def _emit_block(self, block: Any, label: str = "^bb0") -> Any:
      """Docstring."""
      ...

    def _annotation_to_string(self, ann: Any) -> str:
      """Docstring."""
      ...

  def _emit_class_def(self, node: cst.ClassDef) -> OperationNode:
    """Converts a Python class definition to `sw.module`.

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
    """Converts a Python function definition to `sw.func`.

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
        block_args.append((val, TypeNode(t_str)))

    body_block = self._emit_block(node.body, label="^entry")
    body_block.arguments = block_args
    op = OperationNode(
      name="sw.func", attributes=[AttributeNode("sym_name", f'"{func_name}"')], regions=[RegionNode(blocks=[body_block])]
    )
    self.ctx.exit_scope()
    return op
