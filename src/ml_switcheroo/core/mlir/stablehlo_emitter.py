"""
StableHLO Emitter Backend.

Translates Python CST to MLIR using the StableHLO dialect for math operations
and the Func/Builtin dialects for structure. It relies on the SemanticsManager
to map Python source APIs (like `torch.abs`) to StableHLO operations (like `stablehlo.abs`).
"""

import libcst as cst
from typing import List, Tuple, Optional

from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter
from ml_switcheroo.core.mlir.nodes import (
  OperationNode,
  AttributeNode,
  RegionNode,
  TypeNode,
  ValueNode,
  BlockNode,
)
from ml_switcheroo.semantics.manager import SemanticsManager


class StableHloEmitter(PythonToMlirEmitter):
  """
  Specialized Emitter that produces StableHLO, Func, and Builtin dialect operations.
  """

  def __init__(self, semantics: SemanticsManager):
    """
    Initialize the emitter with access to the Semantic Knowledge Base.

    Args:
        semantics: The manager instance to use for API resolution.
    """
    super().__init__()
    self.semantics = semantics

  def _emit_class_def(self, node: cst.ClassDef) -> OperationNode:
    """
    Maps Python Class to 'builtin.module'.

    Args:
        node: LibCST ClassDef node.

    Returns:
        MLIR OperationNode representing a module.
    """
    self.ctx.enter_scope()
    # Standard MLIR uses @name syntax for symbols, but module ops usually
    # take a symbol name attribute if nested, or just define a scope.
    # We model it as a nested module for structural parity.
    name_attr = AttributeNode(name="sym_name", value=f'"{node.name.value}"')
    attributes = [name_attr]

    region = RegionNode(blocks=[self._emit_block(node.body)])
    op = OperationNode(name="module", attributes=attributes, regions=[region])
    self.ctx.exit_scope()
    return op

  def _emit_func_def(self, node: cst.FunctionDef) -> OperationNode:
    """
    Maps Python Function to 'func.func'.

    Args:
        node: LibCST FunctionDef node.

    Returns:
        MLIR OperationNode representing the function.
    """
    self.ctx.enter_scope()
    func_name = node.name.value

    block_args = []
    for param in node.params.params:
      if isinstance(param.name, cst.Name):
        p_name = param.name.value
        val = self.ctx.allocate_ssa(prefix=f"%{p_name}")
        self.ctx.declare(p_name, val)

        # Type mapping
        t_str = "tensor<*xf32>"  # Default assumption for ML tensors
        if param.annotation:
          anno_str = self._annotation_to_string(param.annotation.annotation)
          t_str = self._map_py_type_to_mlir(anno_str)

        block_args.append((val, TypeNode(t_str)))

    body_block = self._emit_block(node.body, label="^entry")
    body_block.arguments = block_args

    # FuncOp attributes
    attrs = [
      AttributeNode(name="sym_name", value=f'"{func_name}"'),
      AttributeNode(
        name="function_type",
        value="...",  # Placeholder: Full type calculation requires multi-pass analysis
      ),
    ]

    # Determine result types
    result_types = []
    if node.returns:
      rt_str = self._annotation_to_string(node.returns.annotation)
      result_types.append(TypeNode(self._map_py_type_to_mlir(rt_str)))

    op = OperationNode(
      name="func.func",
      attributes=attrs,
      regions=[RegionNode(blocks=[body_block])],
      result_types=result_types,
    )
    self.ctx.exit_scope()
    return op

  def _emit_return(self, node: cst.Return) -> List[OperationNode]:
    """
    Maps Python Return to 'func.return'.

    Args:
        node: LibCST Return node.

    Returns:
        List of operations (expression evaluation + return).
    """
    ops = []
    operands = []
    if node.value:
      val, expr_ops = self._emit_expression(node.value)
      ops.extend(expr_ops)
      operands.append(val)

    op = OperationNode(name="func.return", operands=operands)
    ops.append(op)
    return ops

  def _emit_expression(self, expr: cst.BaseExpression) -> Tuple[ValueNode, List[OperationNode]]:
    """
    Overrides expression generation to intercept and resolve Semantic Operations.

    Args:
        expr: LibCST Expression node.

    Returns:
        Tuple of (Result Value, List of Ops).
    """
    # Delegate to base logic first
    val, ops = super()._emit_expression(expr)

    # Post-process the generated operations to resolve 'sw.op' to 'stablehlo.*'
    resolved_ops = []
    for op in ops:
      if op.name == "sw.op":
        self._resolve_sw_op(op)
      resolved_ops.append(op)

    return val, resolved_ops

  def _resolve_sw_op(self, op: OperationNode) -> None:
    """
    Mutates a 'sw.op' node into a 'stablehlo' node if a mapping exists.
    Removes the 'type' attribute upon successful resolution.

    Args:
        op: The operation node to mutate in-place.
    """
    # Find type attribute
    type_attr = next((a for a in op.attributes if a.name == "type"), None)
    if not type_attr:
      return

    api_name = str(type_attr.value).strip('"').strip("'")
    mapped_name = self._lookup_stablehlo_op(api_name)

    if mapped_name:
      op.name = mapped_name
      # Remove the 'type' attribute as it is now encoded in the op name
      op.attributes = [a for a in op.attributes if a.name != "type"]
      # Inject default tensor result type if missing
      if not op.result_types:
        op.result_types = [TypeNode("tensor<*xf32>")]

  def _lookup_stablehlo_op(self, api_name: str) -> Optional[str]:
    """
    Queries the SemanticsManager for the StableHLO variant of the given API.

    Args:
        api_name: Logic API string (e.g. 'torch.abs').

    Returns:
        StableHLO operation name (e.g. 'stablehlo.abs') or None.
    """
    # 1. Reverse lookup to get Abstract ID
    defn = self.semantics.get_definition(api_name)
    if not defn:
      return None

    _abstract_id, details = defn
    variants = details.get("variants", {})

    # 2. Check for 'stablehlo' variant
    if "stablehlo" in variants and variants["stablehlo"]:
      return variants["stablehlo"].get("api")

    return None

  def _map_py_type_to_mlir(self, type_str: str) -> str:
    """
    Maps Python type strings to MLIR types.

    Args:
        type_str: Python Type Hint string.

    Returns:
        MLIR Type string (e.g. 'f32').
    """
    clean = type_str.lower().strip()
    if clean == "int":
      return "i32"
    if clean == "float":
      return "f32"
    if clean == "bool":
      return "i1"
    if "tensor" in clean or "array" in clean:
      # Unranked tensor of floats as generous default
      return "tensor<*xf32>"
    return "!sw.unknown"
