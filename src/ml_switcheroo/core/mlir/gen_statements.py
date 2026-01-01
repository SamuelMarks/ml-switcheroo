"""
Statement Generation Mixin for MLIR->Python.

This module implements the transformation logic for structural MLIR operations
(`sw.module`, `sw.func`, `sw.setattr`, `sw.return`) into Python LibCST statements.
It operates as a mixin to be combined with expression generation logic in the main generator.
"""

import libcst as cst
from collections import defaultdict
from typing import Optional, List

from ml_switcheroo.core.mlir.nodes import OperationNode, BlockNode
from ml_switcheroo.core.mlir.gen_base import BaseGeneratorMixin
from ml_switcheroo.core.mlir.naming import NamingContext


class StatementGeneratorMixin(BaseGeneratorMixin):
  """
  Mixin class for generating LibCST Statements from MLIR Operations.

  This class handles high-level structures (Classes, Functions) and statement-level
  operations like Assignments (`sw.setattr`) and Returns (`sw.return`).
  It relies on the host class to provide context and expression resolution methods.
  """

  # Interface requirements from host class (MlirToPythonGenerator)
  ctx: NamingContext
  usage_counts: defaultdict
  usage_consumers: dict

  def _resolve_operand(self, ssa_name: str) -> cst.BaseExpression:
    """
    Abstract method: Resolves an SSA ID to a Python Expression.
    Must be implemented by the host generator.

    Args:
        ssa_name: The SSA identifier (e.g. "%0").

    Returns:
        A LibCST Expression representing the variable or value.
    """
    raise NotImplementedError

  def _convert_block(self, block: BlockNode) -> List[cst.BaseStatement]:
    """
    Abstract method: Converts a Block of operations into a list of Statements.
    Must be implemented by the host generator.

    Args:
        block: The MLIR BlockNode.

    Returns:
        List of python statements.
    """
    raise NotImplementedError

  def _scan_block_usage(self, block: BlockNode) -> None:
    """
    Abstract method: Pre-scans a block to determine variable usage counts.
    Must be implemented by the host generator.

    Args:
        block: The MLIR BlockNode.
    """
    raise NotImplementedError

  def _convert_setattr(self, op: OperationNode) -> cst.SimpleStatementLine:
    """
    Converts a `sw.setattr` operation to a Python assignment statement.

    Structure: `sw.setattr %base %val {name="attr"}` becomes `base.attr = val`.

    Args:
        op: The `sw.setattr` OperationNode.

    Returns:
        A LibCST SimpleStatementLine containing the assignment.
        Returns `pass` if operands are missing (error recovery).
    """
    if len(op.operands) < 2:
      return cst.SimpleStatementLine(body=[cst.Pass()])

    obj_expr = self._resolve_operand(op.operands[0].name)
    val_expr = self._resolve_operand(op.operands[1].name)
    attr_name = (self._get_attr(op, "name") or "unknown").strip('"')

    target = cst.Attribute(value=obj_expr, attr=cst.Name(attr_name))
    assign = cst.Assign(targets=[cst.AssignTarget(target=target)], value=val_expr)
    return cst.SimpleStatementLine(body=[assign])

  def _convert_return(self, op: OperationNode) -> cst.SimpleStatementLine:
    """
    Converts a `sw.return` operation to a Python return statement.

    Args:
        op: The `sw.return` OperationNode.

    Returns:
        A LibCST SimpleStatementLine.
    """
    val_node = None
    if op.operands:
      val_node = self._resolve_operand(op.operands[0].name)
    return cst.SimpleStatementLine(body=[cst.Return(value=val_node)])

  def _convert_class_def(self, op: OperationNode) -> cst.ClassDef:
    """
    Converts a `sw.module` operation to a Python Class definition.

    Extracts the class name from `sym_name` attribute and base classes from `bases`.
    Recursively converts the inner region.

    Args:
        op: The `sw.module` OperationNode.

    Returns:
        A LibCST ClassDef node.
    """
    name_attr = self._get_attr(op, "sym_name")
    class_name = name_attr.strip('"') if name_attr else "UnknownClass"

    base_nodes = []
    bases_attr = self._get_attr(op, "bases")
    if bases_attr:
      clean = bases_attr.strip("[]")
      if clean:
        parts = clean.split(",")
        for p in parts:
          b = p.strip().strip('"').strip("'")
          if b:
            base_nodes.append(cst.Arg(value=self._create_dotted_name(b)))

    body_stmts = []
    if op.regions and op.regions[0].blocks:
      # Recursively convert body block
      body_stmts = self._convert_block(op.regions[0].blocks[0])

    if not body_stmts:
      body_stmts = [cst.SimpleStatementLine(body=[cst.Pass()])]

    return cst.ClassDef(name=cst.Name(class_name), bases=base_nodes, body=cst.IndentedBlock(body=body_stmts))

  def _convert_func_def(self, op: OperationNode) -> cst.FunctionDef:
    """
    Converts a `sw.func` operation to a Python Function definition.

    Handles:
    1.  Creating a fresh `NamingContext` to isolate function scope variables.
    2.  Resetting usage counts for local analysis.
    3.  Registering block arguments as function parameters.
    4.  Parsing type hints from MLIR types.

    Args:
        op: The `sw.func` OperationNode.

    Returns:
        A LibCST FunctionDef node.
    """
    name_attr = self._get_attr(op, "sym_name")
    func_name = name_attr.strip('"') if name_attr else "unknown_func"

    # Scope Reset: Logic for functions should be isolated from parent naming context (mostly)
    # This prevents 'self' from becoming 'self13' due to collisions with constructor context.
    prev_ctx = self.ctx
    self.ctx = NamingContext()

    # Reset Usage Counts for new scope to prevent external block counts affecting internal logic
    prev_usage_counts = self.usage_counts
    self.usage_counts = defaultdict(int)
    # Also reset consumers map
    prev_usage_consumers = self.usage_consumers
    self.usage_consumers = {}

    params = []
    body_stmts = []

    try:
      if op.regions and op.regions[0].blocks:
        block0 = op.regions[0].blocks[0]
        # Pre-analyze usage for this scope
        self._scan_block_usage(block0)

        # Register arguments in local context to establish 'self', 'x', etc.
        for val, typ in block0.arguments:
          py_name = self.ctx.register(val.name, hint=val.name)
          annotation = None
          type_str = typ.body
          if type_str and type_str.startswith("!sw.type<"):
            inner = type_str[9:-1].strip('"').strip("'")
            if inner != "Any":
              annotation = cst.Annotation(annotation=self._create_dotted_name(inner))
          params.append(cst.Param(name=cst.Name(py_name), annotation=annotation))

        body_stmts = self._convert_block(block0)

      if not body_stmts:
        body_stmts = [cst.SimpleStatementLine(body=[cst.Pass()])]

    finally:
      self.ctx = prev_ctx
      self.usage_counts = prev_usage_counts
      self.usage_consumers = prev_usage_consumers

    return cst.FunctionDef(
      name=cst.Name(func_name), params=cst.Parameters(params=params), body=cst.IndentedBlock(body=body_stmts)
    )
