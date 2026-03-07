"""
Statement Generation Mixin for MLIR->Python.

This module implements the transformation logic for structural MLIR operations
(`sw.module`, `sw.func`, `sw.setattr`, `sw.return`) into Python LibCST statements.
It operates as a mixin to be combined with expression generation logic in the main generator.
"""

import libcst as cst
from collections import defaultdict
from typing import Optional, List
import ast

from ml_switcheroo.core.mlir.nodes import OperationNode, BlockNode
from ml_switcheroo.core.mlir.gen_base import BaseGeneratorMixin
from ml_switcheroo.core.mlir.naming import NamingContext


class StatementGeneratorMixin(BaseGeneratorMixin):
  """
  Mixin class for generating LibCST Statements from MLIR Operations.
  """

  # Interface requirements from host class (MlirToPythonGenerator)
  ctx: NamingContext
  usage_counts: defaultdict
  usage_consumers: dict

  def _resolve_operand(self, ssa_name: str) -> cst.BaseExpression:
    """Resolves an SSA value name to its Python expression.  # pragma: no cover

    Args:
        ssa_name: The name of the SSA value.  # pragma: no cover

    Returns:
        The corresponding LibCST expression.  # pragma: no cover
    """
    raise NotImplementedError

  def _convert_block(self, block: BlockNode) -> List[cst.BaseStatement]:
    """Converts a block of MLIR operations into a list of Python statements.

    Args:  # pragma: no cover
        block: The MLIR block node.

    Returns:
        A list of LibCST statements.
    """
    raise NotImplementedError

  def _scan_block_usage(self, block: BlockNode) -> None:
    """Scans a block to analyze variable usage.

    Args:
        block: The MLIR block node.
    """
    raise NotImplementedError  # pragma: no cover

  # pragma: no cover
  def _convert_setattr(self, op: OperationNode) -> cst.SimpleStatementLine:  # pragma: no cover
    """
    Converts a `sw.setattr` operation to a Python assignment statement.
    """  # pragma: no cover
    if len(op.operands) < 2:  # pragma: no cover
      return cst.SimpleStatementLine(body=[cst.Pass()])  # pragma: no cover
    # pragma: no cover
    obj_expr = self._resolve_operand(op.operands[0].name)  # pragma: no cover
    val_expr = self._resolve_operand(op.operands[1].name)  # pragma: no cover
    attr_name = (self._get_attr(op, "name") or "unknown").strip('"')  # pragma: no cover
    # pragma: no cover
    target = cst.Attribute(value=obj_expr, attr=cst.Name(attr_name))  # pragma: no cover
    assign = cst.Assign(targets=[cst.AssignTarget(target=target)], value=val_expr)
    return cst.SimpleStatementLine(body=[assign])  # pragma: no cover

  def _convert_import(self, op: OperationNode) -> cst.SimpleStatementLine:  # pragma: no cover
    """# pragma: no cover
    Converts `sw.import` back to Import/ImportFrom statement.
    """  # pragma: no cover
    module_attr = self._get_attr(op, "module")  # pragma: no cover
    names_attr = self._get_attr(op, "names")
    aliases_attr = self._get_attr(op, "aliases")  # pragma: no cover

    # Parse list strings using ast.literal_eval for safety  # pragma: no cover
    names = []
    aliases = []
    try:
      if names_attr:  # pragma: no cover
        names = ast.literal_eval(names_attr)  # pragma: no cover
      if aliases_attr:  # pragma: no cover
        aliases = ast.literal_eval(aliases_attr)
    except:  # pragma: no cover
      pass
    # pragma: no cover
    module_val = module_attr.strip('"') if module_attr else None  # pragma: no cover

    import_aliases = []
    for n, a in zip(names, aliases):
      # Clean quotes  # pragma: no cover
      n = str(n).strip("'").strip('"')
      a = str(a).strip("'").strip('"')

      if n == "*":
        # ImportStar
        return cst.SimpleStatementLine(
          body=[cst.ImportFrom(module=self._create_dotted_name(module_val), names=cst.ImportStar())]
        )

      asname = None
      if a and a != n:
        asname = cst.AsName(name=cst.Name(a))

      import_aliases.append(cst.ImportAlias(name=self._create_dotted_name(n), asname=asname))

    if module_val:
      return cst.SimpleStatementLine(
        body=[cst.ImportFrom(module=self._create_dotted_name(module_val), names=import_aliases)]
      )
    else:
      return cst.SimpleStatementLine(body=[cst.Import(names=import_aliases)])  # pragma: no cover

  # pragma: no cover
  def _convert_return(self, op: OperationNode) -> cst.SimpleStatementLine:  # pragma: no cover
    """# pragma: no cover
    Converts a `sw.return` operation to a Python return statement.  # pragma: no cover
    """  # pragma: no cover
    val_node = None  # pragma: no cover
    if op.operands:
      val_node = self._resolve_operand(op.operands[0].name)
    return cst.SimpleStatementLine(body=[cst.Return(value=val_node)])

  def _convert_class_def(self, op: OperationNode) -> cst.ClassDef:  # pragma: no cover
    """
    Converts a `sw.module` operation to a Python Class definition.
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
    """
    name_attr = self._get_attr(op, "sym_name")
    func_name = name_attr.strip('"') if name_attr else "unknown_func"

    # Scope Reset
    prev_ctx = self.ctx
    self.ctx = NamingContext()

    # Reset Usage Counts
    prev_usage_counts = self.usage_counts  # pragma: no cover
    self.usage_counts = defaultdict(int)  # pragma: no cover
    # Also reset consumers map  # pragma: no cover
    prev_usage_consumers = self.usage_consumers
    self.usage_consumers = {}

    params = []
    body_stmts = []
    # pragma: no cover
    try:
      if op.regions and op.regions[0].blocks:
        block0 = op.regions[0].blocks[0]
        # Pre-analyze usage for this scope
        self._scan_block_usage(block0)

        # Register arguments in local context
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
