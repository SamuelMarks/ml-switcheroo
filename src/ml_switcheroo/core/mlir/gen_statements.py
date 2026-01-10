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
    raise NotImplementedError

  def _convert_block(self, block: BlockNode) -> List[cst.BaseStatement]:
    raise NotImplementedError

  def _scan_block_usage(self, block: BlockNode) -> None:
    raise NotImplementedError

  def _convert_setattr(self, op: OperationNode) -> cst.SimpleStatementLine:
    """
    Converts a `sw.setattr` operation to a Python assignment statement.
    """
    if len(op.operands) < 2:
      return cst.SimpleStatementLine(body=[cst.Pass()])

    obj_expr = self._resolve_operand(op.operands[0].name)
    val_expr = self._resolve_operand(op.operands[1].name)
    attr_name = (self._get_attr(op, "name") or "unknown").strip('"')

    target = cst.Attribute(value=obj_expr, attr=cst.Name(attr_name))
    assign = cst.Assign(targets=[cst.AssignTarget(target=target)], value=val_expr)
    return cst.SimpleStatementLine(body=[assign])

  def _convert_import(self, op: OperationNode) -> cst.SimpleStatementLine:
    """
    Converts `sw.import` back to Import/ImportFrom statement.
    """
    module_attr = self._get_attr(op, "module")
    names_attr = self._get_attr(op, "names")
    aliases_attr = self._get_attr(op, "aliases")

    # Parse list strings using ast.literal_eval for safety
    names = []
    aliases = []
    try:
      if names_attr:
        names = ast.literal_eval(names_attr)
      if aliases_attr:
        aliases = ast.literal_eval(aliases_attr)
    except:
      pass

    module_val = module_attr.strip('"') if module_attr else None

    import_aliases = []
    for n, a in zip(names, aliases):
      # Clean quotes
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
      return cst.SimpleStatementLine(body=[cst.Import(names=import_aliases)])

  def _convert_return(self, op: OperationNode) -> cst.SimpleStatementLine:
    """
    Converts a `sw.return` operation to a Python return statement.
    """
    val_node = None
    if op.operands:
      val_node = self._resolve_operand(op.operands[0].name)
    return cst.SimpleStatementLine(body=[cst.Return(value=val_node)])

  def _convert_class_def(self, op: OperationNode) -> cst.ClassDef:
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
