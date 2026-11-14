"""Statement Generation Delegate for MLIR -> Python.

Provides ``MlirStatementGenerator``, a single-responsibility delegate translating
structural MLIR operations (``sw.module``, ``sw.func`` / ``func.func``,
``sw.setattr``, ``sw.import``, and ``sw.return``) into Python LibCST statements.
"""

import ast
import typing
from collections import defaultdict

import libcst as cst

from ml_switcheroo.core.mlir.cst import BlockNode, OperationNode
from ml_switcheroo.core.mlir.gen_base import MlirGenHelpers
from ml_switcheroo.core.mlir.naming import NamingContext


class MlirStmtGeneratorHost(typing.Protocol):
  """Structural protocol required by ``MlirStatementGenerator``.

  Attributes:
      ctx: Active ``NamingContext`` for variable naming.
      usage_counts: Mapping of SSA value names to usage counts.
      usage_consumers: Mapping of SSA value names to consumer operations.
  """

  ctx: NamingContext
  usage_counts: defaultdict[str, int]
  usage_consumers: dict[str, OperationNode]

  def _resolve_operand(self, ssa_name: str) -> cst.BaseExpression:
    """Resolve an SSA value name to its Python ``cst.BaseExpression``.

    Args:
        ssa_name: SSA value identifier.

    Returns:
        Resolved ``cst.BaseExpression``.
    """
    ...

  def _convert_block(self, block: BlockNode) -> list[cst.SimpleStatementLine | cst.BaseCompoundStatement]:
    """Convert an MLIR ``BlockNode`` into a list of Python statements.

    Args:
        block: MLIR ``BlockNode`` to convert.

    Returns:
        List of LibCST statements.
    """
    ...

  def _scan_block_usage(self, block: BlockNode) -> None:
    """Analyze SSA operand usage counts within ``block``.

    Args:
        block: MLIR ``BlockNode`` to scan.
    """
    ...


class MlirStatementGenerator:
  """Single-responsibility delegate generating LibCST statements from structural MLIR ops.

  Attributes:
      host: Bound ``MlirStmtGeneratorHost`` instance.
      helpers: ``MlirGenHelpers`` utility instance.
  """

  def __init__(self, host: MlirStmtGeneratorHost, helpers: MlirGenHelpers | None = None) -> None:
    """Initialize ``MlirStatementGenerator`` bound to ``host``.

    Args:
        host: Bound ``MlirStmtGeneratorHost`` instance.
        helpers: Optional ``MlirGenHelpers`` instance.
    """
    self.host: MlirStmtGeneratorHost = host
    self.helpers: MlirGenHelpers = helpers if helpers is not None else MlirGenHelpers()

  def convert_setattr(self, op: OperationNode) -> cst.SimpleStatementLine:
    """Convert an ``sw.setattr`` operation into a Python attribute assignment statement.

    Args:
        op: ``OperationNode`` representing ``sw.setattr``.

    Returns:
        ``cst.SimpleStatementLine`` containing the assignment (or ``pass`` if malformed).
    """
    if len(op.operands) < 2:
      return cst.SimpleStatementLine(body=[cst.Pass()])

    obj_expr = self.host._resolve_operand(op.operands[0].name)
    val_expr = self.host._resolve_operand(op.operands[1].name)
    attr_name = (self.helpers.get_attr(op, "name") or "unknown").strip('"')

    target = cst.Attribute(value=obj_expr, attr=cst.Name(attr_name))
    assign = cst.Assign(targets=[cst.AssignTarget(target=target)], value=val_expr)
    return cst.SimpleStatementLine(body=[assign])

  def convert_import(self, op: OperationNode) -> cst.SimpleStatementLine:
    """Convert an ``sw.import`` operation back into a Python ``import`` / ``from ... import`` line.

    Args:
        op: ``OperationNode`` representing ``sw.import``.

    Returns:
        ``cst.SimpleStatementLine`` containing the reconstructed import statement.
    """
    module_attr = self.helpers.get_attr(op, "module")
    names_attr = self.helpers.get_attr(op, "names")
    aliases_attr = self.helpers.get_attr(op, "aliases")

    names: list[str] = []
    aliases: list[str] = []
    try:
      if names_attr:
        parsed_names = ast.literal_eval(names_attr)
        if isinstance(parsed_names, list):
          names = [str(item) for item in parsed_names]
      if aliases_attr:
        parsed_aliases = ast.literal_eval(aliases_attr)
        if isinstance(parsed_aliases, list):
          aliases = [str(item) for item in parsed_aliases]
    except (ValueError, SyntaxError):
      pass

    module_val = module_attr.strip('"') if module_attr else None

    import_aliases: list[cst.ImportAlias] = []
    for n, a in zip(names, aliases, strict=False):
      clean_n = str(n).strip("'").strip('"')
      clean_a = str(a).strip("'").strip('"')

      if clean_n == "*":
        return cst.SimpleStatementLine(
          body=[cst.ImportFrom(module=self.helpers.create_dotted_name(module_val or ""), names=cst.ImportStar())]
        )

      asname = cst.AsName(name=cst.Name(clean_a)) if (clean_a and clean_a != clean_n) else None
      import_aliases.append(cst.ImportAlias(name=self.helpers.create_dotted_name(clean_n), asname=asname))

    if module_val:
      if import_aliases:
        return cst.SimpleStatementLine(
          body=[cst.ImportFrom(module=self.helpers.create_dotted_name(module_val), names=import_aliases)]
        )
      return cst.SimpleStatementLine(
        body=[cst.Import(names=[cst.ImportAlias(name=self.helpers.create_dotted_name(module_val))])]
      )

    if import_aliases:
      return cst.SimpleStatementLine(body=[cst.Import(names=import_aliases)])
    return cst.SimpleStatementLine(body=[cst.Pass()])

  def convert_return(self, op: OperationNode) -> cst.SimpleStatementLine:
    """Convert a return operation into a Python ``return`` statement.

    Args:
        op: ``OperationNode`` representing ``sw.return`` / ``func.return`` / ``stablehlo.return``.

    Returns:
        ``cst.SimpleStatementLine`` containing ``cst.Return``.
    """
    val_node = self.host._resolve_operand(op.operands[0].name) if op.operands else None
    return cst.SimpleStatementLine(body=[cst.Return(value=val_node)])

  def convert_class_def(self, op: OperationNode) -> cst.ClassDef:
    """Convert an ``sw.module`` operation into a Python ``cst.ClassDef``.

    Args:
        op: ``OperationNode`` representing ``sw.module``.

    Returns:
        Constructed ``cst.ClassDef``.
    """
    name_attr = self.helpers.get_attr(op, "sym_name")
    class_name = name_attr.strip('"') if name_attr else "UnknownClass"

    base_nodes: list[cst.Arg] = []
    bases_attr = self.helpers.get_attr(op, "bases")
    if bases_attr:
      clean = bases_attr.strip("[]")
      if clean:
        for p in clean.split(","):
          b = p.strip().strip('"').strip("'")
          if b:
            base_nodes.append(cst.Arg(value=self.helpers.create_dotted_name(b)))

    body_stmts: list[cst.SimpleStatementLine | cst.BaseCompoundStatement] = []
    if op.regions and op.regions[0].blocks:
      body_stmts = self.host._convert_block(op.regions[0].blocks[0])

    if not body_stmts:
      body_stmts = [cst.SimpleStatementLine(body=[cst.Pass()])]

    return cst.ClassDef(name=cst.Name(class_name), bases=base_nodes, body=cst.IndentedBlock(body=body_stmts))

  def convert_func_def(self, op: OperationNode) -> cst.FunctionDef:
    """Convert an ``sw.func`` / ``func.func`` operation into a Python ``cst.FunctionDef``.

    Args:
        op: ``OperationNode`` representing the function definition.

    Returns:
        Constructed ``cst.FunctionDef``.
    """
    name_attr = self.helpers.get_attr(op, "sym_name")
    func_name = name_attr.strip('"').lstrip("@") if name_attr else ""
    if not func_name and isinstance(op.name_trivia, list):
      for t in op.name_trivia:
        txt = t.text.strip()
        if txt.startswith("@"):
          func_name = txt.lstrip("@")
          break
    if not func_name:
      func_name = "unknown_func"

    prev_ctx = self.host.ctx
    self.host.ctx = NamingContext()
    prev_usage_counts = self.host.usage_counts
    self.host.usage_counts = defaultdict(int)
    prev_usage_consumers = self.host.usage_consumers
    self.host.usage_consumers = {}

    params: list[cst.Param] = []
    body_stmts: list[cst.SimpleStatementLine | cst.BaseCompoundStatement] = []

    try:
      if op.regions and op.regions[0].blocks:
        block0 = op.regions[0].blocks[0]
        self.host._scan_block_usage(block0)

        for val, typ in block0.arguments:
          py_name = self.host.ctx.register(val.name, hint=val.name)
          annotation = None
          type_str = typ.body
          if type_str and type_str.startswith("!sw.type<"):
            inner = type_str[9:-1].strip('"').strip("'")
            if inner != "Any":
              annotation = cst.Annotation(annotation=self.helpers.create_dotted_name(inner))
          params.append(cst.Param(name=cst.Name(py_name), annotation=annotation))

        body_stmts = self.host._convert_block(block0)

      if not body_stmts:
        body_stmts = [cst.SimpleStatementLine(body=[cst.Pass()])]
    finally:
      self.host.ctx = prev_ctx
      self.host.usage_counts = prev_usage_counts
      self.host.usage_consumers = prev_usage_consumers

    return cst.FunctionDef(
      name=cst.Name(func_name),
      params=cst.Parameters(params=params),
      body=cst.IndentedBlock(body=body_stmts),
    )


__all__ = ["MlirStatementGenerator", "MlirStmtGeneratorHost"]
