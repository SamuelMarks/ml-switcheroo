"""
MLIR to Python (LibCST) Generator.

This module provides the `MlirToPythonGenerator` class, which consumes the
MLIR CST object model and reconstructs valid Python code via LibCST.

It performs the inverse operation of the Emitter:
1.  **Dialect Interpretation**: Maps `sw.*` ops back to Python constructs.
2.  **SSA Resolution**: Converts `%id` references into readable Python variable names.
3.  **Trivia Restoration**: Transforms `// comment` back to `# comment`.
4.  **Expression Folding**: Inlines single-use intermediates (SSA values) to
    produce Pythonic code (e.g. `f(x)` instead of `_1=x; _2=f(_1);`).

Feature Hardening:
- Robustly handles `sw.op` "type" attributes for deep class hierarchies (e.g. `torch.nn.Conv2d`).
- Distinguishes between static operation calls (`sw.op`) and dynamic calls (`sw.call`).
- Ensures valid assignment generation for results.
- **Naming Heuristics**: Preserves variable names based on SSA IDs or Type Hints, prioritizing stability.
- **Type Reconstruction**: Restores Python type hints from MLIR `!sw.type` annotations.
"""

from typing import Dict, List, Optional, Set
import re
from collections import defaultdict

import libcst as cst

from ml_switcheroo.core.mlir.nodes import (
  BlockNode,
  ModuleNode,
  OperationNode,
  TriviaNode,
  ValueNode,
)


class NamingContext:
  """
  Tracks mapping between MLIR SSA IDs and Python variable names.
  Ensures generated names are valid identifiers.
  """

  def __init__(self) -> None:
    # Map SSA Identifier (e.g. "%0") -> Python Identifier (e.g. "v0")
    self._map: Dict[str, str] = {}
    # Track used python names to prevent collision
    self._used_names: Dict[str, bool] = {}
    # Reserved python keywords + return to avoid collision logic
    # Removed 'self' to allow clean reconstruction of methods
    self._reserved = {
      "return",
      "def",
      "class",
      "if",
      "else",
      "for",
      "import",
      "from",
      "as",
      "with",
    }

  def register(self, ssa_name: str, hint: Optional[str] = None) -> str:
    """
    Assigns a valid Python name to an SSA value.

    Naming Strategy:
    1. If hint provided: Use hint (cleaned).
    2. If SSA ID (e.g. %res): Use prefix + ID body (e.g. _res).
    3. Heuristic: Strip trailing numeric counters from SSA hints if base is unique.
       (e.g., %self0 -> self, %x5 -> x).

    Args:
        ssa_name: The MLIR variable name (e.g. "%0", "%arg0").
        hint: Optional string to guide naming (e.g. "x" from original source).

    Returns:
        The resolved Python identifier string.
    """
    base = "v"

    if hint:
      # Clean start/chars
      clean = hint.lstrip("%").replace(".", "_")

      # Heuristic: If hint ends in digits (e.g. self0), try stripping them
      # to recover original name 'self', unless it collides.
      match = re.match(r"([a-zA-Z_]+)\d+$", clean)
      if match:
        candidate = match.group(1)
        # Only use stripped name if it's safe (not reserved/used)
        if candidate not in self._used_names and candidate not in self._reserved:
          base = candidate
        else:
          base = clean
      else:
        base = clean

    elif ssa_name.startswith("%"):
      base = "_" + ssa_name[1:]

    py_name = base

    # Fallback/Collision Resolution
    if not py_name.isidentifier() or py_name in self._reserved or py_name in self._used_names:
      # Collision or invalid: Try prepending underscore
      if not py_name.startswith("_"):
        attempt = f"_{py_name}"
      else:
        attempt = py_name

      # If still used or invalid, fallback to indexed v
      if attempt in self._used_names or not attempt.isidentifier():
        # Simple collision resolution logic
        count = 0
        while True:
          # Clean might be undefined if hint was None
          prefix = "v"
          if hint:
            prefix = hint.lstrip("%").replace(".", "_")

          attempt = f"{prefix}_{count}"
          if attempt not in self._used_names:
            break
          count += 1

      py_name = attempt

    self._map[ssa_name] = py_name
    self._used_names[py_name] = True
    return py_name

  def lookup(self, ssa_name: str) -> str:
    """
    Retrieves Python name for SSA ID.

    Args:
        ssa_name: The MLIR variable name.

    Returns:
        The mapped Python name, or safe fallback if not registered.
    """
    if ssa_name in self._map:
      return self._map[ssa_name]

    # Global references (functions, classes) often stored with @ prefix in Emitter
    if ssa_name.startswith("@"):
      return ssa_name[1:]

    # Fallback replacing % with _ if somehow not registered
    return ssa_name.replace("%", "_")


class MlirToPythonGenerator:
  """
  Transpiler back-end: MLIR CST -> Python LibCST.
  """

  def __init__(self) -> None:
    self.ctx = NamingContext()
    # Store usage counts for inlining logic: {ssa_name: count}
    self.usage_counts: Dict[str, int] = defaultdict(int)
    # Map of ssa_name -> CST Node (Expression) for deferred inlining
    self.deferred_exprs: Dict[str, cst.BaseExpression] = {}

  def generate(self, node: ModuleNode) -> cst.Module:
    """
    Main entry point. Converts MLIR Module to Python Module.

    Args:
        node: The root MLIR ModuleNode.

    Returns:
        A LibCST Module object representing the Python code.
    """
    # Analyze usage for inlining optimization across the whole module
    self._analyze_module_usage(node)

    # Implicit top-level block
    stmt_body = self._convert_block(node.body)

    # Ensure we return a module with valid body sequence
    return cst.Module(body=stmt_body)

  def _analyze_module_usage(self, mod: ModuleNode) -> None:
    """
    Traverses the MLIR tree to count SSAs usage.
    Populates self.usage_counts.
    """
    # We recursively check all blocks
    self._scan_block_usage(mod.body)

  def _scan_block_usage(self, block: BlockNode) -> None:
    for op in block.operations:
      # Count operand usage
      for operand in op.operands:
        self.usage_counts[operand.name] += 1
      # Recurse regions
      for region in op.regions:
        for b in region.blocks:
          self._scan_block_usage(b)

  def _convert_trivia(self, trivia: List[TriviaNode]) -> List[cst.EmptyLine]:
    """
    Converts MLIR comments (//) to Python comments (#).
    Ignores layout whitespace as LibCST handles indentation.
    """
    lines = []
    for t in trivia:
      if t.kind == "comment":
        # Convert // to #
        content = t.content.strip()
        if content.startswith("//"):
          content = "#" + content[2:]
        lines.append(cst.EmptyLine(comment=cst.Comment(content), newline=cst.Newline()))
    return lines

  def _convert_block(self, block: BlockNode) -> List[cst.BaseStatement]:
    """
    Converts operations in a block to a list of Python statements.
    Applies expression folding where possible.
    """
    stmts: List[cst.BaseStatement] = []

    for op in block.operations:
      leading = self._convert_trivia(op.leading_trivia)

      # Try to create expression first
      expr_node = self._create_expression_from_op(op)

      if expr_node:
        # Evaluate if we should inline or emit statement
        if self._should_inline_expression(op, expr_node):
          # Defer emission
          res_ssa = op.results[0].name
          self.deferred_exprs[res_ssa] = expr_node
        else:
          # Wrap as statement (Assignment or Expression Stmt)
          stmt_node = self._wrap_as_statement(op, expr_node)
          if hasattr(stmt_node, "with_changes"):
            stmt_node = stmt_node.with_changes(leading_lines=leading)
          stmts.append(stmt_node)
      else:
        # Handle statements that are never expressions (Control Flow, Class Defs, Defs)
        stmt_node = self._convert_statement_op(op)
        if stmt_node:
          if hasattr(stmt_node, "with_changes"):
            stmt_node = stmt_node.with_changes(leading_lines=leading)
          stmts.append(stmt_node)

    return stmts

  def _should_inline_expression(self, op: OperationNode, expr: cst.BaseExpression) -> bool:
    """
    Determines if the result of an operation should be folded/inlined.
    """
    if not op.results:
      return False  # No result to inline

    res_ssa = op.results[0].name
    count = self.usage_counts[res_ssa]

    # Optimize: Single-use expressions are inlined to avoid temp variables
    return count == 1

  def _resolve_operand(self, ssa_name: str) -> cst.BaseExpression:
    """
    Resolves an SSA value to a CST Expression.
    If previously deferred, returns the AST node (folding).
    Else returns a Name or Attribute reference.
    """
    if ssa_name in self.deferred_exprs:
      # Retrieve deferred expression
      expr = self.deferred_exprs[ssa_name]
      return expr

    # Fallback to variable name lookup
    py_name = self.ctx.lookup(ssa_name)

    # HARDENING FIX: If name is dotted (e.g. self.layer), return Attribute chain
    # using the internal helper to avoid bare Name validation errors.
    if "." in py_name:
      return self._create_dotted_name(py_name)

    return cst.Name(py_name)

  def _create_expression_from_op(self, op: OperationNode) -> Optional[cst.BaseExpression]:
    """
    Attempts to map an Op to a Python Expression (e.g. Call, BinaryOp, Attribute).
    Returns None if the op maps to a Statement structure (ClassDef, FuncDef).
    """
    op_name = op.name.strip('"')

    if op_name == "sw.op":
      return self._expr_sw_op(op)
    elif op_name == "sw.call":
      return self._expr_sw_call(op)
    elif op_name == "sw.getattr":
      return self._expr_sw_getattr(op)
    elif op_name == "sw.constant":
      return self._expr_sw_constant(op)

    # Not an expression-native op
    return None

  def _convert_statement_op(self, op: OperationNode) -> Optional[cst.BaseStatement]:
    """
    Maps structural ops to Statements.
    """
    op_name = op.name.strip('"')

    if op_name == "sw.module":
      return self._convert_class_def(op)
    elif op_name == "sw.func":
      return self._convert_func_def(op)
    elif op_name == "sw.return":
      return self._convert_return(op)
    elif op_name == "sw.setattr":
      return self._convert_setattr(op)

    return None

  def _wrap_as_statement(self, op: OperationNode, expr: cst.BaseExpression) -> cst.BaseStatement:
    """
    Wraps an expression into a statement (Assign or Expr).
    """
    if op.results:
      res_ssa = op.results[0].name
      hint: Optional[str] = None
      is_numeric_ssa = res_ssa.startswith("%") and res_ssa[1:].isdigit()

      if op.name.strip('"') == "sw.op" and not is_numeric_ssa:
        raw_t = self._get_attr(op, "type")
        if raw_t:
          hint = raw_t.strip('"').split(".")[-1].lower()
      elif op.name.strip('"') == "sw.getattr":
        # Hint for attrs
        raw_n = self._get_attr(op, "name")
        if raw_n:
          hint = raw_n.strip('"')

      py_target = self.ctx.register(res_ssa, hint=hint)
      target_node = cst.AssignTarget(target=cst.Name(py_target))
      return cst.SimpleStatementLine(body=[cst.Assign(targets=[target_node], value=expr)])
    else:
      return cst.SimpleStatementLine(body=[cst.Expr(value=expr)])

  # --- Ops Conversion Utils ---

  def _expr_sw_constant(self, op: OperationNode) -> cst.BaseExpression:
    val_str = self._get_attr(op, "value") or "0"
    try:
      return cst.parse_expression(val_str)
    except Exception:
      return cst.Name(val_str)

  def _expr_sw_getattr(self, op: OperationNode) -> cst.BaseExpression:
    if not op.operands:
      return cst.Name("error")
    obj_expr = self._resolve_operand(op.operands[0].name)
    attr_name = (self._get_attr(op, "name") or "unknown").strip('"')
    return cst.Attribute(value=obj_expr, attr=cst.Name(attr_name))

  def _expr_sw_call(self, op: OperationNode) -> cst.BaseExpression:
    # Operands: [func, arg0, arg1...]
    if not op.operands:
      return cst.Call(func=cst.Name("unknown"))

    func_expr = self._resolve_operand(op.operands[0].name)
    args = []
    for op_val in op.operands[1:]:
      arg_expr = self._resolve_operand(op_val.name)
      args.append(cst.Arg(value=arg_expr))

    return cst.Call(func=func_expr, args=args)

  def _expr_sw_op(self, op: OperationNode) -> cst.BaseExpression:
    type_name = self._get_attr(op, "type") or '"unknown"'
    if "binop." in type_name:
      return self._expr_binop(op, type_name)

    dotted_path = type_name.strip('"')
    func_node = self._create_dotted_name(dotted_path)
    args = []
    for op_val in op.operands:
      arg_expr = self._resolve_operand(op_val.name)
      args.append(cst.Arg(value=arg_expr))

    return cst.Call(func=func_node, args=args)

  def _expr_binop(self, op: OperationNode, type_attr: str) -> cst.BaseExpression:
    op_code = type_attr.strip('"').replace("binop.", "")
    if len(op.operands) < 2:
      return cst.Name("error_binop")

    lhs_expr = self._resolve_operand(op.operands[0].name)
    rhs_expr = self._resolve_operand(op.operands[1].name)

    op_map = {
      "add": cst.Add(),
      "sub": cst.Subtract(),
      "mul": cst.Multiply(),
      "div": cst.Divide(),
      "floordiv": cst.FloorDivide(),
      "mod": cst.Modulo(),
      "pow": cst.Power(),
      "matmul": cst.MatrixMultiply(),
      "lshift": cst.LeftShift(),
      "rshift": cst.RightShift(),
      "and": cst.BitAnd(),
      "or": cst.BitOr(),
      "xor": cst.BitXor(),
    }
    return cst.BinaryOperation(left=lhs_expr, operator=op_map.get(op_code, cst.Add()), right=rhs_expr)

  def _convert_setattr(self, op: OperationNode) -> cst.SimpleStatementLine:
    if len(op.operands) < 2:
      return cst.SimpleStatementLine(body=[cst.Pass()])

    obj_expr = self._resolve_operand(op.operands[0].name)
    val_expr = self._resolve_operand(op.operands[1].name)
    attr_name = (self._get_attr(op, "name") or "unknown").strip('"')

    target = cst.Attribute(value=obj_expr, attr=cst.Name(attr_name))
    assign = cst.Assign(targets=[cst.AssignTarget(target=target)], value=val_expr)
    return cst.SimpleStatementLine(body=[assign])

  def _convert_return(self, op: OperationNode) -> cst.SimpleStatementLine:
    val_node = None
    if op.operands:
      val_node = self._resolve_operand(op.operands[0].name)
    return cst.SimpleStatementLine(body=[cst.Return(value=val_node)])

  def _convert_class_def(self, op: OperationNode) -> cst.ClassDef:
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
      body_stmts = self._convert_block(op.regions[0].blocks[0])

    if not body_stmts:
      body_stmts = [cst.SimpleStatementLine(body=[cst.Pass()])]

    return cst.ClassDef(name=cst.Name(class_name), bases=base_nodes, body=cst.IndentedBlock(body=body_stmts))

  def _convert_func_def(self, op: OperationNode) -> cst.FunctionDef:
    name_attr = self._get_attr(op, "sym_name")
    func_name = name_attr.strip('"') if name_attr else "unknown_func"

    # Scope Reset: Logic for functions should be isolated from parent naming context (mostly)
    # This prevents 'self' from becoming 'self13' due to collisions with constructor context.
    prev_ctx = self.ctx
    self.ctx = NamingContext()

    params = []
    body_stmts = []

    try:
      if op.regions and op.regions[0].blocks:
        block0 = op.regions[0].blocks[0]
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
      # Restore if we were nested (though func defs usually aren't nested in this IR structure yet)
      # But cleaning up is good practice in case we reuse generator.
      # Actually, since NamingContext also tracks deferred_exprs counts globally in the current implementation,
      # this scoping might affect inlining counts if not handled carefully, but usage_counts is separate.
      # Usage counts are global to module, so that's fine. Naming is the only thing we reset.
      self.ctx = prev_ctx

    return cst.FunctionDef(
      name=cst.Name(func_name), params=cst.Parameters(params=params), body=cst.IndentedBlock(body=body_stmts)
    )

  def _get_attr(self, op: OperationNode, key: str) -> Optional[str]:
    for attr in op.attributes:
      if attr.name == key:
        if isinstance(attr.value, list):
          return f"[{', '.join(attr.value)}]"
        return attr.value
    return None

  def _create_dotted_name(self, path: str) -> cst.BaseExpression:
    parts = path.split(".")
    if not parts:
      return cst.Name("unknown")
    node: cst.BaseExpression = cst.Name(parts[0])
    for p in parts[1:]:
      node = cst.Attribute(value=node, attr=cst.Name(p))
    return node
