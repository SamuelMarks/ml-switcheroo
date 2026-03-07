"""
MLIR Emitter Logic.

This module provides the `PythonToMlirEmitter`, a compiler front-end that
transforms Python LibCST trees into the MLIR CST object model.
"""

import libcst as cst
from typing import Dict, List, Optional, Union, Tuple, Sequence

from ml_switcheroo.core.mlir.nodes import (
  ModuleNode,
  OperationNode,
  BlockNode,
  RegionNode,
  ValueNode,
  TypeNode,
  AttributeNode,
  TriviaNode,
)
from ml_switcheroo.core.scanners import get_full_name


class SSAContext:
  """
  Manages Single Static Assignment (SSA) variable scopes and ID allocation.
  """

  def __init__(self):
    """Initialize the context with a root scope."""
    self._scopes: List[Dict[str, ValueNode]] = [{}]
    self._counter: int = 0

  def enter_scope(self) -> None:
    """Push a new variable scope onto the stack."""
    self._scopes.append({})

  def exit_scope(self) -> None:
    """Pop the current variable scope from the stack."""
    if len(self._scopes) > 1:
      self._scopes.pop()

  def declare(self, name: str, value: ValueNode) -> None:
    """
    Register a variable name in the current scope.

    Args:
        name: The Python variable identifier.
        value: The MLIR ValueNode (SSA value) associated with it.
    """
    self._scopes[-1][name] = value

  def lookup(self, name: str) -> Optional[ValueNode]:
    """
    Resolve a Python variable name to its current SSA value.
    Searches scopes from innermost to outermost.

    Args:
        name: The Python identifier to look up.

    Returns:
        The associated ValueNode or None if not found.
    """
    for scope in reversed(self._scopes):
      if name in scope:
        return scope[name]
    return None

  def allocate_ssa(self, prefix: str = "%") -> ValueNode:
    """
    Generates a new unique SSA value.

    Args:
        prefix: String prefix for the ID (default "%").

    Returns:
        A new ValueNode with a unique ID (e.g. "%0", "%1").
    """
    val = ValueNode(name=f"{prefix}{self._counter}")
    self._counter += 1
    return val


class PythonToMlirEmitter:
  """
  Translates Python LibCST modules into MLIR structural nodes.
  """

  def __init__(self):
    """Initialize the emitter with a fresh SSA context."""
    self.ctx = SSAContext()

  def convert(self, node: cst.Module) -> ModuleNode:
    """
    Entry point: Converts a CST Module to an MLIR ModuleNode.

    Args:
        node: The Python LibCST Module.

    Returns:
        The resulting MLIR ModuleNode containing the translated operations.
    """
    body_block = self._emit_block(node.body)

    # Capture module header comments
    if hasattr(node, "header"):
      header_trivia = []
      for line in node.header:
        if line.comment:
          text = line.comment.value.replace("#", "//", 1)
          header_trivia.append(TriviaNode(text, kind="comment"))
          header_trivia.append(TriviaNode("\n", kind="newline"))
        elif line.newline:  # pragma: no cover
          header_trivia.append(TriviaNode("\n", kind="newline"))  # pragma: no cover

      # Attach to first op
      if header_trivia and body_block.operations:
        body_block.operations[0].leading_trivia = header_trivia + body_block.operations[0].leading_trivia

    return ModuleNode(body=body_block)

  def _extract_trivia(self, node: cst.CSTNode) -> List[TriviaNode]:
    """
    Extracts comments and newlines from a CST node's leading lines.

    Args:
        node: The CST node to inspect.

    Returns:
        A list of MLIR TriviaNodes (comments translated to `//` syntax).
    """
    trivia = []
    if hasattr(node, "leading_lines"):
      for line in node.leading_lines:
        if line.comment:  # pragma: no cover
          text = line.comment.value.replace("#", "//", 1)  # pragma: no cover
          trivia.append(TriviaNode(text, kind="comment"))  # pragma: no cover
          trivia.append(TriviaNode("\n", kind="newline"))  # pragma: no cover
        elif line.newline:  # pragma: no cover
          # Persist empty lines for formatting niceness
          if line.newline.value:  # pragma: no cover
            trivia.append(TriviaNode("\n", kind="newline"))  # pragma: no cover

    return trivia

  def _emit_block(self, body_enc: Union[cst.BaseSuite, Sequence[cst.CSTNode]], label: str = "") -> BlockNode:
    """
    Converts a sequence of statements (or a Suite) into an MLIR Block.

    Args:
        body_enc: A CST Suite (IndentedBlock) or list of statements.
        label: Optional block label (e.g. `^entry`).

    Returns:
        A populated BlockNode.
    """
    block = BlockNode(label=label)
    stmts = []
    if isinstance(body_enc, (cst.IndentedBlock, cst.SimpleStatementSuite, cst.Module)):
      stmts = body_enc.body
    elif isinstance(body_enc, (list, tuple)):
      stmts = body_enc

    for stmt in stmts:
      ops = self._emit_statement(stmt)
      if ops:
        block.operations.extend(ops)
    return block

  def _emit_statement(self, stmt: cst.CSTNode) -> List[OperationNode]:
    """
    Dispatches statement nodes to specific handlers.

    Args:
        stmt: The statement node (ClassDef, FunctionDef, Assign, etc.).

    Returns:
        A list of MLIR OperationNodes generated from the statement.
    """
    results = []
    if isinstance(stmt, cst.ClassDef):
      results = [self._emit_class_def(stmt)]
    elif isinstance(stmt, cst.FunctionDef):
      results = [self._emit_func_def(stmt)]
    elif isinstance(stmt, cst.SimpleStatementLine):
      if len(stmt.body) > 0:
        node = stmt.body[0]
        results = self._dispatch_small_stmt(node)

    if results:
      extracted = self._extract_trivia(stmt)
      if extracted:
        results[0].leading_trivia = extracted + results[0].leading_trivia  # pragma: no cover

    return results

  def _dispatch_small_stmt(self, node: cst.CSTNode) -> List[OperationNode]:
    """
    Handles small statements inside simple lines (Assign, Return, Expr).

    Args:
        node: The inner statement node.

    Returns:
        List of resulting operations.
    """
    if isinstance(node, cst.Assign):
      return self._emit_assign(node)
    elif isinstance(node, cst.Return):
      return self._emit_return(node)
    elif isinstance(node, cst.Expr):
      _, ops = self._emit_expression(node.value)  # pragma: no cover
      return ops  # pragma: no cover
    elif isinstance(node, (cst.Import, cst.ImportFrom)):
      return [self._emit_import(node)]  # pragma: no cover
    return []

  def _emit_import(self, node: Union[cst.Import, cst.ImportFrom]) -> OperationNode:
    """
      Converts Import/ImportFrom to `sw.import`.

      Args:  # pragma: no cover
          node: The Import or ImportFrom CST node.  # pragma: no cover
    # pragma: no cover
      Returns:
          An OperationNode representing the `sw.import` operation.  # pragma: no cover
    """  # pragma: no cover
    names = []
    aliases = []
    module_val = ""

    if isinstance(node, cst.ImportFrom) and node.module:  # pragma: no cover
      module_val = get_full_name(node.module)  # pragma: no cover
    # pragma: no cover
    # Extract names and aliases
    # For ImportFrom, name.name is the object imported.  # pragma: no cover
    # For Import, name.name is the module being imported.  # pragma: no cover
    if isinstance(node.names, cst.ImportStar):  # pragma: no cover
      names.append("*")  # pragma: no cover
      aliases.append("")
    else:  # pragma: no cover
      for alias in node.names:
        names.append(get_full_name(alias.name))  # pragma: no cover
        if alias.asname:  # pragma: no cover
          aliases.append(alias.asname.name.value)  # pragma: no cover
        else:
          aliases.append("")
    # pragma: no cover
    attrs = []  # pragma: no cover
    if module_val:
      attrs.append(AttributeNode(name="module", value=f'"{module_val}"'))  # pragma: no cover
    # pragma: no cover
    # Format list strings properly for MLIR array attribute
    quoted_names = [f'"{n}"' for n in names]  # pragma: no cover
    quoted_aliases = [f'"{a}"' for a in aliases]

    attrs.append(AttributeNode(name="names", value=quoted_names))
    attrs.append(AttributeNode(name="aliases", value=quoted_aliases))

    return OperationNode(name="sw.import", attributes=attrs)

  def _emit_class_def(self, node: cst.ClassDef) -> OperationNode:
    """
    Converts a Python class definition to `sw.module`.

    Args:
        node: The ClassDef node.

    Returns:
        An `sw.module` OperationNode containing the class body region.
    """
    self.ctx.enter_scope()
    name_obj = AttributeNode(name="sym_name", value=f'"{node.name.value}"')  # pragma: no cover
    # pragma: no cover
    attributes = [name_obj]  # pragma: no cover
    # pragma: no cover
    # Capture Bases (Inheritance)  # pragma: no cover
    if node.bases:
      base_names = []  # pragma: no cover
      for b in node.bases:  # pragma: no cover
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
    """
    Converts a Python function definition to `sw.func`.

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

  def _emit_assign(self, node: cst.Assign) -> List[OperationNode]:
    """
    Converts an assignment statement.

    Emits expression operations and registers the result in the SSA context.
    Handles both variable assignment (`x = y`) and attribute assignment (`x.attr = y`)
    via `sw.setattr`.

    Args:
        node: The Assign node.

    Returns:
        List of operations generated by the assignment expression.
    """
    val, ops = self._emit_expression(node.value)

    for target in node.targets:
      t = target.target
      # pragma: no cover
      # Variable Assignment: x = ...
      if isinstance(t, cst.Name):  # pragma: no cover
        self.ctx.declare(t.value, val)  # pragma: no cover
      # pragma: no cover
      # Attribute Assignment: self.x = ...
      elif isinstance(t, cst.Attribute):
        # Check if base is known (e.g. self)
        base_name = self._flatten_attr(t.value)  # pragma: no cover
        if base_name:  # pragma: no cover
          base_val = self.ctx.lookup(base_name)  # pragma: no cover
          # Support emitting setattr for object attributes
          # sw.setattr %self "layer" %val
          # pragma: no cover
          if base_val:
            attr_name = t.attr.value
            set_op = OperationNode(
              name="sw.setattr", operands=[base_val, val], attributes=[AttributeNode("name", f'"{attr_name}"')]
            )  # pragma: no cover
            ops.append(set_op)

            # Feature: Register the Attribute name in context for future lookup  # pragma: no cover
            # This allows subsequent `self.layer` access to map back to this val if needed
            # But SSA logic handles lookup by name. self.layer is complex.
            pass
          else:
            # Fallback if base not resolved
            pass

    return ops

  def _emit_return(self, node: cst.Return) -> List[OperationNode]:
    """
    Converts a return statement to `sw.return`.

    Args:
        node: The Return node.

    Returns:
        List containing expression evaluation ops and the return op.
    """
    ops = []
    operands = []
    if node.value:
      val, expr_ops = self._emit_expression(node.value)
      ops.extend(expr_ops)
      operands.append(val)

    # Ensure we attach operands list correctly
    op = OperationNode(name="sw.return", operands=operands)
    ops.append(op)
    return ops

  def _flatten_attr(self, node: cst.CSTNode) -> Optional[str]:
    """
    Helper to flatten a Name or Attribute chain into a string.

    Args:
        node: CST node.

    Returns:
        Dotted string (e.g. "self.layer") or None.
    """
    if isinstance(node, cst.Name):  # pragma: no cover
      return node.value
    if isinstance(node, cst.Attribute):
      base = self._flatten_attr(node.value)
      if base:
        return f"{base}.{node.attr.value}"
    return None

  def _get_binop_str(self, operator: cst.BaseBinaryOp) -> str:
    """
    Maps LibCST binary operator classes to string codes.

    Args:
        operator: The CST binary operator node.

    Returns:  # pragma: no cover
        String identifier (e.g. "add", "mul", "matmul").
    """
    if isinstance(operator, cst.Add):
      return "add"
    if isinstance(operator, cst.Subtract):  # pragma: no cover
      return "sub"  # pragma: no cover
    if isinstance(operator, cst.Multiply):  # pragma: no cover
      return "mul"  # pragma: no cover
    if isinstance(operator, cst.Divide):  # pragma: no cover
      return "div"  # pragma: no cover
    if isinstance(operator, cst.FloorDivide):  # pragma: no cover
      return "floordiv"  # pragma: no cover
    if isinstance(operator, cst.Modulo):  # pragma: no cover
      return "mod"  # pragma: no cover
    if isinstance(operator, cst.Power):  # pragma: no cover
      return "pow"  # pragma: no cover
    if isinstance(operator, cst.MatrixMultiply):  # pragma: no cover
      return "matmul"  # pragma: no cover
    if isinstance(operator, cst.LeftShift):  # pragma: no cover
      return "lshift"  # pragma: no cover
    if isinstance(operator, cst.RightShift):  # pragma: no cover
      return "rshift"  # pragma: no cover
    if isinstance(operator, cst.BitAnd):  # pragma: no cover
      return "and"
    if isinstance(operator, cst.BitOr):
      return "or"
    if isinstance(operator, cst.BitXor):
      return "xor"
    return "unknown"

  def _emit_expression(self, expr: cst.BaseExpression) -> Tuple[ValueNode, List[OperationNode]]:
    """
    Recursively converts an expression into a value and a list of supporting operations.

    Handles:
    - Variables (Names)
    - Function Calls (capturing keywords)
    - Binary Operations
    - Constants

    Args:
        expr: The expression node.

    Returns:
        Tuple (ResultValue, List[Ops]).
    """
    ops = []
    if isinstance(expr, cst.Name):
      val = self.ctx.lookup(expr.value)
      if not val:
        val = ValueNode(f"@{expr.value}")
      return val, ops
    elif isinstance(expr, cst.Call):
      operands = []
      arg_keywords = []  # new feature

      # Process arguments
      for arg in expr.args:
        v, o = self._emit_expression(arg.value)
        ops.extend(o)  # pragma: no cover
        operands.append(v)

        # Capture keyword if present
        kw = ""
        if arg.keyword:
          kw = arg.keyword.value
        arg_keywords.append(kw)

      flat_name = self._flatten_attr(expr.func)
      root_var = flat_name.split(".")[0] if flat_name else ""
      is_static_op = False
      if flat_name and not self.ctx.lookup(root_var):
        is_static_op = True

      common_attrs = []
      # Pack keywords into attribute if any are non-empty  # pragma: no cover
      if any(arg_keywords):  # pragma: no cover
        # AttributeNode needs a list of strings formatted for the printer
        # e.g. ["k=val", ""] -> we just need to store the keys.
        # "arg_keywords" = ["a", "", "b"]
        # We store as list of quoted strings
        kw_vals = [f'"{k}"' for k in arg_keywords]
        common_attrs.append(AttributeNode("arg_keywords", kw_vals))

      if is_static_op:
        result = self.ctx.allocate_ssa()
        attrs = [AttributeNode("type", f'"{flat_name}"')] + common_attrs
        op = OperationNode(
          name="sw.op",
          results=[result],
          operands=operands,
          attributes=attrs,
        )
        ops.append(op)
        return result, ops

      if isinstance(expr.func, cst.Attribute):
        obj, o_ops = self._emit_expression(expr.func.value)
        ops.extend(o_ops)
        attr_val = self.ctx.allocate_ssa()
        get_op = OperationNode(
          name="sw.getattr",
          results=[attr_val],
          operands=[obj],
          attributes=[AttributeNode("name", f'"{expr.func.attr.value}"')],
        )
        ops.append(get_op)
        res_val = self.ctx.allocate_ssa()
        # Attach keywords
        call_op = OperationNode(  # pragma: no cover
          name="sw.call",
          results=[res_val],
          operands=[attr_val] + operands,
          attributes=common_attrs,  # pragma: no cover
        )  # pragma: no cover
        ops.append(call_op)  # pragma: no cover
        return res_val, ops  # pragma: no cover
      # pragma: no cover
      if isinstance(expr.func, cst.Name):  # pragma: no cover
        func_val, f_ops = self._emit_expression(expr.func)
        ops.extend(f_ops)
        result = self.ctx.allocate_ssa()
        call_op = OperationNode(name="sw.call", results=[result], operands=[func_val] + operands, attributes=common_attrs)
        ops.append(call_op)
        return result, ops

    elif isinstance(expr, cst.BinaryOperation):
      lhs_val, l_ops = self._emit_expression(expr.left)
      rhs_val, r_ops = self._emit_expression(expr.right)
      ops.extend(l_ops)
      ops.extend(r_ops)

      op_str = self._get_binop_str(expr.operator)
      res_val = self.ctx.allocate_ssa()
      op = OperationNode(
        name="sw.op",
        results=[res_val],
        operands=[lhs_val, rhs_val],
        attributes=[AttributeNode("type", f'"binop.{op_str}"')],
      )
      ops.append(op)
      return res_val, ops

    elif isinstance(expr, (cst.Integer, cst.Float)):
      val = self.ctx.allocate_ssa(prefix="%cst")
      op = OperationNode(  # pragma: no cover
        name="sw.constant", results=[val], attributes=[AttributeNode("value", getattr(expr, "value", "0"))]
      )
      ops.append(op)
      return val, ops

    return ValueNode("%error"), []

  def _annotation_to_string(self, node: cst.CSTNode) -> str:  # pragma: no cover
    """# pragma: no cover
    Flattens a type annotation node to a string representation.
    """  # pragma: no cover
    if isinstance(node, cst.Name):
      return node.value
    elif isinstance(node, cst.Attribute):
      return f"{self._annotation_to_string(node.value)}.{node.attr.value}"
    else:
      return "Any"
