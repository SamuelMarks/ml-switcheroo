"""Helpers for StructuralTransformer."""

from typing import List, Tuple, Optional
import libcst as cst
from typing import TYPE_CHECKING, Any


class StructuralTransformerHelpersMixin:
  """Docstring."""

  if TYPE_CHECKING:

    def _create_dotted_name(self, name: str) -> Any:
      """Docstring."""
      ...

  def _strip_argument_from_signature(self, node: cst.FunctionDef, arg_name: str) -> cst.FunctionDef:
    """Removes an argument by name from the function definition."""
    params = list(node.params.params)
    new_params = [p for p in params if not (isinstance(p.name, cst.Name) and p.name.value == arg_name)]
    return self._fix_comma(node, new_params)

  def _inject_argument_to_signature(
    self, node: cst.FunctionDef, arg_name: str, annotation: Optional[str]
  ) -> cst.FunctionDef:
    """Injects a new argument after 'self'."""
    params = list(node.params.params)
    insert_idx = 0
    if params and params[0].name.value == "self":
      insert_idx = 1

    anno_node = cst.Annotation(annotation=self._create_dotted_name(annotation)) if annotation else None
    new_param = cst.Param(name=cst.Name(arg_name), annotation=anno_node, comma=cst.MaybeSentinel.DEFAULT)

    params.insert(insert_idx, new_param)
    return self._fix_comma(node, params)

  def _fix_comma(self, node: cst.FunctionDef, params: List[cst.Param]) -> cst.FunctionDef:
    """Ensures logic commas are correct for argument lists."""
    for i in range(len(params) - 1):
      if params[i].comma == cst.MaybeSentinel.DEFAULT:
        params[i] = params[i].with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

    if len(params) > 0:
      last = params[-1]
      if last.comma != cst.MaybeSentinel.DEFAULT:
        params[-1] = last.with_changes(comma=cst.MaybeSentinel.DEFAULT)

    new_params_node = node.params.with_changes(params=params)
    return node.with_changes(params=new_params_node)

  def _apply_preamble(self, node: cst.FunctionDef, stmts_code: List[str]) -> cst.FunctionDef:
    """Injects source code statements at the start of the function body."""
    new_stmts = []  # type: ignore
    for code in stmts_code:
      try:
        mod = cst.parse_module(code)
        new_stmts.extend(mod.body)
      except Exception:
        pass

    return self._inject_stmts_to_body(node, new_stmts)

  def _inject_stmts_to_body(self, node: cst.FunctionDef, new_stmts: List[cst.BaseStatement]) -> cst.FunctionDef:
    """Helper to insert statements respecting docstrings."""
    if isinstance(node.body, cst.SimpleStatementSuite):
      node = self._convert_to_indented_block(node)

    existing = list(node.body.body)
    idx = 0
    # Skip docstring if exists
    if existing and isinstance(existing[0], cst.SimpleStatementLine) and len(existing[0].body) == 1:
      expr = existing[0].body[0]
      if isinstance(expr, cst.Expr) and isinstance(expr.value, (cst.SimpleString, cst.ConcatenatedString)):
        idx = 1

    final_body = existing[:idx] + new_stmts + existing[idx:]
    return node.with_changes(body=node.body.with_changes(body=final_body))

  def _convert_to_indented_block(self, node: cst.FunctionDef) -> cst.FunctionDef:
    """Unwraps simple one-liners to indented blocks for injection."""
    if isinstance(node.body, cst.SimpleStatementSuite):
      new_stmts = [cst.SimpleStatementLine(body=[s]) for s in node.body.body]
      return node.with_changes(body=cst.IndentedBlock(body=new_stmts))
    return node

  def _ensure_super_init(self, node: cst.FunctionDef) -> cst.FunctionDef:
    """Injects super().__init__() call."""
    if self._has_super_init(node):
      return node
    stmt = cst.SimpleStatementLine(
      body=[
        cst.Expr(value=cst.Call(func=cst.Attribute(value=cst.Call(func=cst.Name("super")), attr=cst.Name("__init__"))))
      ]
    )
    return self._inject_stmts_to_body(node, [stmt])

  def _strip_super_init(self, node: cst.FunctionDef) -> cst.FunctionDef:
    """Removes super().__init__() call."""
    if isinstance(node.body, cst.SimpleStatementSuite):
      return node
    if not hasattr(node.body, "body"):
      return node

    new_body = [s for s in node.body.body if not self._is_super_init_call(s)]
    return node.with_changes(body=node.body.with_changes(body=new_body))

  def _has_super_init(self, node: cst.FunctionDef) -> bool:
    """Checks for presence of super().__init__()."""
    if hasattr(node.body, "body"):
      for stmt in node.body.body:
        if self._is_super_init_call(stmt):
          return True
    return False

  def _is_super_init_call(self, stmt: cst.CSTNode) -> bool:
    """Detects super init pattern in a statement node."""
    if isinstance(stmt, cst.SimpleStatementLine) and len(stmt.body) == 1:
      expr_or_assign = stmt.body[0]
      val = expr_or_assign.value if hasattr(expr_or_assign, "value") else None
      if isinstance(val, cst.Call) and isinstance(val.func, cst.Attribute) and val.func.attr.value == "__init__":
        inner = val.func.value
        if isinstance(inner, cst.Call) and isinstance(inner.func, cst.Name) and inner.func.value == "super":
          return True
    return False

  def _update_docstring(self, node: cst.FunctionDef, args: List[Tuple[str, Optional[str]]]) -> cst.FunctionDef:
    """Appends argument descriptions to the docstring."""
    if not hasattr(node.body, "body") or not node.body.body:
      return node
    stmt = node.body.body[0]
    if not isinstance(stmt, cst.SimpleStatementLine) or len(stmt.body) != 1:
      return node
    expr = stmt.body[0]
    if not isinstance(expr, cst.Expr) or not isinstance(expr.value, cst.SimpleString):
      return node

    val = expr.value.value
    # Simple string manipulation to append args
    if '"""' in val:
      content = val.replace('"""', "")
      injection = "\n" + "\n".join([f"    {n}: Injected." for n, _ in args])
      new_val = f'"""{content}{injection}\n    """'
      new_expr = expr.with_changes(value=cst.SimpleString(new_val))
      new_stmt = stmt.with_changes(body=[expr.with_changes(value=new_expr)])
      return node.with_changes(body=node.body.with_changes(body=[new_stmt] + list(node.body.body[1:])))

    return node
