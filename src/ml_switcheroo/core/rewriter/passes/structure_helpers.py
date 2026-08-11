"""Helper utilities and structural manipulation tools for StructuralTransformer.

This module provides `StructuralTransformerHelpersMixin`, a mixin class that
houses several AST-manipulation helpers using `libcst`. It assists in altering
function signatures (stripping/injecting arguments), modifying commas, prepending
preambles, injecting statements into function bodies, wrapping/unwrapping
one-liners, managing `super().__init__()` calls, and updating docstrings.
"""

from typing import List, Tuple, Optional
import libcst as cst
from typing import TYPE_CHECKING, Any


class StructuralTransformerHelpersMixin:
  """Mixin class providing AST-manipulation helper methods for structural code transformations.

  This mixin contains common utilities used during AST rewriting passes, particularly
  when restructuring class methods, function definitions, signatures, and body statements.
  It relies on the availability of LibCST node builders and AST parsing capabilities.
  """

  if TYPE_CHECKING:

    def _create_dotted_name(self, name: str) -> Any:
      """Creates a dotted name or attribute node from a dot-separated string representation.

      Args:
          self: The mixin instance.
          name: A dot-separated string representing the name (e.g., 'foo.bar.baz').

      Returns:
          A CST node representation of the dotted name, typically a `cst.Attribute`
          or `cst.Name` node.
      """
      return None

  def _strip_argument_from_signature(self, node: cst.FunctionDef, arg_name: str) -> cst.FunctionDef:
    """Removes an argument by name from the function definition signature.

    Args:
        node: The original `cst.FunctionDef` node whose signature is to be modified.
        arg_name: The string name of the argument/parameter to remove.

    Returns:
        A new `cst.FunctionDef` node with the specified argument removed and proper comma spacing.
    """
    params = list(node.params.params)
    new_params = [p for p in params if not (isinstance(p.name, cst.Name) and p.name.value == arg_name)]
    return self._fix_comma(node, new_params)

  def _inject_argument_to_signature(
    self, node: cst.FunctionDef, arg_name: str, annotation: Optional[str]
  ) -> cst.FunctionDef:
    """Injects a new argument after 'self' in the function's parameter list.

    If the first parameter is 'self', the new parameter is inserted immediately after it.
    Otherwise, it is inserted at the beginning of the parameter list.

    Args:
        node: The `cst.FunctionDef` node whose parameter list will be modified.
        arg_name: The name of the parameter/argument to inject.
        annotation: An optional string representation of the type annotation for the parameter.

    Returns:
        A new `cst.FunctionDef` node with the parameter injected and correct formatting.
    """
    params = list(node.params.params)
    insert_idx = 0
    if params and params[0].name.value == "self":
      insert_idx = 1

    anno_node = cst.Annotation(annotation=self._create_dotted_name(annotation)) if annotation else None
    new_param = cst.Param(name=cst.Name(arg_name), annotation=anno_node, comma=cst.MaybeSentinel.DEFAULT)

    params.insert(insert_idx, new_param)
    return self._fix_comma(node, params)

  def _fix_comma(self, node: cst.FunctionDef, params: List[cst.Param]) -> cst.FunctionDef:
    """Ensures that syntax commas are logically correct and properly spaced for parameter lists.

    This method walks through the list of parameters, ensuring that trailing commas are
    only present on non-terminal parameters and that spacing whitespace after commas is
    correctly configured.

    Args:
        node: The `cst.FunctionDef` node being updated.
        params: A list of `cst.Param` nodes that form the new parameter list.

    Returns:
        A new `cst.FunctionDef` node containing the updated, correctly-spaced parameter list.
    """
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
    """Injects source code statements at the start of the function body.

    Parses the list of statement strings using LibCST and prepends them as AST nodes
    immediately at the start of the target function body (respecting any existing docstring).

    Args:
        node: The target `cst.FunctionDef` node where the preamble is injected.
        stmts_code: A list of source code strings representing the statements to parse and inject.

    Returns:
        A modified `cst.FunctionDef` node containing the injected preamble statements.
    """
    new_stmts = []  # type: ignore
    for code in stmts_code:
      try:
        mod = cst.parse_module(code)
        new_stmts.extend(mod.body)
      except Exception:
        pass

    return self._inject_stmts_to_body(node, new_stmts)

  def _inject_stmts_to_body(self, node: cst.FunctionDef, new_stmts: List[cst.BaseStatement]) -> cst.FunctionDef:
    """Inserts a list of statements into a function's body while respecting any existing docstring.

    If the function body is a single-line statement suite, it will be converted to an
    indented block first. If the first statement of the function is a docstring, the
    new statements will be injected directly after that docstring instead of before it.

    Args:
        node: The `cst.FunctionDef` node whose body will receive the new statements.
        new_stmts: A list of LibCST `cst.BaseStatement` nodes to be injected.

    Returns:
        A modified `cst.FunctionDef` node containing the injected statements in its body.
    """
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
    """Unwraps a simple, single-line function body into a multi-line indented block.

    This is a prerequisite for injecting multiple statements into a previously one-line function.

    Args:
        node: The `cst.FunctionDef` node containing the body to convert.

    Returns:
        A new `cst.FunctionDef` node with its body converted to a `cst.IndentedBlock` if it
        was a `cst.SimpleStatementSuite`, or the original node unchanged.
    """
    if isinstance(node.body, cst.SimpleStatementSuite):
      new_stmts = [cst.SimpleStatementLine(body=[s]) for s in node.body.body]
      return node.with_changes(body=cst.IndentedBlock(body=new_stmts))
    return node

  def _ensure_super_init(self, node: cst.FunctionDef) -> cst.FunctionDef:
    """Injects a call to `super().__init__()` at the start of the function if not already present.

    Ensures that inherited initialization logic is preserved by prepending a call to
    `super().__init__()` in the body, placing it after any existing docstring.

    Args:
        node: The `cst.FunctionDef` node representing the `__init__` method to modify.

    Returns:
        A new `cst.FunctionDef` node with `super().__init__()` injected if missing, or the
        original node unchanged if a super-init call was already detected.
    """
    if self._has_super_init(node):
      return node
    stmt = cst.SimpleStatementLine(
      body=[
        cst.Expr(value=cst.Call(func=cst.Attribute(value=cst.Call(func=cst.Name("super")), attr=cst.Name("__init__"))))
      ]
    )
    return self._inject_stmts_to_body(node, [stmt])

  def _strip_super_init(self, node: cst.FunctionDef) -> cst.FunctionDef:
    """Removes the `super().__init__()` call from the function body.

    Useful when restructuring initializer sequences where a standard `super()` call is no
    longer desired or is replaced.

    Args:
        node: The `cst.FunctionDef` node from which to strip the super init call.

    Returns:
        A new `cst.FunctionDef` node with any matching `super().__init__()` statement removed.
    """
    if isinstance(node.body, cst.SimpleStatementSuite):
      return node
    if not hasattr(node.body, "body"):
      return node

    new_body = [s for s in node.body.body if not self._is_super_init_call(s)]
    return node.with_changes(body=node.body.with_changes(body=new_body))

  def _has_super_init(self, node: cst.FunctionDef) -> bool:
    """Checks for the presence of a `super().__init__()` call within the function body.

    Walks through the high-level statements in the body of the function to detect if
    a super class initialization is being invoked.

    Args:
        node: The `cst.FunctionDef` node to inspect.

    Returns:
        True if a `super().__init__()` call statement is found; False otherwise.
    """
    if hasattr(node.body, "body"):
      for stmt in node.body.body:
        if self._is_super_init_call(stmt):
          return True
    return False

  def _is_super_init_call(self, stmt: cst.CSTNode) -> bool:
    """Detects if a given statement node is an invocation of `super().__init__()`.

    Args:
        stmt: The CST node representing a statement to inspect.

    Returns:
        True if the statement represents a standard `super().__init__()` call; False otherwise.
    """
    if isinstance(stmt, cst.SimpleStatementLine) and len(stmt.body) == 1:
      expr_or_assign = stmt.body[0]
      val = expr_or_assign.value if hasattr(expr_or_assign, "value") else None
      if isinstance(val, cst.Call) and isinstance(val.func, cst.Attribute) and val.func.attr.value == "__init__":
        inner = val.func.value
        if isinstance(inner, cst.Call) and isinstance(inner.func, cst.Name) and inner.func.value == "super":
          return True
    return False

  def _update_docstring(self, node: cst.FunctionDef, args: List[Tuple[str, Optional[str]]]) -> cst.FunctionDef:
    """Appends descriptive definitions for injected arguments to the existing docstring.

    Modifies the function's docstring if it exists, inserting formatted parameter entries
    for newly introduced arguments.

    Args:
        node: The `cst.FunctionDef` node whose docstring will be updated.
        args: A list of (arg_name, annotation) tuples representing the injected arguments.

    Returns:
        A modified `cst.FunctionDef` node containing the updated docstring, or the
        original node if no docstring is present or if its structure cannot be modified.
    """
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
