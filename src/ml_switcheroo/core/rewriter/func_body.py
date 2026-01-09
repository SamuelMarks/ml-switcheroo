"""
Function Body Rewriting Logic.

Helpers for manipulating function bodies:
1. Preamble injection.
2. Super Init handling (Injection/Stripping).
3. Indented Block conversion.
"""

from typing import List
import libcst as cst
from libcst import BaseSuite, SimpleStatementSuite, IndentedBlock, SimpleStatementLine

from ml_switcheroo.core.escape_hatch import EscapeHatch


class FuncBodyMixin:
  """
  Mixin for manipulating function bodies.
  """

  def _is_body_accessible(self, body: BaseSuite) -> bool:
    """Checks if body is an indented block."""
    return isinstance(body, IndentedBlock)

  def _convert_to_indented_block(self, node: cst.FunctionDef) -> cst.FunctionDef:
    """
    Converts one-line bodies to indented blocks.
    """
    if isinstance(node.body, SimpleStatementSuite):
      new_body_stmts = []
      for stmt in node.body.body:
        new_body_stmts.append(cst.SimpleStatementLine(body=[stmt]))

      new_block = cst.IndentedBlock(body=new_body_stmts)
      return node.with_changes(body=new_block)
    return node

  def _ensure_super_init(self, node: cst.FunctionDef) -> cst.FunctionDef:
    """
    Injects super().__init__() at the start.
    """
    if isinstance(node.body, SimpleStatementSuite):
      node = self._convert_to_indented_block(node)

    if self._has_super_init(node):
      return node

    super_stmt = cst.SimpleStatementLine(
      body=[
        cst.Expr(
          value=cst.Call(
            func=cst.Attribute(
              value=cst.Call(func=cst.Name("super")),
              attr=cst.Name("__init__"),
            )
          )
        )
      ]
    )

    stmts = list(node.body.body)
    insert_idx = 0
    if (
      stmts
      and isinstance(stmts[0], cst.SimpleStatementLine)
      and isinstance(stmts[0].body[0], cst.Expr)
      and isinstance(stmts[0].body[0].value, (cst.SimpleString, cst.ConcatenatedString))
    ):
      insert_idx = 1

    stmts.insert(insert_idx, super_stmt)
    return node.with_changes(body=node.body.with_changes(body=stmts))

  def _strip_super_init(self, node: cst.FunctionDef) -> cst.FunctionDef:
    """
    Removes super().__init__() calls.
    """
    if isinstance(node.body, SimpleStatementSuite):
      return node

    if not hasattr(node.body, "body"):
      return node

    new_body_stmts = []
    skip_next = False

    for i, stmt in enumerate(node.body.body):
      if skip_next:
        skip_next = False
        continue

      if self._is_super_init_stmt(stmt):
        # Remove escape hatch footer if present
        if i + 1 < len(node.body.body):
          next_stmt = node.body.body[i + 1]
          if self._is_escape_footer(next_stmt):
            skip_next = True
        continue

      new_body_stmts.append(stmt)

    return node.with_changes(body=node.body.with_changes(body=new_body_stmts))

  def _is_escape_footer(self, stmt: cst.CSTNode) -> bool:
    """Checks if statement is an Escape Hatch footer."""
    if isinstance(stmt, cst.SimpleStatementLine) and len(stmt.body) > 0:
      if isinstance(stmt.body[0], cst.Expr) and isinstance(stmt.body[0].value, cst.Ellipsis):
        for line in stmt.leading_lines:
          if EscapeHatch.END_MARKER in line.comment.value:
            return True
    return False

  def _has_super_init(self, node: cst.FunctionDef) -> bool:
    """Checks for presence of super().__init__()"""
    if hasattr(node.body, "body"):
      for stmt in node.body.body:
        if self._is_super_init_stmt(stmt):
          return True
    return False

  def _is_super_init_stmt(self, stmt: cst.CSTNode) -> bool:
    """Detects super init call pattern."""
    if isinstance(stmt, cst.SimpleStatementLine) and len(stmt.body) == 1:
      small = stmt.body[0]
      call_node = None

      if isinstance(small, cst.Expr):
        call_node = small.value
      elif isinstance(small, cst.Assign):
        call_node = small.value

      if isinstance(call_node, cst.Call):
        call = call_node
        if isinstance(call.func, cst.Attribute) and call.func.attr.value == "__init__":
          receiver = call.func.value
          if isinstance(receiver, cst.Call) and isinstance(receiver.func, cst.Name):
            if receiver.func.value == "super":
              return True
    return False

  def _apply_preamble(self, node: cst.FunctionDef, stmts_code: List[str]) -> cst.FunctionDef:
    """
    Injects code strings into function preamble.
    """
    new_stmts = []
    for code in stmts_code:
      try:
        mod = cst.parse_module(code)
        new_stmts.extend(mod.body)
      except Exception:
        pass

    if isinstance(node.body, SimpleStatementSuite):
      node = self._convert_to_indented_block(node)

    existing = list(node.body.body)
    idx = (
      1
      if (
        existing
        and isinstance(existing[0], cst.SimpleStatementLine)
        and isinstance(existing[0].body[0], cst.Expr)
        and isinstance(
          existing[0].body[0].value,
          (cst.SimpleString, cst.ConcatenatedString),
        )
      )
      else 0
    )

    final_body = existing[:idx] + new_stmts + existing[idx:]
    return node.with_changes(body=node.body.with_changes(body=final_body))
