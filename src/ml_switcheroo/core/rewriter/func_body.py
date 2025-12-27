"""
Function Body Rewriting Logic.

This module provides the `FuncBodyMixin` used by the `structure_func` rewriter.
It handles structural manipulations of the function body, including:
1.  Preamble injection (injecting statements at the start).
2.  Super Init injection (`super().__init__()`) for OO frameworks.
3.  Super Init stripping for Functional/Component frameworks (Pax/Praxis).
4.  Converting one-line bodies to indented blocks.
"""

from typing import List
import libcst as cst
from libcst import BaseSuite, SimpleStatementSuite, IndentedBlock, SimpleStatementLine

from ml_switcheroo.core.escape_hatch import EscapeHatch


class FuncBodyMixin:
  """
  Mixin for manipulating function bodies.

  Provides utilities to inject statements into the beginning of a function body,
  handle the conversion of simple suites to indented blocks, and manage
  `super().__init__()` calls.
  """

  def _is_body_accessible(self, body: BaseSuite) -> bool:
    """Checks if the function body is a standard indented block (not a one-liner)."""
    return isinstance(body, IndentedBlock)

  def _convert_to_indented_block(self, node: cst.FunctionDef) -> cst.FunctionDef:
    """
    Converts simple one-line function bodies to indented blocks.
    Necessary when injecting statements (preamble/super) into ``def f(): pass``.

    Args:
        node: Function Definition node.

    Returns:
        FunctionDef node with an ``IndentedBlock`` body.
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
    Injects ``super().__init__()`` at the start of the function body.
    Idempotent: Checks if the call already exists before injecting.

    Args:
        node: The function definition.

    Returns:
        The modified function with super init call.
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
    # Skip Docstring if present
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
    Removes ``super().__init__()`` calls from the function body.

    Crucial for converting object-oriented PyTorch models (where super init is required)
    to frameworks like PaxML/Praxis (where setup methods are implicit).

    Args:
        node: The function definition.

    Returns:
        The modified function without super init calls.
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
        # Check if the NEXT statement is an Escape Hatch Footer that should also be removed
        # (happens if the super call was wrapped due to warning/error)
        if i + 1 < len(node.body.body):
          next_stmt = node.body.body[i + 1]
          if self._is_escape_footer(next_stmt):
            skip_next = True
        continue

      new_body_stmts.append(stmt)

    return node.with_changes(body=node.body.with_changes(body=new_body_stmts))

  def _is_escape_footer(self, stmt: cst.CSTNode) -> bool:
    """Checks if a statement is the closure of an Escape Hatch (Ellipsis marker)."""
    # Ends usually look like: ... (Ellipsis) with a comment leading lines
    if isinstance(stmt, cst.SimpleStatementLine) and len(stmt.body) > 0:
      if isinstance(stmt.body[0], cst.Expr) and isinstance(stmt.body[0].value, cst.Ellipsis):
        # Check comments for END_MARKER
        for line in stmt.leading_lines:
          if EscapeHatch.END_MARKER in line.comment.value:
            return True
    return False

  def _has_super_init(self, node: cst.FunctionDef) -> bool:
    """Checks for presence of ``super().__init__()`` in function body."""
    if hasattr(node.body, "body"):
      for stmt in node.body.body:
        if self._is_super_init_stmt(stmt):
          return True
    return False

  def _is_super_init_stmt(self, stmt: cst.CSTNode) -> bool:
    """
    Detects if statement is ``super().__init__()`` or ``super(Type, self).__init__()``.
    """
    if isinstance(stmt, cst.SimpleStatementLine) and len(stmt.body) == 1:
      small = stmt.body[0]
      if isinstance(small, cst.Expr) and isinstance(small.value, cst.Call):
        call = small.value
        # Check for .__init__()
        if isinstance(call.func, cst.Attribute) and call.func.attr.value == "__init__":
          receiver = call.func.value
          # Check for super(...)
          if isinstance(receiver, cst.Call) and isinstance(receiver.func, cst.Name):
            if receiver.func.value == "super":
              return True
    return False

  def _apply_preamble(self, node: cst.FunctionDef, stmts_code: List[str]) -> cst.FunctionDef:
    """
    Injects a list of statement strings at the beginning of the function body.

    Skips over docstrings so they remain the first element.

    Args:
        node: The function definition.
        stmts_code: List of python source strings to parse and inject.

    Returns:
        The modified function definition.
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
