"""
Plugin hooks for PaxML patterns.

Handles:
1. Renaming `__init__` to `setup`.
2. Stripping `super().__init__()` calls (which are invalid in Praxis setup).
"""

import libcst as cst
from typing import Union

from ml_switcheroo.core.hooks import HookContext


def migrate_init_to_setup(node: cst.FunctionDef, ctx: HookContext) -> cst.FunctionDef:
  """
  Hook to transform __init__ methods for PaxML targets.

  Args:
      node: The FunctionDef node.
      ctx: Hook execution context.

  Returns:
      Transformed FunctionDef (renamed to setup, super init removed).
  """
  # Guard: Only apply for PaxML
  if ctx.target_fw != "paxml":
    return node

  # Guard: Only apply to __init__
  if node.name.value != "__init__":
    return node

  # 1. Rename to 'setup'
  new_node = node.with_changes(name=cst.Name("setup"))

  # 2. Strip super().__init__()
  # We filter the body statements
  if isinstance(new_node.body, cst.IndentedBlock):
    new_body_stmts = []
    for stmt in new_node.body.body:
      # Check if stmt is expr -> call -> super.__init__
      if _is_super_init(stmt):
        continue
      new_body_stmts.append(stmt)

    new_block = new_node.body.with_changes(body=new_body_stmts)
    new_node = new_node.with_changes(body=new_block)

  return new_node


def _is_super_init(stmt: cst.BaseStatement) -> bool:
  """Helper to detect `super().__init__()` statement."""
  if isinstance(stmt, cst.SimpleStatementLine):
    if len(stmt.body) == 1 and isinstance(stmt.body[0], cst.Expr):
      expr = stmt.body[0]
      if isinstance(expr.value, cst.Call):
        call = expr.value
        # Check func is attribute (super().init)
        if isinstance(call.func, cst.Attribute) and call.func.attr.value == "__init__":
          # Check receiver is super()
          receiver = call.func.value
          if isinstance(receiver, cst.Call) and isinstance(receiver.func, cst.Name):
            if receiver.func.value == "super":
              return True
  return False
