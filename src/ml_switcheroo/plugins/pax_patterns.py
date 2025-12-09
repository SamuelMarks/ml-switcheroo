"""
Plugin for handling PaxML / Praxis specific state patterns.

PaxML uses HParam-based configuration and a `setup()` method for layer
initialization, differing from PyTorch's `__init__` constructor pattern.

This plugin aims to:
1.  Map standard PyTorch constructor logic to `setup()`.
2.  Provide functional pass-through for layer calls, aligning with JAX/Flax semantics.
"""

import libcst as cst
from typing import Optional, List

from ml_switcheroo.core.hooks import register_hook, HookContext


@register_hook("pax_setup_migration")
def migrate_init_to_setup(node: cst.FunctionDef, ctx: HookContext) -> cst.FunctionDef:
  """
  Plugin Hook: Rename `__init__` to `setup` for Praxis Layer definitions.

  Triggers:
      Operations marked with `requires_plugin: "pax_setup_migration"`.
      Currently intended for `torch.nn.Module` -> `praxis.base_layer.BaseLayer` transformations.

  Action:
      - Renames `__init__` to `setup`.
      - Strips `super().__init__()` calls (Praxis BaseLayer handles this implicitly or differently).

  Args:
      node: The function definition node (expected to be `__init__`).
      ctx: The hook context.

  Returns:
      The transformed FunctionDef node.
  """
  # Safety: Only apply if targeting PaxML
  if ctx.target_fw != "paxml":
    return node

  # Ensure we are modifying __init__
  if node.name.value != "__init__":
    return node

  # 1. Rename Method -> 'setup'
  # This aligns with Praxis lifecycle where layers define components in setup()
  new_node = node.with_changes(name=cst.Name("setup"))

  # 2. Strip super().__init__() calls
  # Praxis layers generally do not require explicit super init in setup()
  new_body = _strip_super_init(new_node.body)
  new_node = new_node.with_changes(body=new_body)

  return new_node


def _strip_super_init(body: cst.IndentedBlock) -> cst.IndentedBlock:
  """
  Removes `super().__init__()` statements from the function body.
  """
  new_stmts = []
  for stmt in body.body:
    if _is_super_init_call(stmt):
      continue
    new_stmts.append(stmt)

  # If body becomes empty, insert 'pass' to maintain validity
  if not new_stmts:
    new_stmts = [cst.SimpleStatementLine(body=[cst.Expr(value=cst.Pass())])]

  return body.with_changes(body=new_stmts)


def _is_super_init_call(stmt: cst.BaseStatement) -> bool:
  """
  Detects `super().__init__(...)` calls.
  """
  if not isinstance(stmt, cst.SimpleStatementLine):
    return False

  for small_stmt in stmt.body:
    if isinstance(small_stmt, cst.Expr) and isinstance(small_stmt.value, cst.Call):
      call = small_stmt.value
      # Check: super().__init__
      if isinstance(call.func, cst.Attribute) and call.func.attr.value == "__init__":
        # Check receiver: super()
        receiver = call.func.value
        if isinstance(receiver, cst.Call) and isinstance(receiver.func, cst.Name):
          if receiver.func.value == "super":
            return True
  return False
