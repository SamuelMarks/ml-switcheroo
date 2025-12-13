"""
Plugin for Lifecycle Flags (Training/Eval Modes).

This plugin bridges the gap between PyTorch's Object-Oriented state management
and the Functional patterns favored by JAX/Flax and Keras.

Transforms:
- `model.train()`       -> `training = True`
- `model.train(False)`  -> `training = False`
- `model.eval()`        -> `training = False`

Usage:
  - Automatic for loops targetting JAX or Keras 3.
  - Useful for conditioning `Dropout` or `BatchNorm` layers downstream.
"""

import libcst as cst
from typing import Optional, Union

from ml_switcheroo.core.hooks import register_hook, HookContext


def _is_lifecycle_method(node: cst.Call, method_name: str) -> bool:
  """Checks if a Call node matches `object.method_name()`."""
  if not isinstance(node.func, cst.Attribute):
    return False
  return node.func.attr.value == method_name


def _extract_boolean_arg(node: cst.Call, default: bool) -> bool:
  """
  Extracts the boolean state from a call like `train(True)`.
  Returns `default` if no arguments are provided.
  """
  if not node.args:
    return default

  # Check first arg
  first_arg = node.args[0].value
  if isinstance(first_arg, cst.Name):
    if first_arg.value == "True":
      return True
    if first_arg.value == "False":
      return False

  # If it's a variable or complex expression, we default to the default
  # because resolving variable values is out of scope for AST rewriting.
  return default


@register_hook("convert_lifecycle")
def convert_lifecycle_flags(node: cst.CSTNode, ctx: HookContext) -> Union[cst.CSTNode, cst.RemovalSentinel]:
  """
  Rewrites module state calls to functional flag assignments.

  Hooks into `TopLevel` or `Expr` nodes to replace statements.
  We target `Expr` (Expression Statement) because converting a `Call` expression
  inside another expression (e.g. `func(model.train())`) to an Assignment is invalid syntax.

  Target logic:
  1. Rewrite `*.train()` -> `training = True`
  2. Rewrite `*.eval()`  -> `training = False`
  """

  # Passthrough for Torch targets (they support these methods natively)
  if ctx.target_fw == "torch":
    return node

  # We only care about expression statements (standalone lines)
  if not isinstance(node, cst.Expr):
    return node

  # Analyze the inner expression
  expr = node.value
  if not isinstance(expr, cst.Call):
    return node

  # 1. Handle .train()
  if _is_lifecycle_method(expr, "train"):
    # train(True) [default] or train(False)
    is_training = _extract_boolean_arg(expr, default=True)
    return _create_assignment("training", is_training)

  # 2. Handle .eval()
  if _is_lifecycle_method(expr, "eval"):
    # eval() is always training=False
    return _create_assignment("training", False)

  return node


def _create_assignment(var_name: str, value: bool) -> cst.Assign:
  """Helper to generate `var_name = value` CST Node."""
  return cst.Assign(targets=[cst.AssignTarget(target=cst.Name(var_name))], value=cst.Name("True" if value else "False"))
