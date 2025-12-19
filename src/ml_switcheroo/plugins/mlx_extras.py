"""
Plugin for MLX Ecosystem Mapping.

Handles:
1. Compilation: `@torch.compile` -> `@mx.compile`.
2. Eager Evaluation: `torch.cuda.synchronize()` -> `mx.eval(state)`.
3. Streams: `torch.cuda.stream` -> `mx.stream(mx.gpu)`.
"""

import libcst as cst
from typing import Union

from ml_switcheroo.core.hooks import register_hook, HookContext


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  """Helper: Creates a CST Attribute chain from string."""
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


@register_hook("mlx_compiler")
def transform_compiler(node: Union[cst.Decorator, cst.Call], ctx: HookContext) -> cst.CSTNode:
  """
  Hook: Maps JIT compilation decorators.

  Triggers: `torch.compile` (via `requires_plugin: "mlx_compiler"`).

  Transformation:
      Input:  `@torch.compile(fullgraph=True, dynamic=True)`
      Output: `@mx.compile` (stripping incompatible kwargs).

  Note: MLX's compiler (`mx.compile`) is largely drop-in but does not support
  PyTorch specific flags like `fullgraph` or `backend`.
  """
  if ctx.target_fw != "mlx":
    return node

  # Handling Decorator Node
  # Structure: Decorator(decorator=Call(...)) or Decorator(decorator=Name(...))

  decorator_expr = node.decorator

  # New API
  target_api = "mlx.core.compile"  # Aliased usually as mx.compile
  new_func = _create_dotted_name(target_api)

  # If explicit call `@torch.compile(...)`
  if isinstance(decorator_expr, cst.Call):
    # MLX compile takes function as first arg (if used as wrapper) or acts as decorator.
    # But `@mx.compile(f)` is valid? No, usually `@mx.compile` without parens
    # or `@partial(mx.compile, static_argnums=...)` pattern via functools.
    # Standard usage: `@mx.compile`

    # We strip arguments completely because Torch kwargs (backend="inductor")
    # are invalid in MLX.
    # Returning `mx.compile` without parens as the decorator expression.
    return node.with_changes(decorator=new_func)

  # If implicit call `@torch.compile`
  return node.with_changes(decorator=new_func)


@register_hook("mlx_synchronize")
def transform_synchronize(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
  """
  Hook: Maps barrier synchronization.

  Input:  `torch.cuda.synchronize()`
  Output: `mx.eval(state_vars)` or `mx.async_eval()`?

  MLX is lazy. Correct sync is `mx.eval(tensors)`.
  Since `synchronize()` in Torch catches up everything globally,
  we map it to a comment or specific stream sync if possible.

  Strict mapping: `mx.eval()` requires arguments (what to eval).
  If we can't find arguments, we emit a comment.
  """
  if ctx.target_fw != "mlx":
    return node

  # Construct comment node logic?
  # BaseRewriter supports swapping nodes. Escaping to comment requires wrapping.
  # Plugin can return `cst.Expr(value=cst.Ellipsis())` with attached comment.
  # Or simply return a pass `cst.Pass()`.

  message = "# [MLX] Global sync requires explicit tensor args: mx.eval(tensors)"

  # We return a `None` call which rewriter handles, or pass.
  # Let's start with a Pass node placeholder.
  # BUT hooks replace Expressions usually. Pass is a Statement.
  # Valid replacement for `f()` expression is `None`.

  return cst.Call(func=cst.Name("print"), args=[cst.Arg(value=cst.SimpleString(f"'{message}'"))])
