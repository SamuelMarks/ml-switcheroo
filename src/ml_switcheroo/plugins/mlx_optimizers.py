"""
Plugin for MLX Optimizer translation.

Handles impedance mismatches between PyTorch and Apple MLX optimizers.
"""

import libcst as cst
from typing import Union

from ml_switcheroo.core.hooks import register_hook, HookContext
from ml_switcheroo.core.escape_hatch import EscapeHatch


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


@register_hook("mlx_optimizer_init")
def transform_mlx_optimizer_init(node: cst.Call, ctx: HookContext) -> cst.Call:
  if ctx.target_fw != "mlx":
    return node

  # 1. Rename API
  old_name = ""
  if isinstance(node.func, cst.Name):
    old_name = node.func.value
  elif isinstance(node.func, cst.Attribute):
    old_name = node.func.attr.value

  target_api = f"mlx.optimizers.{old_name}"
  new_func = _create_dotted_name(target_api)

  # 2. Filter Args & Rename Keywords
  new_args = []

  # Heuristic: Strip first arg if Positional (params)
  start_idx = 0
  if len(node.args) > 0 and node.args[0].keyword is None:
    start_idx = 1

  for i in range(start_idx, len(node.args)):
    arg = node.args[i]

    # Explicit Rename: lr -> learning_rate
    # This handles cases where the core argument rewriter might not have triggered
    # (e.g. in partial mock tests) or for robust enforcement.
    if arg.keyword and arg.keyword.value == "lr":
      arg = arg.with_changes(keyword=cst.Name("learning_rate"))

    new_args.append(arg)

  return node.with_changes(func=new_func, args=new_args)


@register_hook("mlx_optimizer_step")
def transform_mlx_optimizer_step(node: cst.Call, ctx: HookContext) -> Union[cst.Call, cst.FlattenSentinel]:
  if ctx.target_fw != "mlx":
    return node

  optimizer_var = None
  if isinstance(node.func, cst.Attribute):
    optimizer_var = node.func.value
  else:
    optimizer_var = cst.Name("optimizer")

  new_func = cst.Attribute(value=optimizer_var, attr=cst.Name("update"))
  args = [
    cst.Arg(value=cst.Name("model"), comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
    cst.Arg(value=cst.Name("grads")),
  ]
  new_call = cst.Call(func=new_func, args=args)

  reason = "MLX requires explicit `update(model, grads)`. Variables inferred placeholders."
  return EscapeHatch.mark_failure(new_call, reason)


@register_hook("mlx_zero_grad")
def transform_mlx_zero_grad(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
  if ctx.target_fw != "mlx":
    return node
  return node.with_changes(func=cst.Name("None"), args=[])
