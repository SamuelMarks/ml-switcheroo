"""
Plugin for MLX Optimizer translation.

Handles impedance mismatches for Functional Optimizers.
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
  """
  Hook: Transforms Optimizer Constructor.

  1. Renames API based on context lookup or dynamic class construction.
  2. Strips parameter argument (Arg 0).
  3. Renames `lr` -> `learning_rate`.
  """
  # 1. Rename API
  # Try strict lookup first
  op_id = ctx.current_op_id or "Adam"
  target_api = ctx.lookup_api(op_id)

  if not target_api:
    # Fallback: Construct 'mlx.optimizers.<Name>' if source was 'optim.<Name>'
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
    if arg.keyword and arg.keyword.value == "lr":
      arg = arg.with_changes(keyword=cst.Name("learning_rate"))

    new_args.append(arg)

  return node.with_changes(func=new_func, args=new_args)


@register_hook("mlx_optimizer_step")
def transform_mlx_optimizer_step(node: cst.Call, ctx: HookContext) -> Union[cst.Call, cst.FlattenSentinel]:
  """
  Hook: Transforms `optimizer.step()` into an EscapeHatch pattern.
  Functional optimizers (like MLX/Optax) require explicit update calls `opt.update(model, state)`.
  """
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

  reason = "Functional optimizers require explicit `update(model, grads)`. Variables inferred placeholders."
  return EscapeHatch.mark_failure(new_call, reason)


@register_hook("mlx_zero_grad")
def transform_mlx_zero_grad(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
  """
  Hook: Transforms `optimizer.zero_grad()` into `None` (No-Op).
  """
  return cst.Name("None")
