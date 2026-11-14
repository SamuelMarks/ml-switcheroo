"""Plugin for Functional Optimizer Translation.

Handles impedance mismatches between stateful parameter-bound optimizers and
functional stateless optimizer constructors, updates, and gradient resets
without hardcoded target framework strings.
"""

import libcst as cst

from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.core.hooks import HookContext, register_hook
from ml_switcheroo.plugins.utils import (
  create_dotted_name,
  resolve_variant_api,
  resolve_variant_arg_name,
)

_PARAM_KEYWORDS = frozenset({"params", "parameters", "var_list", "trainable_variables"})


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  """Create a CST attribute chain from a dotted API path string.

  Args:
      name_str: Dotted API path string.

  Returns:
      Resulting CST ``Attribute`` or ``Name`` node.
  """
  return create_dotted_name(name_str)


def _is_numeric_literal(expr: cst.BaseExpression) -> bool:
  """Check whether an expression is a scalar numeric literal (e.g., learning rate).

  Args:
      expr: Candidate CST expression.

  Returns:
      True if ``expr`` is a float, int, or negated numeric literal.
  """
  if isinstance(expr, (cst.Float, cst.Integer)):
    return True
  return isinstance(expr, cst.UnaryOperation) and isinstance(expr.expression, (cst.Float, cst.Integer))


@register_hook("mlx_optimizer_init")
def transform_mlx_optimizer_init(node: cst.Call, ctx: HookContext) -> cst.Call:
  """Transform an optimizer constructor call using ``HookContext.current_variant``.

  Resolves the target API strictly from ``ctx.current_variant`` or semantic lookup
  (aborting safely if unmapped), strips parameter collection arguments (whether
  passed positionally or by keyword), and remaps ``lr`` to the target learning-rate
  parameter declared in ``current_variant.args`` (defaulting to ``learning_rate``).

  Args:
      node: Original CST call node.
      ctx: Hook execution context.

  Returns:
      Transformed optimizer initialization call, or ``node`` if unmapped.
  """
  target_api = resolve_variant_api(ctx, "Adam")
  if not target_api:
    return node

  new_func = _create_dotted_name(target_api)
  lr_kw = resolve_variant_arg_name(
    ctx,
    "learning_rate",
    resolve_variant_arg_name(ctx, "lr", "learning_rate"),
  )

  new_args: list[cst.Arg] = []
  for idx, arg in enumerate(node.args):
    if arg.star:
      continue
    if arg.keyword is not None:
      kw_name = arg.keyword.value
      if kw_name in _PARAM_KEYWORDS:
        continue
      if kw_name == "lr":
        arg = arg.with_changes(keyword=cst.Name(lr_kw))
      else:
        mapped_kw = resolve_variant_arg_name(ctx, kw_name, kw_name)
        if mapped_kw != kw_name:
          arg = arg.with_changes(keyword=cst.Name(mapped_kw))
      new_args.append(arg)
      continue

    if idx == 0 and not _is_numeric_literal(arg.value):
      continue
    new_args.append(arg)

  return node.with_changes(func=new_func, args=new_args)


@register_hook("mlx_optimizer_step")
def transform_mlx_optimizer_step(
  node: cst.Call, ctx: HookContext
) -> cst.CSTNode | cst.FlattenSentinel[cst.BaseStatement]:
  """Transform imperative optimizer step calls into an EscapeHatch functional update.

  Args:
      node: Original CST call node.
      ctx: Hook execution context.

  Returns:
      Call node wrapped in an EscapeHatch failure marker.
  """
  optimizer_var: cst.BaseExpression = node.func.value if isinstance(node.func, cst.Attribute) else cst.Name("optimizer")
  new_func = cst.Attribute(value=optimizer_var, attr=cst.Name("update"))
  args = [
    cst.Arg(value=cst.Name("model"), comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
    cst.Arg(value=cst.Name("grads")),
  ]
  new_call = cst.Call(func=new_func, args=args)

  reason = "Functional optimizers require explicit `update(model, grads)`. Variables inferred placeholders."
  return EscapeHatch.mark_failure(new_call, reason)


@register_hook("mlx_zero_grad")
def transform_mlx_zero_grad(node: cst.Call, ctx: HookContext) -> cst.CSTNode | cst.FlattenSentinel[cst.BaseStatement]:
  """Transform ``optimizer.zero_grad()`` into a ``None`` no-op expression.

  Args:
      node: Original CST call node.
      ctx: Hook execution context.

  Returns:
      ``cst.Name("None")`` no-op node.
  """
  return cst.Name("None")
