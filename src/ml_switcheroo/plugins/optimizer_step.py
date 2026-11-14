"""Plugin for Functional Optimizer Constructor & Step Translation.

Handles conversion of imperative parameter-bound optimizer constructors, steps,
and gradient resets to functional factory patterns driven by
``HookContext.current_variant``.
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
  """Create a CST Attribute chain from a dotted API path.

  Args:
      name_str: Dot-separated hierarchical API name.

  Returns:
      LibCST expression node representing the attribute chain.
  """
  return create_dotted_name(name_str)


def _is_scalar_hyperparam(expr: cst.BaseExpression) -> bool:
  """Check whether an expression is a numeric scalar literal (e.g., learning rate).

  Args:
      expr: Candidate CST expression.

  Returns:
      True if ``expr`` is an ``Integer``, ``Float``, or negated numeric literal.
  """
  if isinstance(expr, (cst.Float, cst.Integer)):
    return True
  return isinstance(expr, cst.UnaryOperation) and isinstance(expr.expression, (cst.Float, cst.Integer))


@register_hook("optimizer_constructor")
def transform_optimizer_init(node: cst.Call, ctx: HookContext) -> cst.Call:
  """Rewrite optimizer instantiation for functional stateless optimizer factories.

  Strips parameter-collection arguments (whether passed positionally or via
  ``params=``/``parameters=``/``var_list=`` keywords) while preserving scalar
  hyperparameters (e.g., ``Adam(0.001)``) and remapping keyword names declared
  in ``ctx.current_variant.args``.

  Args:
      node: Original CST call node.
      ctx: Active ``HookContext``.

  Returns:
      Transformed CST call node.
  """
  target_api = resolve_variant_api(ctx) if ctx.current_op_id else None
  new_func = _create_dotted_name(target_api) if target_api else node.func

  new_args: list[cst.Arg] = []
  for idx, arg in enumerate(node.args):
    if arg.keyword is not None:
      kw = arg.keyword.value
      if kw in _PARAM_KEYWORDS:
        continue
      mapped_kw = resolve_variant_arg_name(ctx, kw, kw)
      if mapped_kw != kw:
        arg = arg.with_changes(keyword=cst.Name(mapped_kw))
      new_args.append(arg)
      continue

    if idx == 0 and not arg.star and not _is_scalar_hyperparam(arg.value):
      continue
    new_args.append(arg)

  return node.with_changes(func=new_func, args=new_args)


@register_hook("optimizer_step")
def transform_optimizer_step(node: cst.Call, ctx: HookContext) -> cst.CSTNode | cst.FlattenSentinel[cst.BaseStatement]:
  """Flag imperative ``optimizer.step()`` calls with an explicit EscapeHatch diagnostic.

  Args:
      node: Original CST call node.
      ctx: Active ``HookContext``.

  Returns:
      CST node wrapped with EscapeHatch comments.
  """
  reason = (
    f"Imperative `{_get_func_name(node)}` cannot be automatically converted to functional update. "
    "Manual intervention required (e.g. `updates, state = opt.update(grads, state)`)."
  )
  return EscapeHatch.mark_failure(node, reason)


@register_hook("optimizer_zero_grad")
def strip_zero_grad(node: cst.Call, ctx: HookContext) -> cst.CSTNode | cst.FlattenSentinel[cst.BaseStatement]:
  """Replace imperative ``optimizer.zero_grad()`` calls with a no-op expression.

  Args:
      node: Original CST call node.
      ctx: Active ``HookContext``.

  Returns:
      Call rewritten to ``None`` no-op expression.
  """
  return node.with_changes(func=cst.Name("None"), args=[])


def _get_func_name(node: cst.Call) -> str:
  """Extract the leaf function or method name from a ``cst.Call`` node.

  Args:
      node: LibCST ``Call`` node to inspect.

  Returns:
      Extracted function/method name string, or ``"step"`` if unnamed.
  """
  if isinstance(node.func, cst.Attribute):
    return node.func.attr.value
  return "step"
