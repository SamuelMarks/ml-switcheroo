"""Plugin for Converting Explicit State Variable Wrappers to Parameter Constructors.

Handles impedance mismatches between explicit state container declarations
(``Param``, ``BatchStat``, ``Variable``) and parameter wrappers using
``HookContext.current_variant`` without hardcoded target framework paths.
"""

import typing

import libcst as cst

from ml_switcheroo.core.hooks import HookContext, register_hook
from ml_switcheroo.plugins.utils import (
  create_dotted_name,
  resolve_variant_api,
  resolve_variant_arg_name,
)

_NON_TRAINABLE_KINDS = frozenset({"BatchStat", "Variable", "Cache", "Intermediate"})


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  """Create a CST attribute chain from a dot-separated string name.

  Args:
      name_str: Dot-separated API path string.

  Returns:
      Constructed CST ``Name`` or nested ``Attribute`` expression.
  """
  return create_dotted_name(name_str)


def _extract_leaf_name(node: cst.BaseExpression) -> str | None:
  """Extract the right-most identifier name from a call target expression.

  Args:
      node: Call ``func`` expression node.

  Returns:
      Right-most identifier string, or ``None`` if unsupported.
  """
  if isinstance(node, cst.Name):
    return node.value
  if isinstance(node, cst.Attribute):
    return node.attr.value
  return None


def _normalize_param_args(args: typing.Sequence[cst.Arg]) -> list[cst.Arg]:
  """Normalize ``value=`` keyword wrappers into positional initializers where needed.

  Args:
      args: Original call arguments.

  Returns:
      Normalized list of ``cst.Arg`` nodes.
  """
  normalized: list[cst.Arg] = []
  for idx, arg in enumerate(args):
    if idx == 0 and arg.keyword is not None and arg.keyword.value == "value":
      normalized.append(arg.with_changes(keyword=None, equal=cst.MaybeSentinel.DEFAULT))
    else:
      normalized.append(arg)
  return normalized


@register_hook("nnx_param_to_torch")
def transform_nnx_param(node: cst.Call, ctx: HookContext) -> cst.Call:
  """Transform explicit Variable/Param declarations via ``HookContext.current_variant``.

  Args:
      node: Original CST Call node (e.g., ``nnx.Param(zeros(1))``).
      ctx: HookContext containing active operation ID and variant configuration.

  Returns:
      Transformed ``cst.Call`` node, or ``node`` unchanged if target API is unmapped.
  """
  leaf_name = _extract_leaf_name(node.func)
  op_kind = ctx.current_op_id if isinstance(ctx.current_op_id, str) and ctx.current_op_id else leaf_name
  is_trainable = op_kind not in _NON_TRAINABLE_KINDS and leaf_name not in _NON_TRAINABLE_KINDS

  target_api = resolve_variant_api(ctx, "Param")
  if not target_api:
    return node

  new_func = _create_dotted_name(target_api)
  new_args = _normalize_param_args(node.args)
  grad_kw = resolve_variant_arg_name(ctx, "requires_grad", "requires_grad")

  if not is_trainable and not any(a.keyword and a.keyword.value == grad_kw for a in new_args):
    if new_args and new_args[-1].comma == cst.MaybeSentinel.DEFAULT:
      new_args[-1] = new_args[-1].with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))
    new_args.append(
      cst.Arg(
        keyword=cst.Name(grad_kw),
        value=cst.Name("False"),
        equal=cst.AssignEqual(
          whitespace_before=cst.SimpleWhitespace(""),
          whitespace_after=cst.SimpleWhitespace(""),
        ),
      )
    )

  return node.with_changes(func=new_func, args=new_args)
