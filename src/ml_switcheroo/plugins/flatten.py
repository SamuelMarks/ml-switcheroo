"""Plugin for Dimension-Range Flattening.

PyTorch's `flatten(start_dim, end_dim)` collapses a range of dimensions.
Mapping strategies:
1. JAX: `jax.lax.collapse(x, start, stop)` - Most robust for dynamic shapes.
2. NumPy/Default: `x.reshape(...)` or `x.ravel()`.
"""

import typing

import libcst as cst

from ml_switcheroo.core.hooks import HookContext, register_hook
from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.plugins.utils import create_dotted_name


@typing.runtime_checkable
class _VariantArgsProtocol(typing.Protocol):
  """Protocol for variant objects exposing `args`.

  Attributes:
      args: The args (`dict[str, str] | None`).
  """

  args: dict[str, str] | None


@typing.runtime_checkable
class _VariantOpTypeProtocol(typing.Protocol):
  """Protocol for variant objects exposing `op_type`.

  Attributes:
      op_type: The op type (`str | None`).
  """

  op_type: str | None


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  """Support to create a CST Attribute chain from string.

  Args:
      name_str: A dot-separated string representing the target API name,
          such as "jax.lax.collapse".

  Returns:
      A LibCST expression node representing the dotted attribute chain.
  """
  return create_dotted_name(name_str)


def _create_integer(val: int) -> cst.BaseExpression:
  """Create a CST node for an integer, handling negative values.

  Args:
      val: Integer value to convert to a LibCST expression node.

  Returns:
      LibCST Integer or UnaryOperation expression node.
  """
  if val < 0:
    return cst.UnaryOperation(operator=cst.Minus(), expression=cst.Integer(str(-val)))
  return cst.Integer(str(val))


def _parse_signed_int_cst(expr: cst.BaseExpression) -> int | None:
  """Extract a signed integer literal from a CST ``Integer`` or unary ``-Integer`` node.

  Args:
      expr: Candidate CST expression node.

  Returns:
      Parsed integer value if ``expr`` is a base-10 integer literal, otherwise ``None``.
  """
  try:
    if isinstance(expr, cst.Integer):
      return int(expr.value)
    if (
      isinstance(expr, cst.UnaryOperation)
      and isinstance(expr.operator, cst.Minus)
      and isinstance(expr.expression, cst.Integer)
    ):
      return -int(expr.expression.value)
  except ValueError:
    pass
  return None


def _extract_flatten_dims(args: list[cst.Arg]) -> tuple[int, int]:
  """Extract ``(start_dim, end_dim)`` from positional and keyword flatten arguments.

  Args:
      args: Call argument list starting with the input tensor expression.

  Returns:
      Tuple of ``(start_dim, end_dim)`` defaulting to ``(0, -1)``.
  """
  start_dim = 0
  end_dim = -1

  if len(args) > 1:
    pos_start = _parse_signed_int_cst(args[1].value)
    if pos_start is not None:
      start_dim = pos_start
  if len(args) > 2:
    pos_end = _parse_signed_int_cst(args[2].value)
    if pos_end is not None:
      end_dim = pos_end

  for arg in args:
    if arg.keyword:
      parsed = _parse_signed_int_cst(arg.value)
      if parsed is not None:
        if arg.keyword.value == "start_dim":
          start_dim = parsed
        elif arg.keyword.value == "end_dim":
          end_dim = parsed

  return start_dim, end_dim


def _synthesize_collapse_call(
  node: cst.Call,
  target_api: str,
  input_arg: cst.Arg,
  start_dim: int,
  end_dim: int,
) -> cst.Call:
  """Synthesize an exclusive ``stop_dimension`` collapse call (e.g. ``jax.lax.collapse``).

  Args:
      node: Original CST ``Call`` node.
      target_api: Dotted target API path string.
      input_arg: First input tensor ``cst.Arg`` node.
      start_dim: Inclusive start dimension index.
      end_dim: Inclusive end dimension index (``-1`` denotes ``x.ndim``).

  Returns:
      Rewritten collapse ``cst.Call`` node.
  """
  new_func = _create_dotted_name(target_api)
  arg0 = input_arg.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))
  arg1 = cst.Arg(value=_create_integer(start_dim), comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))
  arg2_val: cst.BaseExpression = (
    cst.Attribute(value=input_arg.value, attr=cst.Name("ndim")) if end_dim == -1 else _create_integer(end_dim + 1)
  )
  return node.with_changes(func=new_func, args=[arg0, arg1, cst.Arg(value=arg2_val)])


def _synthesize_batch_reshape_call(node: cst.Call, target_api: str, input_arg: cst.Arg) -> cst.Call:
  """Synthesize ``reshape(x, (x.shape[0], -1))`` for batch-preserving ``flatten(x, 1, -1)``.

  Args:
      node: Original CST ``Call`` node.
      target_api: Dotted target reshape API path string.
      input_arg: First input tensor ``cst.Arg`` node.

  Returns:
      Rewritten reshape ``cst.Call`` node.
  """
  new_func = _create_dotted_name(target_api)
  shape_attr = cst.Attribute(value=input_arg.value, attr=cst.Name("shape"))
  batch_dim = cst.Subscript(value=shape_attr, slice=[cst.SubscriptElement(slice=cst.Index(value=cst.Integer("0")))])
  neg_one = cst.UnaryOperation(operator=cst.Minus(), expression=cst.Integer("1"))
  shape_tuple = cst.Tuple(
    elements=[
      cst.Element(value=batch_dim, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
      cst.Element(value=neg_one),
    ]
  )
  fmt_input = (
    input_arg.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))
    if input_arg.comma == cst.MaybeSentinel.DEFAULT
    else input_arg
  )
  return node.with_changes(func=new_func, args=[fmt_input, cst.Arg(value=shape_tuple)])


@register_hook("flatten_range")
def transform_flatten(node: cst.Call, ctx: HookContext) -> cst.Call:
  """Transform Transforms `flatten(x, start, end)` into target-specific logic.

  This function identifies flattening operations, parses their arguments (including
  start and end dimension specifications), looks up the configured target API,
  and transforms the LibCST call into target-specific patterns (e.g., JAX collapse,
  Ravel, or Reshape with shape calculations).

  Args:
      node: The original LibCST Call node representing the flatten operation.
      ctx: The translation HookContext holding API configurations, state,
          and lookup methods.

  Returns:
      The transformed LibCST Call node or the original node if no target
      API was found or transformation is not applicable.
  """
  args = list(node.args)
  if not args:
    return node

  input_arg = args[0]
  start_dim, end_dim = _extract_flatten_dims(args)

  target_api = ctx.lookup_api(ctx.current_op_id) if ctx.current_op_id else None
  if not target_api:
    target_api = ctx.lookup_api("flatten") or ctx.lookup_api("Flatten")
  if not target_api:
    return node

  from ml_switcheroo.core.dsl import FrameworkVariant

  target_variant: (
    FrameworkVariant | dict[str, RuntimeTensorOrValue] | _VariantArgsProtocol | _VariantOpTypeProtocol | None
  ) = ctx.current_variant
  raw_v_args = (
    target_variant.get("args")
    if isinstance(target_variant, dict)
    else (target_variant.args if isinstance(target_variant, _VariantArgsProtocol) else None)
  )
  variant_args: dict[str, str] = raw_v_args if isinstance(raw_v_args, dict) else {}
  api_leaf = target_api.rsplit(".", 1)[-1]

  if "stop_dimension" in variant_args or api_leaf == "collapse":
    return _synthesize_collapse_call(node, target_api, input_arg, start_dim, end_dim)

  if "start_axis" in variant_args or target_api.endswith(".core.flatten"):
    new_func = _create_dotted_name(target_api)
    arg0 = input_arg.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))
    arg1 = cst.Arg(value=_create_integer(start_dim), comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))
    return node.with_changes(func=new_func, args=[arg0, arg1, cst.Arg(value=_create_integer(end_dim))])

  is_class = (
    isinstance(target_variant, _VariantOpTypeProtocol)
    and target_variant.op_type is not None
    and "class" in str(target_variant.op_type).lower()
  )
  if is_class or target_api.endswith(".Flatten"):
    instantiation = cst.Call(func=_create_dotted_name(target_api), args=[])
    return cst.Call(func=instantiation, args=[input_arg.with_changes(comma=cst.MaybeSentinel.DEFAULT)])

  if start_dim == 0 and end_dim == -1 and ("ravel" in target_api.lower() or "flatten" in target_api.lower()):
    return node.with_changes(func=_create_dotted_name(target_api), args=[input_arg])

  if start_dim == 1 and end_dim == -1:
    return _synthesize_batch_reshape_call(node, target_api, input_arg)

  return node
