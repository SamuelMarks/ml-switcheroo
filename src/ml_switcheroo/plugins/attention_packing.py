"""Plugin for MultiHead Attention Constructor & Call Repacking.

Provides modular argument-mapping hooks for transforming MultiHeadAttention
constructors and forward calls across positional, keyword, Keras ``(q, v, key=k)``,
and 2-argument self-attention conventions using ``HookContext.current_variant``.
"""

import typing

import libcst as cst

from ml_switcheroo.core.hooks import HookContext, register_hook
from ml_switcheroo.plugins.utils import create_dotted_name, resolve_variant_api

_QUERY_KEYWORDS = frozenset({"query", "q", "inputs_q"})
_KEY_KEYWORDS = frozenset({"key", "k"})
_VALUE_KEYWORDS = frozenset({"value", "v", "inputs_kv"})
_CONSTRUCTOR_KEYWORDS = frozenset({"embed_dim", "num_heads", "key_dim", "in_features", "dropout_rate"})


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  """Create a CST attribute chain from a dotted string.

  Args:
      name_str: Dot-separated API path string.

  Returns:
      CST ``Name`` or nested ``Attribute`` expression node.
  """
  return create_dotted_name(name_str)


def _resolve_target_class(ctx: HookContext) -> cst.BaseExpression | None:
  """Resolve the ``MultiheadAttention`` target class from ``HookContext.current_variant``.

  Args:
      ctx: HookContext providing ``current_variant`` and API lookup.

  Returns:
      CST node for the target class, or ``None`` if mapping is missing.
  """
  api = resolve_variant_api(ctx, "MultiheadAttention")
  if not api:
    return None
  return _create_dotted_name(api)


def _is_constructor_signature(args: typing.Sequence[cst.Arg]) -> bool:
  """Detect constructor initialization vs forward inference call arguments.

  Args:
      args: Sequence of ``cst.Arg`` nodes on the call.

  Returns:
      True if the arguments match a constructor signature, False otherwise.
  """
  if any(arg.keyword and arg.keyword.value in _CONSTRUCTOR_KEYWORDS for arg in args):
    return True
  return len(args) == 2 and not any(a.keyword for a in args) and isinstance(args[0].value, cst.Integer)


def _strip_kw(arg: cst.Arg, with_comma: bool = False) -> cst.Arg:
  """Convert an argument to a clean positional ``cst.Arg``.

  Args:
      arg: Source ``cst.Arg`` node.
      with_comma: Whether to attach a trailing comma with space.

  Returns:
      Positional ``cst.Arg`` node.
  """
  comma_val = cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")) if with_comma else cst.MaybeSentinel.DEFAULT
  return arg.with_changes(keyword=None, equal=cst.MaybeSentinel.DEFAULT, comma=comma_val)


def _extract_qkv_and_rest(
  args: typing.Sequence[cst.Arg],
) -> tuple[cst.Arg, cst.Arg, cst.Arg, list[cst.Arg]] | None:
  """Unpack ``(query, key, value, remaining_args)`` across framework conventions.

  Supports:
  - Standard 3-positional ``(q, k, v, ...)``
  - Keras positional+keyword ``(q, v, key=k, ...)``
  - Keyword Q/K/V ``(query=q, key=k, value=v, ...)``
  - Keyword 2-arg self-attention ``(query=q, value=v, ...)`` (where ``key`` defaults to ``value``)

  Args:
      args: Call arguments from the source forward invocation.

  Returns:
      Tuple ``(q_arg, k_arg, v_arg, remaining_args)`` or ``None`` if insufficient arguments.
  """
  q_kw: cst.Arg | None = None
  k_kw: cst.Arg | None = None
  v_kw: cst.Arg | None = None
  positional: list[cst.Arg] = []
  remaining: list[cst.Arg] = []

  for arg in args:
    if arg.keyword is not None:
      kw = arg.keyword.value
      if kw in _QUERY_KEYWORDS and q_kw is None:
        q_kw = arg
      elif kw in _KEY_KEYWORDS and k_kw is None:
        k_kw = arg
      elif kw in _VALUE_KEYWORDS and v_kw is None:
        v_kw = arg
      else:
        remaining.append(arg)
    else:
      positional.append(arg)

  if q_kw is None and positional:
    q_kw = positional.pop(0)

  if k_kw is not None and v_kw is None and positional:
    v_kw = positional.pop(0)
  else:
    if k_kw is None and positional:
      k_kw = positional.pop(0)
    if v_kw is None and positional:
      v_kw = positional.pop(0)

  if q_kw is not None and v_kw is not None and k_kw is None and v_kw.keyword is not None:
    k_kw = v_kw

  if q_kw is None or k_kw is None or v_kw is None:
    return None

  return q_kw, k_kw, v_kw, positional + remaining


def _remap_mask_kwargs(
  remaining_args: typing.Sequence[cst.Arg],
  source_names: frozenset[str],
  target_name: str,
) -> list[cst.Arg]:
  """Rename mask keyword arguments in ``remaining_args`` to ``target_name``.

  Args:
      remaining_args: Extra trailing arguments on the attention call.
      source_names: Set of mask keyword names to rename.
      target_name: Target mask keyword name.

  Returns:
      Updated list of ``cst.Arg`` nodes.
  """
  mapped: list[cst.Arg] = []
  for arg in remaining_args:
    if arg.keyword is not None and arg.keyword.value in source_names:
      mapped.append(arg.with_changes(keyword=cst.Name(target_name)))
    else:
      mapped.append(arg)
  return mapped


@register_hook("repack_attn_keras")
def repack_attn_keras(node: cst.Call, ctx: HookContext) -> cst.Call:
  """Repack MultiHeadAttention constructor and forward calls into Keras convention.

  Args:
      node: Original ``cst.Call`` node.
      ctx: HookContext for API lookup.

  Returns:
      Transformed ``cst.Call`` node, or ``node`` if unmapped/incomplete.
  """
  args = list(node.args)
  if _is_constructor_signature(args):
    new_func = _resolve_target_class(ctx)
    if not new_func:
      return node
    new_args = [
      arg.with_changes(keyword=cst.Name("key_dim")) if arg.keyword and arg.keyword.value == "embed_dim" else arg
      for arg in args
    ]
    return node.with_changes(func=new_func, args=new_args)

  unpacked = _extract_qkv_and_rest(args)
  if unpacked is None:
    return node

  q_arg, k_arg, v_arg, remaining_args = unpacked
  k_kw = cst.Arg(
    keyword=cst.Name("key"),
    value=k_arg.value,
    equal=cst.AssignEqual(
      whitespace_before=cst.SimpleWhitespace(""),
      whitespace_after=cst.SimpleWhitespace(""),
    ),
    comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
  )
  new_args = [
    _strip_kw(q_arg, with_comma=True),
    _strip_kw(v_arg, with_comma=True),
    k_kw,
    *_remap_mask_kwargs(remaining_args, frozenset({"attn_mask", "key_padding_mask", "mask"}), "attention_mask"),
  ]
  return node.with_changes(args=new_args)


@register_hook("repack_attn_flax")
def repack_attn_flax(node: cst.Call, ctx: HookContext) -> cst.Call:
  """Repack MultiHeadAttention constructor and forward calls into Flax NNX convention.

  Args:
      node: Original ``cst.Call`` node.
      ctx: HookContext for API lookup.

  Returns:
      Transformed ``cst.Call`` node, or ``node`` if unmapped/incomplete.
  """
  args = list(node.args)
  if _is_constructor_signature(args):
    new_func = _resolve_target_class(ctx)
    if not new_func:
      return node
    return node.with_changes(func=new_func)

  unpacked = _extract_qkv_and_rest(args)
  if unpacked is None:
    return node

  q_arg, k_arg, v_arg, remaining_args = unpacked
  mapped_rest = _remap_mask_kwargs(
    remaining_args,
    frozenset({"attn_mask", "key_padding_mask", "attention_mask"}),
    "mask",
  )
  new_args = [
    _strip_kw(q_arg, with_comma=True),
    _strip_kw(k_arg, with_comma=True),
    _strip_kw(v_arg, with_comma=bool(mapped_rest)),
    *mapped_rest,
  ]
  return node.with_changes(args=new_args)


@register_hook("repack_attn_torch")
def repack_attn_torch(node: cst.Call, ctx: HookContext) -> cst.Call:
  """Repack MultiHeadAttention constructor and forward calls into PyTorch convention.

  Args:
      node: Original ``cst.Call`` node.
      ctx: HookContext for API lookup.

  Returns:
      Transformed ``cst.Call`` node, or ``node`` if unmapped/incomplete.
  """
  args = list(node.args)
  if _is_constructor_signature(args):
    new_func = _resolve_target_class(ctx)
    if not new_func:
      return node
    new_args: list[cst.Arg] = []
    for arg in args:
      if arg.keyword:
        kw = arg.keyword.value
        if kw in {"key_dim", "in_features"}:
          new_args.append(arg.with_changes(keyword=cst.Name("embed_dim")))
        elif kw == "dropout_rate":
          new_args.append(arg.with_changes(keyword=cst.Name("dropout")))
        else:
          new_args.append(arg)
      else:
        new_args.append(arg)
    if not any(a.keyword and a.keyword.value == "batch_first" for a in new_args):
      new_args.append(cst.Arg(keyword=cst.Name("batch_first"), value=cst.Name("True")))
    return node.with_changes(func=new_func, args=new_args)

  unpacked = _extract_qkv_and_rest(args)
  if unpacked is None:
    return node

  q_arg, k_arg, v_arg, remaining_args = unpacked
  mapped_rest = _remap_mask_kwargs(
    remaining_args,
    frozenset({"attention_mask", "mask"}),
    "attn_mask",
  )
  new_args = [
    _strip_kw(q_arg, with_comma=True),
    _strip_kw(k_arg, with_comma=True),
    _strip_kw(v_arg, with_comma=bool(mapped_rest)),
    *mapped_rest,
  ]
  return node.with_changes(args=new_args)
