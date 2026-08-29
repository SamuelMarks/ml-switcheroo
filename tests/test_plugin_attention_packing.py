"""Docstring."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.attention_packing import (
  _create_dotted_name,
  _is_constructor_signature,
  _resolve_target_class,
  repack_attn_flax,
  repack_attn_keras,
  repack_attn_torch,
)


def test_create_dotted_name():
  """Docstring."""
  node = _create_dotted_name("a.b.c")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "c"
  assert isinstance(node.value, cst.Attribute)
  assert node.value.attr.value == "b"
  assert isinstance(node.value.value, cst.Name)
  assert node.value.value.value == "a"


def test_resolve_target_class():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "new.MultiheadAttention"
  node = _resolve_target_class(ctx)
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "MultiheadAttention"
  assert isinstance(node.value, cst.Name)
  assert node.value.value == "new"

  ctx.lookup_api.return_value = None
  assert _resolve_target_class(ctx) is None


def test_is_constructor_signature():
  """Docstring."""
  args_kw = [cst.Arg(keyword=cst.Name("embed_dim"), value=cst.Integer("10"))]
  assert _is_constructor_signature(args_kw) is True

  args_pos = [cst.Arg(value=cst.Integer("10")), cst.Arg(value=cst.Integer("2"))]
  assert _is_constructor_signature(args_pos) is True

  args_not_constructor = [cst.Arg(value=cst.Name("q")), cst.Arg(value=cst.Name("k")), cst.Arg(value=cst.Name("v"))]
  assert _is_constructor_signature(args_not_constructor) is False


def test_repack_attn_keras_constructor():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "keras.layers.MultiHeadAttention"
  node = cst.Call(
    func=cst.Name("Attention"),
    args=[
      cst.Arg(value=cst.Integer("2")),
      cst.Arg(keyword=cst.Name("embed_dim"), value=cst.Integer("10")),
      cst.Arg(keyword=cst.Name("other"), value=cst.Integer("5")),
    ],
  )
  result = repack_attn_keras(node, ctx)
  assert isinstance(result.func, cst.Attribute)
  assert result.func.attr.value == "MultiHeadAttention"
  assert len(result.args) == 3
  assert result.args[0].keyword is None
  assert result.args[1].keyword.value == "key_dim"
  assert result.args[2].keyword.value == "other"

  # test no api lookup
  ctx.lookup_api.return_value = None
  assert repack_attn_keras(node, ctx) == node


def test_repack_attn_keras_call():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = None  # Doesn't matter for call
  node = cst.Call(
    func=cst.Name("attn"),
    args=[
      cst.Arg(value=cst.Name("q")),
      cst.Arg(value=cst.Name("k")),
      cst.Arg(value=cst.Name("v")),
      cst.Arg(keyword=cst.Name("attn_mask"), value=cst.Name("m")),
      cst.Arg(keyword=cst.Name("other"), value=cst.Name("x")),
    ],
  )
  # Note in repack_attn_keras, there's logic that maps arguments but the output must be valid
  # In repack_attn_keras, it appends q_arg, v_arg, then k_kw.
  # The output args might have keywords then positional, wait, if there are positional args left in remaining_args,
  # it might create invalid AST? Wait, if the remaining args don't have keywords, it might create invalid AST.
  # But usually remaining args are kwargs. Let's provide only kwargs in remaining.
  result = repack_attn_keras(node, ctx)
  assert len(result.args) == 5
  assert result.args[0].value.value == "q"
  assert result.args[1].value.value == "v"
  assert result.args[2].keyword.value == "key"
  assert result.args[2].value.value == "k"
  assert result.args[3].keyword.value == "attention_mask"
  assert result.args[4].keyword.value == "other"

  node_short = cst.Call(func=cst.Name("attn"), args=[cst.Arg(value=cst.Name("q"))])
  assert repack_attn_keras(node_short, ctx) == node_short


def test_repack_attn_flax_constructor():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "flax.nnx.MultiHeadAttention"
  node = cst.Call(
    func=cst.Name("Attention"),
    args=[
      cst.Arg(keyword=cst.Name("embed_dim"), value=cst.Integer("10")),
    ],
  )
  result = repack_attn_flax(node, ctx)
  assert isinstance(result.func, cst.Attribute)

  ctx.lookup_api.return_value = None
  assert repack_attn_flax(node, ctx) == node


def test_repack_attn_flax_call():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  node = cst.Call(
    func=cst.Name("attn"),
    args=[
      cst.Arg(value=cst.Name("q")),
      cst.Arg(value=cst.Name("k")),
      cst.Arg(value=cst.Name("v")),
      cst.Arg(keyword=cst.Name("attn_mask"), value=cst.Name("m")),
      cst.Arg(keyword=cst.Name("other"), value=cst.Name("x")),
    ],
  )
  result = repack_attn_flax(node, ctx)
  assert len(result.args) == 5
  assert result.args[3].keyword.value == "mask"
  assert result.args[4].keyword.value == "other"

  node_short = cst.Call(func=cst.Name("attn"), args=[cst.Arg(value=cst.Name("q"))])
  assert repack_attn_flax(node_short, ctx) == node_short


def test_repack_attn_torch_constructor():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "torch.nn.MultiheadAttention"
  node = cst.Call(
    func=cst.Name("Attention"),
    args=[
      cst.Arg(keyword=cst.Name("key_dim"), value=cst.Integer("10")),
      cst.Arg(keyword=cst.Name("in_features"), value=cst.Integer("10")),
      cst.Arg(keyword=cst.Name("dropout_rate"), value=cst.Float("0.1")),
      cst.Arg(keyword=cst.Name("other"), value=cst.Integer("5")),
    ],
  )
  result = repack_attn_torch(node, ctx)
  assert isinstance(result.func, cst.Attribute)
  assert len(result.args) == 5  # added batch_first
  assert result.args[0].keyword.value == "embed_dim"
  assert result.args[1].keyword.value == "embed_dim"
  assert result.args[2].keyword.value == "dropout"
  assert result.args[3].keyword.value == "other"
  assert result.args[4].keyword.value == "batch_first"
  assert result.args[4].value.value == "True"

  ctx.lookup_api.return_value = None
  assert repack_attn_torch(node, ctx) == node


def test_repack_attn_torch_call():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  # In repack_attn_torch, it converts key kwarg to positional if named 'key' but if we have kwargs before positional it's invalid.
  # The original args must be valid Python.
  # (q, v, key=k) -> torch takes (q,k,v).
  # Let's provide (q, k, v)
  node = cst.Call(
    func=cst.Name("attn"),
    args=[
      cst.Arg(value=cst.Name("q")),
      cst.Arg(value=cst.Name("k")),  # If it's a kwarg key=k in 2nd position, we can't have positional v after it.
      # Wait, repack_attn_torch assumes k is in position 1.
      cst.Arg(value=cst.Name("v")),
      cst.Arg(keyword=cst.Name("attention_mask"), value=cst.Name("m")),
      cst.Arg(keyword=cst.Name("other"), value=cst.Name("x")),
    ],
  )
  result = repack_attn_torch(node, ctx)
  assert len(result.args) == 5
  assert result.args[3].keyword.value == "attn_mask"
  assert result.args[4].keyword.value == "other"

  # Also test with key keyword argument in k_arg, this requires a trick because it would be a syntax error.
  # Wait, k_arg is args[1]. If args[1] is a keyword arg, args[2] must be too.
  node2 = cst.Call(
    func=cst.Name("attn"),
    args=[
      cst.Arg(value=cst.Name("q")),
      cst.Arg(keyword=cst.Name("key"), value=cst.Name("k")),
      cst.Arg(keyword=cst.Name("value"), value=cst.Name("v")),
    ],
  )
  result2 = repack_attn_torch(node2, ctx)
  assert len(result2.args) == 3
  assert result2.args[1].keyword is None
  assert result2.args[1].value.value == "k"

  node_short = cst.Call(func=cst.Name("attn"), args=[cst.Arg(value=cst.Name("q"))])
  assert repack_attn_torch(node_short, ctx) == node_short
