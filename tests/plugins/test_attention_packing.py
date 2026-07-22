"""Test suite for the Attention Packing module."""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.attention_packing import repack_attn_keras, repack_attn_flax, repack_attn_torch


@pytest.fixture
def mock_ctx():
  """Provides a mock ctx for testing."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "target.Attention"
  return ctx


def parse_call_node(code):
  """Parses call node."""
  tree = cst.parse_module(code)
  stmt = tree.body[0]
  if isinstance(stmt, cst.SimpleStatementLine):
    body = stmt.body[0]
    if isinstance(body, cst.Assign):
      return body.value
    if isinstance(body, cst.Expr):
      return body.value
  raise ValueError(f"Could not extract Call node from: {code}")


def to_code(node):
  """Helper to to code."""
  return cst.Module(body=[cst.SimpleStatementLine([cst.Expr(node)])]).code


def test_keras_strategy_constructor_happy_path(mock_ctx):
  """Verifies the behavior of Keras strategy constructor happy path."""
  mock_ctx.lookup_api.return_value = "keras.layers.MultiHeadAttention"
  code = "m = torch.nn.MultiheadAttention(embed_dim=256, num_heads=8)"
  call_node = parse_call_node(code)
  res = repack_attn_keras(call_node, mock_ctx)
  res_code = to_code(res)
  assert "keras.layers.MultiHeadAttention" in res_code
  assert "key_dim=256" in res_code


def test_keras_strategy_constructor_missing_api_aborts(mock_ctx):
  """Verifies the behavior of Keras strategy constructor missing API aborts."""
  mock_ctx.lookup_api.return_value = None
  code = "m = torch.nn.MultiheadAttention(embed_dim=256, num_heads=8)"
  call_node = parse_call_node(code)
  res = repack_attn_keras(call_node, mock_ctx)
  assert res is call_node


def test_keras_strategy_forward(mock_ctx):
  """Verifies the behavior of Keras strategy forward."""
  code = "y = self.attn(q, k, v, attn_mask=m)"
  call_node = parse_call_node(code)
  res = repack_attn_keras(call_node, mock_ctx)
  res_code = to_code(res)
  clean = res_code.replace(" ", "")
  assert "(q,v," in clean
  assert "key=k" in clean
  assert "attention_mask=m" in clean
  assert "attn_mask" not in clean


def test_flax_strategy_constructor_happy_path(mock_ctx):
  """Verifies the behavior of Flax strategy constructor happy path."""
  mock_ctx.lookup_api.return_value = "flax.nnx.MultiHeadAttention"
  code = "m = MultiheadAttention(embed_dim=10, num_heads=2)"
  call_node = parse_call_node(code)
  res = repack_attn_flax(call_node, mock_ctx)
  res_code = to_code(res)
  assert "flax.nnx.MultiHeadAttention" in res_code


def test_flax_strategy_constructor_missing_api_aborts(mock_ctx):
  """Verifies the behavior of Flax strategy constructor missing API aborts."""
  mock_ctx.lookup_api.return_value = None
  code = "m = MultiheadAttention(embed_dim=10, num_heads=2)"
  call_node = parse_call_node(code)
  res = repack_attn_flax(call_node, mock_ctx)
  assert res is call_node


def test_flax_strategy_forward(mock_ctx):
  """Verifies the behavior of Flax strategy forward."""
  code = "y = self.attn(q, k, v, key_padding_mask=m)"
  call_node = parse_call_node(code)
  res = repack_attn_flax(call_node, mock_ctx)
  res_code = to_code(res)
  assert "mask=m" in res_code
  assert "key_padding_mask" not in res_code


def test_torch_strategy_constructor_happy_path(mock_ctx):
  """Verifies the behavior of PyTorch strategy constructor happy path."""
  mock_ctx.lookup_api.return_value = "torch.nn.MultiheadAttention"
  code = "m = keras.layers.MultiHeadAttention(key_dim=256, in_features=128, dropout_rate=0.1, other_arg=1)"
  call_node = parse_call_node(code)
  res = repack_attn_torch(call_node, mock_ctx)
  res_code = to_code(res).replace(" ", "")
  assert "torch.nn.MultiheadAttention" in res_code
  assert "embed_dim=256" in res_code
  assert "embed_dim=128" in res_code
  assert "dropout=0.1" in res_code
  assert "other_arg=1" in res_code
  assert "batch_first=True" in res_code


def test_torch_strategy_constructor_with_batch_first(mock_ctx):
  """Verifies the behavior of PyTorch strategy constructor with batch first."""
  mock_ctx.lookup_api.return_value = "torch.nn.MultiheadAttention"
  code = "m = Attention(key_dim=256, batch_first=False)"
  call_node = parse_call_node(code)
  res = repack_attn_torch(call_node, mock_ctx)
  res_code = to_code(res).replace(" ", "")
  assert "batch_first=False" in res_code


def test_torch_strategy_constructor_missing_api(mock_ctx):
  """Verifies the behavior of PyTorch strategy constructor missing API."""
  mock_ctx.lookup_api.return_value = None
  code = "m = keras.layers.MultiHeadAttention(key_dim=256)"
  call_node = parse_call_node(code)
  res = repack_attn_torch(call_node, mock_ctx)
  assert res is call_node


def test_torch_strategy_forward(mock_ctx):
  """Verifies the behavior of PyTorch strategy forward."""
  code = "y = self.attn(q, k, v, mask=m1, attention_mask=m2, other_arg=2)"
  call_node = parse_call_node(code)
  res = repack_attn_torch(call_node, mock_ctx)
  res_code = to_code(res).replace(" ", "")
  assert "(q,k,v" in res_code
  assert "attn_mask=m1" in res_code
  assert "attn_mask=m2" in res_code
  assert "other_arg=2" in res_code


def test_torch_strategy_forward_keras_style(mock_ctx):
  """Verifies the behavior of PyTorch strategy forward Keras style."""
  code2 = "y = self.attn(q, key=k, v=v, attention_mask=m)"
  call_node2 = parse_call_node(code2)
  res2 = repack_attn_torch(call_node2, mock_ctx)
  res_code2 = to_code(res2).replace(" ", "")
  assert "(q,k,v=v" in res_code2
  assert "attn_mask=m" in res_code2


def test_torch_strategy_forward_too_few_args(mock_ctx):
  """Verifies the behavior of PyTorch strategy forward too few arguments."""
  code = "y = self.attn(q, k)"
  call_node = parse_call_node(code)
  res = repack_attn_torch(call_node, mock_ctx)
  assert res is call_node


def test_torch_constructor_fallback(mock_ctx):
  """Verifies the behavior of PyTorch constructor fallback."""
  mock_ctx.lookup_api.return_value = "torch.nn.MultiheadAttention"
  code = "m = Attention(256, 8)"
  call_node = parse_call_node(code)
  res = repack_attn_torch(call_node, mock_ctx)
  res_code = to_code(res).replace(" ", "")
  assert "torch.nn.MultiheadAttention(256,8,batch_first=True)" in res_code


def test_keras_strategy_constructor_positional(mock_ctx):
  """Verifies the behavior of Keras strategy constructor positional."""
  mock_ctx.lookup_api.return_value = "keras.layers.MultiHeadAttention"
  code = "m = torch.nn.MultiheadAttention(256, num_heads=8)"
  call_node = parse_call_node(code)
  res = repack_attn_keras(call_node, mock_ctx)
  res_code = to_code(res)
  assert "256" in res_code


def test_keras_strategy_forward_positional_after_kwargs(mock_ctx):
  """Verifies the behavior of Keras strategy forward positional after keyword arguments."""
  code = "y = self.attn(q, k, v, 10)"
  call_node = parse_call_node(code)
  with pytest.raises(cst.CSTValidationError) as excinfo:
    repack_attn_keras(call_node, mock_ctx)
  assert "Cannot have positional argument after keyword argument" in str(excinfo.value)


def test_keras_strategy_forward_too_few_args(mock_ctx):
  """Verifies the behavior of Keras strategy forward too few arguments."""
  code = "y = self.attn(q, k)"
  call_node = parse_call_node(code)
  res = repack_attn_keras(call_node, mock_ctx)
  assert res is call_node


def test_flax_strategy_forward_too_few_args(mock_ctx):
  """Verifies the behavior of Flax strategy forward too few arguments."""
  code = "y = self.attn(q, k)"
  call_node = parse_call_node(code)
  res = repack_attn_flax(call_node, mock_ctx)
  assert res is call_node


def test_flax_strategy_forward_other_args(mock_ctx):
  """Verifies the behavior of Flax strategy forward other arguments."""
  code = "y = self.attn(q, k, v, 10, other_arg=2)"
  call_node = parse_call_node(code)
  res = repack_attn_flax(call_node, mock_ctx)
  res_code = to_code(res)
  assert "10" in res_code
  assert "other_arg=2" in res_code


def test_keras_strategy_forward_other_kwargs(mock_ctx):
  """Verifies the behavior of Keras strategy forward other keyword arguments."""
  code = "y = self.attn(q, k, v, other_arg=2)"
  call_node = parse_call_node(code)
  res = repack_attn_keras(call_node, mock_ctx)
  res_code = to_code(res)
  assert "other_arg=2" in res_code
