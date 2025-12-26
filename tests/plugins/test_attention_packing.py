"""
Tests for MultiHead Attention Argument Alignment Plugin.
"""

import pytest
import libcst as cst

from ml_switcheroo.core.hooks import register_hook, HookContext
from ml_switcheroo.plugins.attention_packing import repack_attention_dispatch, repack_attn_keras, repack_attn_flax


@pytest.fixture
def mock_ctx():
  """Reduced Context object."""

  class Ctx:
    target_fw = "keras"

  return Ctx()


def parse_call_node(code):
  """
  Safely extract the Call node from a statement.
  Assumes `var = call(...)` or just `call(...)`.
  """
  tree = cst.parse_module(code)
  stmt = tree.body[0]

  # Unwrap SimpleStatementLine
  if isinstance(stmt, cst.SimpleStatementLine):
    body = stmt.body[0]
    # Check for Assignment
    if isinstance(body, cst.Assign):
      return body.value
    # Check for Expression
    if isinstance(body, cst.Expr):
      return body.value

  raise ValueError(f"Could not extract Call node from: {code}")


def to_code(node):
  return cst.Module(body=[cst.SimpleStatementLine([cst.Expr(node)])]).code


def test_keras_strategy_constructor(mock_ctx):
  """
  Constructor: torch.MultiheadAttention(embed_dim=256, num_heads=8)
  Expect: keras.layers.MultiHeadAttention(key_dim=256, num_heads=8)
  """
  code = "m = torch.nn.MultiheadAttention(embed_dim=256, num_heads=8)"
  call_node = parse_call_node(code)

  res = repack_attn_keras(call_node, mock_ctx)
  res_code = to_code(res)

  assert "keras.layers.MultiHeadAttention" in res_code
  assert "key_dim=256" in res_code
  assert "embed_dim" not in res_code


def test_keras_strategy_forward(mock_ctx):
  """
  Call: attn(q, k, v, attn_mask=m)
  Expect: call(q, v, key=k, attention_mask=m)
  """
  code = "y = self.attn(q, k, v, attn_mask=m)"
  call_node = parse_call_node(code)

  res = repack_attn_keras(call_node, mock_ctx)
  res_code = to_code(res)

  clean = res_code.replace(" ", "")
  assert "(q,v," in clean
  assert "key=k" in clean
  assert "attention_mask=m" in clean
  assert "attn_mask" not in clean


def test_flax_strategy_constructor(mock_ctx):
  """
  Constructor: torch...
  Expect: flax.nnx...
  """
  code = "m = MultiheadAttention(embed_dim=10, num_heads=2)"
  call_node = parse_call_node(code)

  mock_ctx.target_fw = "flax"
  res = repack_attn_flax(call_node, mock_ctx)
  res_code = to_code(res)

  assert "flax.nnx.MultiHeadAttention" in res_code


def test_flax_strategy_forward(mock_ctx):
  """
  Call: attn(q, k, v, key_padding_mask=m)
  Expect: attn(q, k, v, mask=m)
  """
  code = "y = self.attn(q, k, v, key_padding_mask=m)"
  call_node = parse_call_node(code)

  res = repack_attn_flax(call_node, mock_ctx)
  res_code = to_code(res)

  assert "mask=m" in res_code
  assert "key_padding_mask" not in res_code


def test_dispatch_logic(mock_ctx):
  """Verify dispatch routes based on framework string."""
  code = "y = MultiheadAttention(embed_dim=1, num_heads=1)"
  call_node = parse_call_node(code)

  # Keras Target
  mock_ctx.target_fw = "keras"
  res = repack_attention_dispatch(call_node, mock_ctx)
  assert "keras.layers" in to_code(res)

  # Flax Target
  mock_ctx.target_fw = "flax_nnx"
  res = repack_attention_dispatch(call_node, mock_ctx)
  assert "flax.nnx" in to_code(res)

  # Unknown Target
  mock_ctx.target_fw = "unknown"
  res = repack_attention_dispatch(call_node, mock_ctx)
  # Should be untouched
  assert "MultiheadAttention" in to_code(res)
  assert "keras" not in to_code(res)
