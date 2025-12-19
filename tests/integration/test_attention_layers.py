"""
Integration Tests for MultiHead Attention Layer Transpilation.
"""

import pytest

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.hooks import _HOOKS
from ml_switcheroo.plugins.attention_packing import repack_attention

SOURCE_TORCH = """
import torch.nn as nn

class MyAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=8)

    def forward(self, q, k, v, mask):
        out, _ = self.attn(q, k, v, key_padding_mask=mask)
        return out
"""


@pytest.fixture
def attn_semantics():
  _HOOKS["repack_attention_call"] = repack_attention
  mgr = SemanticsManager()

  mgr.data["MultiheadAttention"] = {
    "std_args": ["embed_dim", "num_heads"],
    "variants": {
      "torch": {"api": "torch.nn.MultiheadAttention"},
      "keras": {
        "api": "keras.layers.MultiHeadAttention",
        "args": {"embed_dim": "key_dim"},
        "requires_plugin": "repack_attention_call",
      },
      "flax_nnx": {"api": "flax.nnx.MultiHeadAttention", "requires_plugin": "repack_attention_call"},
    },
  }

  mgr._reverse_index["torch.nn.MultiheadAttention"] = ("MultiheadAttention", mgr.data["MultiheadAttention"])
  mgr._reverse_index["self.attn"] = ("MultiheadAttention", mgr.data["MultiheadAttention"])

  mgr.framework_configs = {
    "torch": {"traits": {"module_base": "torch.nn.Module", "forward_method": "forward"}},
    "keras": {"traits": {"module_base": "keras.Model", "forward_method": "call"}},
    "flax_nnx": {"traits": {"module_base": "flax.nnx.Module", "forward_method": "__call__"}},
  }

  mgr.is_verified = lambda x: True
  return mgr


def test_torch_to_keras_attention(attn_semantics):
  config = RuntimeConfig(source_framework="torch", target_framework="keras", strict_mode=False)
  engine = ASTEngine(semantics=attn_semantics, config=config)
  result = engine.run(SOURCE_TORCH)
  code = result.code

  assert "class MyAttn(keras.Model):" in code
  assert "keras.layers.MultiHeadAttention" in code
  assert "key_dim=256" in code
  assert "self.attn(q, v, key = k" in code.replace("key=", "key = ")


def test_torch_to_flax_attention(attn_semantics):
  config = RuntimeConfig(source_framework="torch", target_framework="flax_nnx", strict_mode=False)
  engine = ASTEngine(semantics=attn_semantics, config=config)
  result = engine.run(SOURCE_TORCH)
  code = result.code

  # Relaxed assertion for module base class to handle aliasing
  assert "class MyAttn(nnx.Module):" in code or "class MyAttn(flax.nnx.Module):" in code

  assert "self.attn(q, k, v" in code
  assert "mask = mask" in code.replace("mask=", "mask = ")
  assert "key_padding_mask" not in code
