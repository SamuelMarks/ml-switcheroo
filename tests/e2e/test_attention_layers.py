"""Test suite for the Attention Layers module."""

import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.core.hooks import register_hook
from ml_switcheroo.plugins.attention_packing import repack_attn_flax, repack_attn_keras
from ml_switcheroo.semantics.manager import SemanticsManager

SOURCE_TORCH: str = "\nimport torch.nn as nn\n\nclass MyAttn(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=8)\n\n    def forward(self, q, k, v, mask):\n        out, _ = self.attn(q, k, v, key_padding_mask=mask)\n        return out\n"


@pytest.fixture
def attn_semantics(monkeypatch: pytest.MonkeyPatch) -> SemanticsManager:
  """Create a configured `SemanticsManager` fixture for attention layers tests.

  Args:
      monkeypatch: Pytest monkeypatch fixture.

  Returns:
      Configured `attn_semantics` fixture value (SemanticsManager).
  """
  register_hook("repack_attn_keras")(repack_attn_keras)
  register_hook("repack_attn_flax")(repack_attn_flax)
  mgr = SemanticsManager()
  mgr.data["MultiheadAttention"] = {
    "std_args": ["embed_dim", "num_heads"],
    "variants": {
      "torch": {"api": "torch.nn.MultiheadAttention"},
      "keras": {
        "api": "keras.layers.MultiHeadAttention",
        "args": {"embed_dim": "key_dim"},
        "requires_plugin": "repack_attn_keras",
      },
      "flax_nnx": {"api": "flax.nnx.MultiHeadAttention", "requires_plugin": "repack_attn_flax"},
    },
  }
  mgr._reverse_index["torch.nn.MultiheadAttention"] = ("MultiheadAttention", mgr.data["MultiheadAttention"])
  mgr._reverse_index["self.attn"] = ("MultiheadAttention", mgr.data["MultiheadAttention"])
  mgr.framework_configs = {
    "torch": {"traits": {"module_base": "torch.nn.Module", "forward_method": "forward"}},
    "keras": {"traits": {"module_base": "keras.Model", "forward_method": "call"}},
    "flax_nnx": {"traits": {"module_base": "flax.nnx.Module", "forward_method": "__call__"}},
  }
  mgr._providers = {"keras": {}}
  mgr._source_registry = {}
  monkeypatch.setattr(mgr, "is_verified", lambda x: True, raising=False)
  return mgr


def test_torch_to_keras_attention(attn_semantics: SemanticsManager) -> None:
  """Verify `RuntimeConfig` emits `class MyAttn(keras.Model):` for torch to keras attention.

  Args:
      attn_semantics: Input `attn_semantics` parameter value (`SemanticsManager`).
  """
  config = RuntimeConfig(source_framework="torch", target_framework="keras", strict_mode=False)
  engine = ASTEngine(semantics=attn_semantics, config=config)
  result: ConversionResult = engine.run(SOURCE_TORCH)
  code: str = result.code
  assert "class MyAttn(keras.Model):" in code
  assert ".MultiHeadAttention" in code
  assert "key_dim=256" in code
  assert "self.attn(q, v, key = k" in code.replace("key=", "key = ")


def test_torch_to_flax_attention(attn_semantics: SemanticsManager) -> None:
  """Verify `RuntimeConfig` emits `self.attn(q, k, v` for torch to flax attention.

  Args:
      attn_semantics: Input `attn_semantics` parameter value (`SemanticsManager`).
  """
  config = RuntimeConfig(source_framework="torch", target_framework="flax_nnx", strict_mode=False)
  engine = ASTEngine(semantics=attn_semantics, config=config)
  result: ConversionResult = engine.run(SOURCE_TORCH)
  code: str = result.code
  assert "class MyAttn(nnx.Module):" in code or "class MyAttn(flax.nnx.Module):" in code
  assert "self.attn(q, k, v" in code
  assert "mask = mask" in code.replace("mask=", "mask = ")
  assert "key_padding_mask" not in code
