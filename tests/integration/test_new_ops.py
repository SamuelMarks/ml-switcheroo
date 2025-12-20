# tests/integration/test_new_ops.py

"""
Integration Tests for Expanded Operations (Features 08+).
Verifies specific mappings for Embedding, LayerNorm, GELU, and ArgMax.
"""

import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager

# --- Source PyTorch Code ---
SOURCE_TORCH_NN = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # Embedding
        self.embed = nn.Embedding(vocab_size, d_model)
        # LayerNorm
        self.ln = nn.LayerNorm(d_model, eps=1e-6)
        # GELU activation
        self.act = nn.GELU()

    def forward(self, x):
        x = self.embed(x)
        x = self.ln(x)
        x = self.act(x)
        # ArgMax
        return torch.argmax(x, dim=-1)
"""

# Validated Expected Outputs (Positional args preserved where applicable)
EXPECTED_FLAX_NNX = """
import flax.nnx as nnx
import jax.numpy as jnp

class TransformerBlock(nnx.Module):
    def __init__(self, vocab_size, d_model, rngs: nnx.Rngs):
        # Flax Embed(num_embeddings, features, ...)
        self.embed = nnx.Embed(vocab_size, d_model, rngs=rngs)
        self.ln = nnx.LayerNorm(d_model, epsilon=1e-6, rngs=rngs)
        self.act = nnx.gelu

    def __call__(self, x):
        x = self.embed(x)
        x = self.ln(x)
        x = self.act(x)
        return jnp.argmax(x, axis=-1)
"""

EXPECTED_MLX = """
import mlx.nn as nn
import mlx.core as mx

class TransformerBlock(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.ln = nn.LayerNorm(d_model, eps=1e-6)
        self.act = nn.GELU()

    def __call__(self, x):
        x = self.embed(x)
        x = self.ln(x)
        x = self.act(x)
        return mx.argmax(x, axis=-1)
"""


@pytest.fixture(scope="module")
def semantics():
  mgr = SemanticsManager()

  # 1. Embed
  embed_def = {
    "std_args": ["num_embeddings", "embedding_dim"],
    "variants": {
      "torch": {"api": "torch.nn.Embedding"},
      "flax_nnx": {"api": "flax.nnx.Embed", "args": {"embedding_dim": "features"}},
      "mlx": {"api": "mlx.nn.Embedding", "args": {"embedding_dim": "dims"}},
    },
  }

  # 2. LayerNorm
  ln_def = {
    "std_args": ["normalized_shape", "eps"],
    "variants": {
      "torch": {"api": "torch.nn.LayerNorm"},
      # Mapping normalized_shape -> num_features
      "flax_nnx": {"api": "flax.nnx.LayerNorm", "args": {"normalized_shape": "num_features", "eps": "epsilon"}},
      "mlx": {"api": "mlx.nn.LayerNorm", "args": {"normalized_shape": "dims"}},
    },
  }

  # 3. ArgMax
  argmax_def = {
    "std_args": ["input", "dim"],
    "variants": {
      "torch": {"api": "torch.argmax"},
      "flax_nnx": {"api": "jnp.argmax", "args": {"dim": "axis"}},
      "mlx": {"api": "mx.argmax", "args": {"dim": "axis"}},
    },
  }

  # 4. GELU
  gelu_def = {
    "std_args": [],
    "variants": {"torch": {"api": "torch.nn.GELU"}, "flax_nnx": {"api": "flax.nnx.gelu"}, "mlx": {"api": "mlx.nn.GELU"}},
  }

  def inject(name, defn):
    mgr.data[name] = defn
    for fw, v in defn["variants"].items():
      mgr._reverse_index[v["api"]] = (name, defn)
    # Mark as Neural for state injection (except ArgMax)
    if name != "ArgMax":
      mgr._key_origins[name] = "neural"

  inject("Embedding", embed_def)
  inject("LayerNorm", ln_def)
  inject("ArgMax", argmax_def)
  inject("GELU", gelu_def)

  # Aliases
  mgr.framework_configs["flax_nnx"] = {
    "alias": {"module": "flax.nnx", "name": "nnx"},
    "traits": {
      "module_base": "flax.nnx.Module",
      "forward_method": "__call__",
      "inject_magic_args": [("rngs", "flax.nnx.Rngs")],
    },
  }

  # Import Maps (Mocked for isolation)
  mgr.import_data["torch.nn"] = {
    "variants": {
      "flax_nnx": {"root": "flax", "sub": "nnx", "alias": "nnx"},
      "mlx": {"root": "mlx", "sub": "nn", "alias": "nn"},
    }
  }
  mgr.import_data["jnp"] = {"variants": {"flax_nnx": {"root": "jax", "sub": "numpy", "alias": "jnp"}}}
  mgr.import_data["torch"] = {"variants": {"mlx": {"root": "mlx", "sub": "core", "alias": "mx"}}}

  # Special case: Map torch.nn.functional to nothing for MLX to allow cleaning if unused, or map it if used.
  # In source: 'import torch.nn.functional as F'. F is unused in logic, so ImportFixer should prune it.

  return mgr


def test_torch_to_flax_nnx_advanced_layers(semantics):
  config = RuntimeConfig(source_framework="torch", target_framework="flax_nnx", strict_mode=False)
  engine = ASTEngine(semantics=semantics, config=config)
  result = engine.run(SOURCE_TORCH_NN)

  assert result.success
  # Check key mappings (positional args are valid)
  assert "nnx.Embed(vocab_size, d_model" in result.code
  assert "nnx.LayerNorm(d_model" in result.code
  assert "rngs=rngs" in result.code  # State injection
  assert "jnp.argmax(x, axis=-1)" in result.code


def test_torch_to_mlx_advanced_layers(semantics):
  config = RuntimeConfig(source_framework="torch", target_framework="mlx", strict_mode=False)
  engine = ASTEngine(semantics=semantics, config=config)
  result = engine.run(SOURCE_TORCH_NN)

  assert result.success
  # Check MLX outputs (positional valid)
  assert "nn.Embedding(vocab_size, d_model)" in result.code
  assert "nn.LayerNorm(d_model" in result.code
  assert "mx.argmax(x, axis=-1)" in result.code
