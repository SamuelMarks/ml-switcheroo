"""Test suite for the New Ops module."""

from pydantic import JsonValue

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.semantics.manager import SemanticsManager

SOURCE_TORCH_NN: str = "\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\nclass TransformerBlock(nn.Module):\n    def __init__(self, vocab_size, d_model):\n        super().__init__()\n        # Embedding\n        self.embed = nn.Embedding(vocab_size, d_model)\n        # LayerNorm\n        self.ln = nn.LayerNorm(d_model, eps=1e-6)\n        # GELU activation\n        self.act = nn.GELU()\n\n    def forward(self, x):\n        x = self.embed(x)\n        x = self.ln(x)\n        x = self.act(x)\n        # ArgMax\n        return torch.argmax(x, dim=-1)\n"
EXPECTED_FLAX_NNX: str = "\nimport flax.nnx as nnx\nimport jax.numpy as jnp\n\nclass TransformerBlock(nnx.Module):\n    def __init__(self, vocab_size, d_model, rngs: nnx.Rngs):\n        # Flax Embed(num_embeddings, features, ...)\n        self.embed = nnx.Embed(vocab_size, d_model, rngs=rngs)\n        self.ln = nnx.LayerNorm(d_model, epsilon=1e-6, rngs=rngs)\n        self.act = nnx.gelu\n\n    def __call__(self, x):\n        x = self.embed(x)\n        x = self.ln(x)\n        x = self.act(x)\n        return jnp.argmax(x, axis=-1)\n"
EXPECTED_MLX: str = "\nimport mlx.nn as nn\nimport mlx.core as mx\n\nclass TransformerBlock(nn.Module):\n    def __init__(self, vocab_size, d_model):\n        super().__init__()\n        self.embed = nn.Embedding(vocab_size, d_model)\n        self.ln = nn.LayerNorm(d_model, eps=1e-6)\n        self.act = nn.GELU()\n\n    def __call__(self, x):\n        x = self.embed(x)\n        x = self.ln(x)\n        x = self.act(x)\n        return mx.argmax(x, axis=-1)\n"


@pytest.fixture(scope="module")
def semantics() -> SemanticsManager:
  """Helper to semantics.

  Returns:
      Configured `semantics` fixture value (SemanticsManager).
  """
  mgr = SemanticsManager()
  embed_def = {
    "std_args": ["num_embeddings", "embedding_dim"],
    "variants": {
      "torch": {"api": "torch.nn.Embedding"},
      "flax_nnx": {"api": "flax.nnx.Embed", "args": {"embedding_dim": "features"}},
      "mlx": {"api": "mlx.nn.Embedding", "args": {"embedding_dim": "dims"}},
    },
  }
  ln_def = {
    "std_args": ["normalized_shape", "eps"],
    "variants": {
      "torch": {"api": "torch.nn.LayerNorm"},
      "flax_nnx": {"api": "flax.nnx.LayerNorm", "args": {"normalized_shape": "num_features", "eps": "epsilon"}},
      "mlx": {"api": "mlx.nn.LayerNorm", "args": {"normalized_shape": "dims"}},
    },
  }
  argmax_def = {
    "std_args": ["input", "dim"],
    "variants": {
      "torch": {"api": "torch.argmax"},
      "flax_nnx": {"api": "jnp.argmax", "args": {"dim": "axis"}},
      "mlx": {"api": "mx.argmax", "args": {"dim": "axis"}},
    },
  }
  gelu_def = {
    "std_args": [],
    "variants": {"torch": {"api": "torch.nn.GELU"}, "flax_nnx": {"api": "flax.nnx.gelu"}, "mlx": {"api": "mlx.nn.GELU"}},
  }

  def inject(name: str, defn: dict[str, JsonValue]) -> None:
    """Injects .

    Args:
        name: Identifier or operation name string.
        defn: The defn (dict[str, JsonValue]) for inject.
    """
    mgr.data[name] = defn
    for fw, v in getattr(
      {
        "v": getattr(
          {"v": getattr({"v": getattr({"v": defn}, "__getitem__")("v")["variants"]}, "__getitem__")("v").items},
          "__getitem__",
        )("v")()
      },
      "__getitem__",
    )("v"):
      getattr(getattr({"v": getattr({"v": mgr}, "__getitem__")("v")._reverse_index}, "__getitem__")("v"), "__setitem__")(
        getattr({"v": v}, "__getitem__")("v")["api"],
        getattr(
          {"v": (getattr({"v": name}, "__getitem__")("v"), getattr({"v": defn}, "__getitem__")("v"))}, "__getitem__"
        )("v"),
      )
    if name != "ArgMax":
      mgr._key_origins[name] = "neural"

  getattr({"v": inject}, "__getitem__")("v")(
    getattr({"v": "Embedding"}, "__getitem__")("v"), getattr({"v": embed_def}, "__getitem__")("v")
  )
  getattr({"v": inject}, "__getitem__")("v")(
    getattr({"v": "LayerNorm"}, "__getitem__")("v"), getattr({"v": ln_def}, "__getitem__")("v")
  )
  getattr({"v": inject}, "__getitem__")("v")(
    getattr({"v": "ArgMax"}, "__getitem__")("v"), getattr({"v": argmax_def}, "__getitem__")("v")
  )
  getattr({"v": inject}, "__getitem__")("v")(
    getattr({"v": "GELU"}, "__getitem__")("v"), getattr({"v": gelu_def}, "__getitem__")("v")
  )
  getattr(getattr({"v": mgr.framework_configs}, "__getitem__")("v"), "__setitem__")(
    "flax_nnx",
    getattr(
      {
        "v": {
          "alias": {"module": "flax.nnx", "name": "nnx"},
          "traits": getattr(
            {
              "v": {
                "module_base": "flax.nnx.Module",
                "forward_method": "__call__",
                "inject_magic_args": getattr(
                  {
                    "v": [
                      getattr(
                        {
                          "v": (
                            getattr({"v": "rngs"}, "__getitem__")("v"),
                            getattr({"v": "flax.nnx.Rngs"}, "__getitem__")("v"),
                          )
                        },
                        "__getitem__",
                      )("v")
                    ]
                  },
                  "__getitem__",
                )("v"),
              }
            },
            "__getitem__",
          )("v"),
        }
      },
      "__getitem__",
    )("v"),
  )
  mgr._source_registry["torch.nn"] = ("torch", SemanticTier.NEURAL)
  mgr._source_registry["jnp"] = ("jax", SemanticTier.ARRAY_API)
  mgr._source_registry["torch"] = ("torch", SemanticTier.ARRAY_API)
  mgr._providers["flax_nnx"] = {
    SemanticTier.NEURAL: {"root": "flax", "sub": "nnx", "alias": "nnx"},
    SemanticTier.ARRAY_API: {"root": "jax", "sub": "numpy", "alias": "jnp"},
  }
  mgr._providers["mlx"] = {
    SemanticTier.NEURAL: {"root": "mlx", "sub": "nn", "alias": "nn"},
    SemanticTier.ARRAY_API: {"root": "mlx", "sub": "core", "alias": "mx"},
  }
  return mgr


def test_torch_to_flax_nnx_advanced_layers(semantics: SemanticsManager) -> None:
  """Verifies the behavior of PyTorch to Flax NNX advanced layers.

  Args:
      semantics: SemanticsManager instance or mock providing operation mappings.
  """
  config = RuntimeConfig(source_framework="torch", target_framework="flax_nnx", strict_mode=False)
  engine = ASTEngine(semantics=semantics, config=config)
  result: ConversionResult = engine.run(SOURCE_TORCH_NN)
  assert result.success
  assert "nnx.Embed(vocab_size, d_model" in result.code
  assert "nnx.LayerNorm(d_model" in result.code
  assert "rngs=rngs" in result.code
  assert "jnp.argmax(x, axis=-1)" in result.code


def test_torch_to_mlx_advanced_layers(semantics: SemanticsManager) -> None:
  """Verifies the behavior of PyTorch to MLX advanced layers.

  Args:
      semantics: SemanticsManager instance or mock providing operation mappings.
  """
  config = RuntimeConfig(source_framework="torch", target_framework="mlx", strict_mode=False)
  engine = ASTEngine(semantics=semantics, config=config)
  result: ConversionResult = engine.run(SOURCE_TORCH_NN)
  assert result.success
  assert "nn.Embedding(vocab_size, d_model)" in result.code
  assert "nn.LayerNorm(d_model" in result.code
  assert "mx.argmax(x, axis=-1)" in result.code
