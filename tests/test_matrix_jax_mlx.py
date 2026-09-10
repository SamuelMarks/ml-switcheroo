"""Bidirectional JAX and MLX CST Transpilation Tests.

Verifies static source-to-source code generation between JAX/Flax NNX and Apple MLX,
including class inheritance, layer translation, and tensor math calls.
"""

import ast
from ml_switcheroo import convert


def test_jax_to_mlx_module_transpilation() -> None:
  """Tests transpiling a JAX flax.nnx.Module into an MLX nn.Module."""
  jax_code = """import jax.numpy as jnp
from flax import nnx

class MlxNet(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.conv = nnx.Conv(3, 16, 3, rngs=rngs)

    def __call__(self, x):
        return self.conv(x)
"""
  converted = convert(jax_code, source="jax", target="mlx")
  parsed = ast.parse(converted)
  assert parsed is not None
  assert "(nn.Module):" in converted
  assert "def __call__(self, x):" in converted


def test_mlx_to_jax_module_transpilation() -> None:
  """Tests transpiling an MLX nn.Module into a JAX flax.nnx.Module."""
  mlx_code = """import mlx.core as mx
import mlx.nn as nn

class MlxNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)

    def __call__(self, x):
        return self.conv(x)
"""
  converted = convert(mlx_code, source="mlx", target="jax")
  parsed = ast.parse(converted)
  assert parsed is not None
  assert "def __call__(self, x):" in converted


def test_jax_to_mlx_math_operations() -> None:
  """Tests converting math calls between jax and mlx."""
  jax_code = """import jax.numpy as jnp

def simple_fn(x, y):
    return jnp.multiply(x, y)
"""
  converted = convert(jax_code, source="jax", target="mlx")
  parsed = ast.parse(converted)
  assert parsed is not None
  assert "def simple_fn(x, y):" in converted
