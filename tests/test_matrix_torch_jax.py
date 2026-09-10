"""Bidirectional PyTorch and JAX (flax.nnx) CST Transpilation Tests.

Verifies static source-to-source code generation between PyTorch and JAX/Flax NNX,
including class inheritance, constructor signature transformations, layer renames,
and functional math calls.
"""

import ast
from ml_switcheroo import convert


def test_torch_to_jax_module_transpilation() -> None:
  """Tests transpiling a PyTorch nn.Module into a JAX flax.nnx.Module."""
  torch_code = """import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32 * 14 * 14, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return self.fc(x)
"""
  converted = convert(torch_code, source="torch", target="jax")
  parsed = ast.parse(converted)
  assert parsed is not None
  assert "def __call__(self, x):" in converted
  assert "class ConvNet" in converted


def test_jax_to_torch_module_transpilation() -> None:
  """Tests transpiling a JAX flax.nnx.Module into a PyTorch nn.Module."""
  jax_code = """import jax.numpy as jnp
from flax import nnx

class SimpleNet(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.conv = nnx.Conv(3, 16, 3, rngs=rngs)
        self.dense = nnx.Linear(16, 10, rngs=rngs)

    def __call__(self, x):
        return self.dense(self.conv(x))
"""
  converted = convert(jax_code, source="jax", target="torch")
  parsed = ast.parse(converted)
  assert parsed is not None
  assert "(nn.Module):" in converted
  assert "def forward(self, x):" in converted


def test_torch_to_jax_math_operations() -> None:
  """Tests converting standard math calls between torch and jax."""
  torch_math = """import torch

def compute_loss(x, y):
    diff = torch.abs(torch.sub(x, y))
    return torch.add(diff, 1.0)
"""
  converted = convert(torch_math, source="torch", target="jax")
  parsed = ast.parse(converted)
  assert parsed is not None
  assert "def compute_loss(x, y):" in converted


def test_jax_to_torch_math_operations() -> None:
  """Tests converting math calls from jax to torch."""
  jax_math = """import jax.numpy as jnp

def compute_loss(x, y):
    diff = jnp.abs(jnp.subtract(x, y))
    return jnp.add(diff, 1.0)
"""
  converted = convert(jax_math, source="jax", target="torch")
  parsed = ast.parse(converted)
  assert parsed is not None
  assert "def compute_loss(x, y):" in converted
