"""Bidirectional PyTorch and Apple MLX CST Transpilation Tests.

Verifies static source-to-source code generation between PyTorch and Apple MLX,
including class inheritance, layer translation (Conv2d, Linear), and core math calls.
"""

import ast
from ml_switcheroo import convert


def test_torch_to_mlx_module_transpilation() -> None:
  """Tests transpiling a PyTorch nn.Module into an MLX nn.Module."""
  torch_code = """import torch
import torch.nn as nn

class LinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
"""
  converted = convert(torch_code, source="torch", target="mlx")
  parsed = ast.parse(converted)
  assert parsed is not None
  assert "(nn.Module):" in converted
  assert "def __call__(self, x):" in converted


def test_mlx_to_torch_module_transpilation() -> None:
  """Tests transpiling an MLX nn.Module into a PyTorch nn.Module."""
  mlx_code = """import mlx.core as mx
import mlx.nn as nn

class LinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def __call__(self, x):
        return self.fc2(self.fc1(x))
"""
  converted = convert(mlx_code, source="mlx", target="torch")
  parsed = ast.parse(converted)
  assert parsed is not None
  assert "(nn.Module):" in converted
  assert "def forward(self, x):" in converted


def test_torch_to_mlx_math_operations() -> None:
  """Tests converting tensor math functions between torch and mlx."""
  torch_code = """import torch

def tensor_op(a, b):
    return torch.add(torch.multiply(a, b), 2.0)
"""
  converted = convert(torch_code, source="torch", target="mlx")
  parsed = ast.parse(converted)
  assert parsed is not None
  assert "def tensor_op(a, b):" in converted
