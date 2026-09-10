"""Bidirectional PyTorch and Keras 3 CST Transpilation Tests.

Verifies static source-to-source code generation between PyTorch and Keras 3,
including class inheritance, layer translation (Dense, Conv2D), and functional ops.
"""

import ast
from ml_switcheroo import convert


def test_torch_to_keras_module_transpilation() -> None:
  """Tests transpiling a PyTorch nn.Module into a Keras Model."""
  torch_code = """import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc(x)
"""
  converted = convert(torch_code, source="torch", target="keras")
  parsed = ast.parse(converted)
  assert parsed is not None
  assert "keras.Model" in converted or "keras.Layer" in converted
  assert "def call(self, x):" in converted


def test_keras_to_torch_module_transpilation() -> None:
  """Tests transpiling a Keras Model into a PyTorch nn.Module."""
  keras_code = """import keras
import keras.layers as layers

class Classifier(keras.Model):
    def __init__(self):
        super().__init__()
        self.fc = layers.Dense(10)

    def call(self, x):
        return self.fc(x)
"""
  converted = convert(keras_code, source="keras", target="torch")
  parsed = ast.parse(converted)
  assert parsed is not None
  assert "(nn.Module):" in converted
  assert "def forward(self, x):" in converted


def test_torch_to_keras_math_operations() -> None:
  """Tests converting math calls between torch and keras."""
  torch_code = """import torch

def relu_add(a, b):
    return torch.relu(torch.add(a, b))
"""
  converted = convert(torch_code, source="torch", target="keras")
  parsed = ast.parse(converted)
  assert parsed is not None
  assert "def relu_add(a, b):" in converted
