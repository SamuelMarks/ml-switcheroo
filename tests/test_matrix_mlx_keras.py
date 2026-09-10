"""Bidirectional MLX and Keras 3 CST Transpilation Tests.

Verifies static source-to-source code generation between Apple MLX and Keras 3,
including class inheritance, method renaming (__call__ <-> call), and operations.
"""

import ast
from ml_switcheroo import convert


def test_mlx_to_keras_module_transpilation() -> None:
  """Tests transpiling an MLX nn.Module into a Keras Model."""
  mlx_code = """import mlx.core as mx
import mlx.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 10)

    def __call__(self, x):
        return self.fc(x)
"""
  converted = convert(mlx_code, source="mlx", target="keras")
  parsed = ast.parse(converted)
  assert parsed is not None
  assert "keras.Model" in converted or "keras.Layer" in converted
  assert "def call(self, x):" in converted


def test_keras_to_mlx_module_transpilation() -> None:
  """Tests transpiling a Keras Model into an MLX nn.Module."""
  keras_code = """import keras
import keras.layers as layers

class Net(keras.Model):
    def __init__(self):
        super().__init__()
        self.fc = layers.Dense(10)

    def call(self, x):
        return self.fc(x)
"""
  converted = convert(keras_code, source="keras", target="mlx")
  parsed = ast.parse(converted)
  assert parsed is not None
  assert "(nn.Module):" in converted
  assert "def __call__(self, x):" in converted


def test_mlx_to_keras_math_operations() -> None:
  """Tests converting math calls between mlx and keras."""
  mlx_code = """import mlx.core as mx

def math_fn(a, b):
    return mx.add(a, b)
"""
  converted = convert(mlx_code, source="mlx", target="keras")
  parsed = ast.parse(converted)
  assert parsed is not None
  assert "def math_fn(a, b):" in converted
