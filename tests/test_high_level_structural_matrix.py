"""Comprehensive test suite for high-level structural & class transformations.

Covers Section 3.1 of TODO_PLAN.md:
- Base class rewriting across torch, flax_nnx, mlx, and keras.
- Lifecycle & execution method normalization (forward <-> __call__ <-> call).
- Parameter and submodule declaration with stateful/rngs handling and super().__init__().
"""

from ml_switcheroo.core.engine import ASTEngine

TORCH_CLASS_CODE: str = """import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)
"""

FLAX_NNX_CLASS_CODE: str = """import flax.nnx as nnx
import jax.numpy as jnp

class MyModel(nnx.Module):
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
        self.fc = nnx.Linear(in_features, out_features, rngs=rngs)

    def __call__(self, x):
        return self.fc(x)
"""

MLX_CLASS_CODE: str = """import mlx.core as mx
import mlx.nn as nn

class MyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def __call__(self, x):
        return self.fc(x)
"""

KERAS_CLASS_CODE: str = """import keras
import keras.layers as layers

class MyModel(keras.Layer):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = layers.Dense(in_features, out_features)

    def call(self, x):
        return self.fc(x)
"""


def test_torch_to_flax_nnx_structural() -> None:
  """Test structural conversion from PyTorch to Flax NNX."""
  engine = ASTEngine(source="torch", target="flax_nnx")
  res = engine.run(TORCH_CLASS_CODE)
  code = res.code

  assert "class MyModel(nnx.Module):" in code
  assert "rngs: nnx.Rngs" in code
  assert "def __init__(self, rngs: nnx.Rngs, in_features: int, out_features: int):" in code
  assert "super().__init__()" not in code
  assert "self.fc = nnx.Linear(in_features, out_features, rngs=rngs)" in code
  assert "def __call__(self, x):" in code
  assert "def forward" not in code


def test_flax_nnx_to_torch_structural() -> None:
  """Test structural conversion from Flax NNX to PyTorch."""
  engine = ASTEngine(source="flax_nnx", target="torch")
  res = engine.run(FLAX_NNX_CLASS_CODE)
  code = res.code

  assert "class MyModel(nn.Module):" in code
  assert "def __init__(self, in_features: int, out_features: int):" in code
  assert "rngs: nnx.Rngs" not in code
  assert "super().__init__()" in code
  assert "self.fc = nn.Linear(in_features, out_features)" in code
  assert "def forward(self, x):" in code
  assert "def __call__" not in code


def test_torch_to_mlx_structural() -> None:
  """Test structural conversion from PyTorch to Apple MLX."""
  engine = ASTEngine(source="torch", target="mlx")
  res = engine.run(TORCH_CLASS_CODE)
  code = res.code

  assert "class MyModel(nn.Module):" in code
  assert "super().__init__()" in code
  assert "self.fc = nn.Linear(in_features, out_features)" in code
  assert "def __call__(self, x):" in code
  assert "def forward" not in code


def test_mlx_to_torch_structural() -> None:
  """Test structural conversion from Apple MLX to PyTorch."""
  engine = ASTEngine(source="mlx", target="torch")
  res = engine.run(MLX_CLASS_CODE)
  code = res.code

  assert "class MyModel(nn.Module):" in code
  assert "super().__init__()" in code
  assert "self.fc = nn.Linear(in_features, out_features)" in code
  assert "def forward(self, x):" in code
  assert "def __call__" not in code


def test_torch_to_keras_structural() -> None:
  """Test structural conversion from PyTorch to Keras."""
  engine = ASTEngine(source="torch", target="keras")
  res = engine.run(TORCH_CLASS_CODE)
  code = res.code

  assert "class MyModel(keras.Layer):" in code or "class MyModel(keras.Model):" in code
  assert "super().__init__()" in code
  assert "self.fc = layers.Dense(in_features, out_features)" in code
  assert "def call(self, x):" in code
  assert "def forward" not in code


def test_keras_to_torch_structural() -> None:
  """Test structural conversion from Keras to PyTorch."""
  engine = ASTEngine(source="keras", target="torch")
  res = engine.run(KERAS_CLASS_CODE)
  code = res.code

  assert "class MyModel(nn.Module):" in code
  assert "super().__init__()" in code
  assert "self.fc = nn.Linear(in_features, out_features)" in code
  assert "def forward(self, x):" in code
  assert "def call" not in code


def test_flax_nnx_to_keras_structural() -> None:
  """Test structural conversion from Flax NNX to Keras."""
  engine = ASTEngine(source="flax_nnx", target="keras")
  res = engine.run(FLAX_NNX_CLASS_CODE)
  code = res.code

  assert "class MyModel(keras.Layer):" in code or "class MyModel(keras.Model):" in code
  assert "rngs: nnx.Rngs" not in code
  assert "super().__init__()" in code
  assert "self.fc = layers.Dense(in_features, out_features)" in code
  assert "def call(self, x):" in code
  assert "def __call__" not in code


def test_keras_to_flax_nnx_structural() -> None:
  """Test structural conversion from Keras to Flax NNX."""
  engine = ASTEngine(source="keras", target="flax_nnx")
  res = engine.run(KERAS_CLASS_CODE)
  code = res.code

  assert "class MyModel(nnx.Module):" in code
  assert "rngs: nnx.Rngs" in code
  assert "super().__init__()" not in code
  assert "self.fc = nnx.Linear(in_features, out_features, rngs=rngs)" in code
  assert "def __call__(self, x):" in code
  assert "def call" not in code


def test_mlx_to_flax_nnx_structural() -> None:
  """Test structural conversion from Apple MLX to Flax NNX."""
  engine = ASTEngine(source="mlx", target="flax_nnx")
  res = engine.run(MLX_CLASS_CODE)
  code = res.code

  assert "class MyModel(nnx.Module):" in code
  assert "rngs: nnx.Rngs" in code
  assert "super().__init__()" not in code
  assert "self.fc = nnx.Linear(in_features, out_features, rngs=rngs)" in code
  assert "def __call__(self, x):" in code


def test_flax_nnx_to_mlx_structural() -> None:
  """Test structural conversion from Flax NNX to Apple MLX."""
  engine = ASTEngine(source="flax_nnx", target="mlx")
  res = engine.run(FLAX_NNX_CLASS_CODE)
  code = res.code

  assert "class MyModel(nn.Module):" in code
  assert "rngs: nnx.Rngs" not in code
  assert "super().__init__()" in code
  assert "self.fc = nn.Linear(in_features, out_features)" in code
  assert "def __call__(self, x):" in code


def test_mlx_to_keras_structural() -> None:
  """Test structural conversion from Apple MLX to Keras."""
  engine = ASTEngine(source="mlx", target="keras")
  res = engine.run(MLX_CLASS_CODE)
  code = res.code

  assert "class MyModel(keras.Layer):" in code or "class MyModel(keras.Model):" in code
  assert "super().__init__()" in code
  assert "self.fc = layers.Dense(in_features, out_features)" in code
  assert "def call(self, x):" in code


def test_keras_to_mlx_structural() -> None:
  """Test structural conversion from Keras to Apple MLX."""
  engine = ASTEngine(source="keras", target="mlx")
  res = engine.run(KERAS_CLASS_CODE)
  code = res.code

  assert "class MyModel(nn.Module):" in code
  assert "super().__init__()" in code
  assert "self.fc = nn.Linear(in_features, out_features)" in code
  assert "def __call__(self, x):" in code
  assert "def call" not in code
