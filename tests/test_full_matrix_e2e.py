"""Comprehensive End-to-End Test Suite for All 30 Cross-Framework & Cross-ISA Paths.

Tests all permutations of static translation between:
- High-Level Frameworks: PyTorch, JAX, MLX, Keras (12 paths)
- High-Level to Hardware ISAs: PyTorch, JAX, MLX, Keras -> RDNA, NVIDIA SASS (8 paths)
- Hardware ISAs to High-Level: RDNA, NVIDIA SASS -> PyTorch, JAX, MLX, Keras (8 paths)
- Hardware ISA to Hardware ISA: RDNA <-> NVIDIA SASS (2 paths)
Total: 30 bidirectional paths.
"""

import ast
from typing import Dict
import pytest

from ml_switcheroo import convert

# Canonical snippets for high-level source frameworks
HIGH_LEVEL_SNIPPETS: Dict[str, str] = {
  "torch": """import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 32)

    def forward(self, x):
        return self.fc(x)
""",
  "jax": """from flax import nnx
import jax.numpy as jnp

class LinearModel(nn.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.fc = nnx.Linear(16, 32, rngs=rngs)

    def __call__(self, x):
        return self.fc(x)
""",
  "mlx": """import mlx.core as mx
import mlx.nn as nn

class LinearModel(mlx.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 32)

    def __call__(self, x):
        return self.fc(x)
""",
  "keras": """import keras
import tensorflow as tf

class LinearModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.fc = keras.layers.Dense(32)

    def call(self, x):
        return self.fc(x)
""",
}

# Canonical ConvNet snippets for high-level source frameworks
CONVNET_SNIPPETS: Dict[str, str] = {
  "torch": """import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)

    def forward(self, x):
        return self.conv(x)
""",
  "jax": """from flax import nnx
import jax.numpy as jnp

class ConvNet(nn.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.conv = nnx.Conv(3, 16, 3, rngs=rngs)

    def __call__(self, x):
        return self.conv(x)
""",
  "mlx": """import mlx.core as mx
import mlx.nn as nn

class ConvNet(mlx.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)

    def __call__(self, x):
        return self.conv(x)
""",
  "keras": """import keras
import tensorflow as tf

class ConvNet(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv = keras.layers.Conv2D(16, 3)

    def call(self, x):
        return self.conv(x)
""",
}

# Canonical assembly snippets for hardware ISAs
RDNA_SNIPPET = """
; BEGIN Conv2d (conv)
v_add_f32 v1, v0, v0
; END Conv2d (conv)
"""

SASS_SNIPPET = """
// BEGIN Conv2d (conv)
FADD R1, R0, R0;
// END Conv2d (conv)
"""

FRAMEWORKS = ["torch", "jax", "mlx", "keras"]
ISAS = ["rdna", "nvidia_sass"]


@pytest.mark.parametrize("source", FRAMEWORKS)
@pytest.mark.parametrize("target", FRAMEWORKS)
def test_high_level_matrix_e2e(source: str, target: str) -> None:
  """Test static CST translation between all high-level frameworks (12 paths).

  Args:
      source: Source framework identifier.
      target: Target framework identifier.
  """
  if source == target:
    pytest.skip("Self-translation skipped.")

  code = HIGH_LEVEL_SNIPPETS[source]
  converted = convert(code, source=source, target=target)

  # Verify valid AST syntax
  parsed = ast.parse(converted)
  assert parsed is not None
  assert "class LinearModel" in converted

  # Verify target-specific structural requirements
  if target == "torch":
    assert "nn.Module" in converted
    assert "def forward(self" in converted
  elif target == "jax":
    assert "def __call__(self" in converted
  elif target == "mlx":
    assert "mlx.nn.Module" in converted or "nn.Module" in converted
    assert "def __call__(self" in converted
  elif target == "keras":
    assert "keras.Model" in converted or "keras.Layer" in converted
    assert "def call(self" in converted


@pytest.mark.parametrize("source", FRAMEWORKS)
@pytest.mark.parametrize("target", FRAMEWORKS)
def test_convnet_matrix_e2e(source: str, target: str) -> None:
  """Test static CST translation of convolutional neural networks between high-level frameworks (12 paths).

  Args:
      source: Source framework identifier.
      target: Target framework identifier.
  """
  if source == target:
    pytest.skip("Self-translation skipped.")

  code = CONVNET_SNIPPETS[source]
  converted = convert(code, source=source, target=target)

  parsed = ast.parse(converted)
  assert parsed is not None
  assert "class ConvNet" in converted

  if target == "torch":
    assert "nn.Module" in converted
    assert "def forward(self" in converted
  elif target == "jax":
    assert "def __call__(self" in converted
  elif target == "mlx":
    assert "mlx.nn.Module" in converted or "nn.Module" in converted
    assert "def __call__(self" in converted
  elif target == "keras":
    assert "keras.Model" in converted or "keras.Layer" in converted
    assert "def call(self" in converted


@pytest.mark.parametrize("source", FRAMEWORKS)
@pytest.mark.parametrize("target", ISAS)
def test_high_level_to_hardware_isa_e2e(source: str, target: str) -> None:
  """Test lowering high-level models to hardware ISA assembly (8 paths).

  Args:
      source: Source high-level framework identifier.
      target: Target hardware ISA identifier.
  """
  code = HIGH_LEVEL_SNIPPETS[source]
  converted = convert(code, source=source, target=target)

  assert isinstance(converted, str)
  if target == "rdna":
    assert "; BEGIN" in converted or "v_" in converted
  elif target == "nvidia_sass":
    assert "// BEGIN" in converted or "MOV" in converted or "FADD" in converted or "FFMA" in converted


@pytest.mark.parametrize("source", ISAS)
@pytest.mark.parametrize("target", FRAMEWORKS)
def test_hardware_isa_to_high_level_e2e(source: str, target: str) -> None:
  """Test lifting hardware ISA assembly to high-level framework models (8 paths).

  Args:
      source: Source hardware ISA identifier.
      target: Target high-level framework identifier.
  """
  code = RDNA_SNIPPET if source == "rdna" else SASS_SNIPPET
  converted = convert(code, source=source, target=target)

  parsed = ast.parse(converted)
  assert parsed is not None
  assert "class " in converted

  if target == "torch":
    assert "nn.Module" in converted
    assert "def forward(self" in converted
  elif target == "jax":
    assert "def __call__(self" in converted
  elif target == "mlx":
    assert "mlx.nn.Module" in converted
    assert "def __call__(self" in converted
  elif target == "keras":
    assert "keras.Model" in converted or "keras.Layer" in converted
    assert "def call(self" in converted


def test_rdna_to_sass_e2e() -> None:
  """Test cross-ISA transpilation from AMD RDNA to NVIDIA SASS."""
  converted = convert(RDNA_SNIPPET, source="rdna", target="nvidia_sass")
  assert "Conv2d" in converted
  assert "// BEGIN" in converted or "FFMA" in converted or "MOV" in converted


def test_sass_to_rdna_e2e() -> None:
  """Test cross-ISA transpilation from NVIDIA SASS to AMD RDNA."""
  converted = convert(SASS_SNIPPET, source="nvidia_sass", target="rdna")
  assert "Conv2d" in converted
  assert "; BEGIN" in converted or "v_fmac" in converted or "v_mov" in converted
