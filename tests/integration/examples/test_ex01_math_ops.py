"""
Integration Tests for EX01: Math Ops (Tier 1).

Source: PyTorch
Targets: JAX, TensorFlow, NumPy, MLX, Keras
"""

import ast
import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.utils.ast_utils import cmp_ast

# --- Source Code ---
# Derived from tests/examples/ex01_math_ops.torch.py
SOURCE_TORCH = """
import torch

def compute_loss(prediction, target):
    diff = torch.abs(prediction - target)
    loss = torch.mean(diff)
    return loss
"""

# --- Expected Outputs ---

EXPECTED_JAX = """
import jax.numpy as jnp

def compute_loss(prediction, target):
    diff = jnp.abs(prediction - target)
    loss = jnp.mean(diff)
    return loss
"""

EXPECTED_NUMPY = """
import numpy as np

def compute_loss(prediction, target):
    diff = np.abs(prediction - target)
    loss = np.mean(diff)
    return loss
"""

EXPECTED_TENSORFLOW = """
import tensorflow as tf

def compute_loss(prediction, target):
    diff = tf.abs(prediction - target)
    loss = tf.math.reduce_mean(diff)
    return loss
"""

EXPECTED_MLX = """
import mlx.core as mx

def compute_loss(prediction, target):
    diff = mx.abs(prediction - target)
    loss = mx.mean(diff)
    return loss
"""

EXPECTED_KERAS = """
import keras

def compute_loss(prediction, target):
    diff = keras.ops.abs(prediction - target)
    loss = keras.ops.mean(diff)
    return loss
"""


@pytest.fixture(scope="module")
def semantics():
  return SemanticsManager()


@pytest.mark.parametrize(
  "target_fw, expected_code",
  [
    ("jax", EXPECTED_JAX),
    ("numpy", EXPECTED_NUMPY),
    ("tensorflow", EXPECTED_TENSORFLOW),
    ("mlx", EXPECTED_MLX),
    ("keras", EXPECTED_KERAS),
  ],
)
def test_ex01_math_transpilation(semantics, target_fw, expected_code):
  """
  Verifies that basic math operations are correctly mapped to all supported backends.
  """
  config = RuntimeConfig(source_framework="torch", target_framework=target_fw, strict_mode=True)
  engine = ASTEngine(semantics=semantics, config=config)

  result = engine.run(SOURCE_TORCH)

  assert result.success, f"Failed converting to {target_fw}: {result.errors}"

  # Normalize whitespaces for comparison
  generated_ast = ast.parse(result.code)
  expected_ast = ast.parse(expected_code)

  try:
    assert cmp_ast(generated_ast, expected_ast)
  except AssertionError:
    print(f"\n--- Expected ({target_fw}) ---\n{expected_code}")
    print(f"\n--- Actual ({target_fw}) ---\n{result.code}")
    raise
