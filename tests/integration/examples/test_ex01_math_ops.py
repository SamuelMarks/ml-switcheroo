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
import numpy as np

def compute_loss(prediction, target): 
    diff = keras.ops.abs(prediction - target) 
    loss = keras.ops.mean(diff) 
    return loss
"""


@pytest.fixture(scope="module")
def semantics():
  """
  Provides a SemanticsManager with explicitly defined Math operations.
  """
  mgr = SemanticsManager()

  # Inject manual definitions to ensure tests pass even if JSONs are corrupt/missing

  # 1. Abs
  abs_def = {
    "std_args": ["x"],
    "variants": {
      "torch": {"api": "torch.abs"},
      "jax": {"api": "jax.numpy.abs"},
      "numpy": {"api": "numpy.abs"},
      "tensorflow": {"api": "tf.abs"},
      "mlx": {"api": "mlx.core.abs"},
      "keras": {"api": "keras.ops.abs"},
    },
  }

  # 2. Mean
  mean_def = {
    "std_args": ["x"],
    "variants": {
      "torch": {"api": "torch.mean"},
      "jax": {"api": "jax.numpy.mean"},
      "numpy": {"api": "numpy.mean"},
      "tensorflow": {"api": "tf.math.reduce_mean"},
      "mlx": {"api": "mlx.core.mean"},
      "keras": {"api": "keras.ops.mean"},
    },
  }

  mgr.update_definition("Abs", abs_def)
  mgr.update_definition("Mean", mean_def)

  return mgr


@pytest.mark.parametrize(
  "target_fw, expected_string",
  [
    ("jax", "jnp.abs"),
    ("numpy", "np.abs"),
    ("tensorflow", "tf.abs"),
    ("mlx", "mx.abs"),
    ("keras", "keras.ops.abs"),
  ],
)
def test_ex01_math_transpilation(semantics, target_fw, expected_string):
  """
  Verifies that basic math operations are correctly mapped to all supported backends.
  """
  config = RuntimeConfig(source_framework="torch", target_framework=target_fw, strict_mode=True)
  engine = ASTEngine(semantics=semantics, config=config)

  result = engine.run(SOURCE_TORCH)

  assert result.success, f"Failed converting to {target_fw}: {result.errors}"

  # Structural checks
  assert expected_string in result.code
  assert "compute_loss" in result.code
  assert "prediction - target" in result.code

  # Check imports
  if target_fw == "jax":
    assert "import jax.numpy as jnp" in result.code
  elif target_fw == "numpy":
    assert "import numpy as np" in result.code
  elif target_fw == "mlx":
    assert "import mlx.core as mx" in result.code
