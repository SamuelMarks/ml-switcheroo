"""
Integration Tests for EX03: Array Manipulation (Plugins).

Validates:
1. `torch.permute` (Varags) -> `jax.numpy.transpose` (Tuple) via `pack_varargs` plugin.
2. `torch.permute` -> `tensorflow.transpose`.
3. `torch.permute` -> `numpy.transpose`.
"""

import ast
import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.utils.ast_utils import cmp_ast

SOURCE_TORCH = """
import torch

def transpose_matrices(batch):
    return torch.permute(batch, 0, 2, 1)
"""

# JAX (Uses pack_varargs plugin defined in semantics)
EXPECTED_JAX = """
import jax.numpy as jnp

def transpose_matrices(batch):
    return jnp.transpose(batch, axes=(0, 2, 1))
"""

# TensorFlow (also uses transpose with perm/axes tuple)
EXPECTED_TENSORFLOW = """
import tensorflow as tf

def transpose_matrices(batch):
    return tf.transpose(batch, perm=(0, 2, 1))
"""

# NumPy
EXPECTED_NUMPY = """
import numpy as np

def transpose_matrices(batch):
    return np.transpose(batch, axes=(0, 2, 1))
"""


@pytest.fixture(scope="module")
def semantics():
  return SemanticsManager()


@pytest.mark.parametrize(
  "target_fw, expected_code",
  [
    ("jax", EXPECTED_JAX),
    ("tensorflow", EXPECTED_TENSORFLOW),
    ("numpy", EXPECTED_NUMPY),
  ],
)
def test_ex03_permute_plugin(semantics, target_fw, expected_code):
  config = RuntimeConfig(source_framework="torch", target_framework=target_fw, strict_mode=True)
  engine = ASTEngine(semantics=semantics, config=config)
  result = engine.run(SOURCE_TORCH)

  assert result.success, f"Errors: {result.errors}"

  try:
    assert cmp_ast(ast.parse(result.code), ast.parse(expected_code))
  except AssertionError:
    print(f"\n--- Expected ({target_fw}) ---\n{expected_code}")
    print(f"\n--- Actual ({target_fw}) ---\n{result.code}")
    raise
