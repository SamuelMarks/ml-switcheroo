"""
Tests for Structural Linter (Forbidden Artifact Detection).

Verifies:
1. Detection of residual imports.
2. Detection of aliased usage.
3. Handling of Multi-part frameworks (Flax/JAX).
4. Validation facade logic.
"""

import pytest
from ml_switcheroo.testing.linter import StructuralLinter, validate_transpilation
from unittest.mock import patch, MagicMock


@pytest.fixture
def linter():
  # Ban 'torch' and 'flax'
  return StructuralLinter(forbidden_roots={"torch", "flax"})


def test_linter_clean_code(linter):
  """Scenario: Valid JAX code, no Torch artifacts."""
  code = """
import jax.numpy as jnp
def f(x):
    return jnp.abs(x)
"""
  errors = linter.check(code)
  assert len(errors) == 0


def test_linter_detects_import(linter):
  """Scenario: 'import torch' persists."""
  code = """
import torch
x = torch.abs(y)
"""
  errors = linter.check(code)
  assert len(errors) > 0
  assert "Forbidden Import: 'torch'" in errors[0]


def test_linter_detects_from_import(linter):
  """Scenario: 'from flax import linen' persists."""
  code = "from flax import linen as nn"
  errors = linter.check(code)
  assert len(errors) > 0
  assert "Forbidden Import: 'from flax ...'" in errors[0]


def test_linter_detects_aliased_usage(linter):
  """
  Scenario: 'import torch as t; t.abs()'.
  Verifies that usage of 't' triggers violation because 't' aliases 'torch'.
  """
  code = """
import torch as t
# Usage of alias
y = t.abs(x)
"""
  # Linter visits Import, logs violation, records alias 't'.
  # Visits Attribute 't.abs', logs usage violation.
  errors = linter.check(code)

  assert len(errors) >= 1
  # We check if usage message works
  usage_errors = [e for e in errors if "Forbidden Usage" in e]
  assert len(usage_errors) > 0
  assert "alias of torch" in usage_errors[0]


def test_facade_flax_inheritance():
  """
  Verify that validate_transpilation bans Parent frameworks too.
  If source='flax_nnx' (inherits jax), output containing 'import jax' should fail.
  """
  # Mock Adapter structure
  mock_adapter = MagicMock()
  mock_adapter.import_alias = ("flax.nnx", "nnx")
  mock_adapter.inherits_from = "jax"

  # We need to test the logic expanding forbidden sets
  with patch("ml_switcheroo.testing.linter.get_adapter", return_value=mock_adapter):
    code = "import jax.numpy as jnp"  # Should be forbidden if moving AWAY from Flax/Jax

    is_valid, errors = validate_transpilation(code, source_fw="flax_nnx")

    assert not is_valid
    assert "Forbidden Import: 'jax'" in errors[0]


def test_facade_mlx_detection(tmp_path):
  """
  Verify detection of the user's specific MLX No-Op bug.
  Code: import mlx.core as mx ... mx.abs(x)
  Target: JAX (Source: MLX)
  """
  # Mock MLX Adapter
  mock_adapter = MagicMock()
  mock_adapter.import_alias = ("mlx.core", "mx")
  mock_adapter.search_modules = ["mlx"]
  mock_adapter.inherits_from = None

  with patch("ml_switcheroo.testing.linter.get_adapter", return_value=mock_adapter):
    code = """
import mlx.core as mx
def f(x):
    return mx.abs(x)
"""
    is_valid, errors = validate_transpilation(code, source_fw="mlx")

    assert not is_valid
    assert any("mlx" in e for e in errors)
