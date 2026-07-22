"""Test suite for the Linter module."""

import pytest
from ml_switcheroo.testing.linter import StructuralLinter, validate_transpilation
from unittest.mock import patch, MagicMock


@pytest.fixture
def linter():
  """Provides a mock linter for testing."""
  return StructuralLinter(forbidden_roots={"torch", "flax"})


def test_linter_clean_code(linter):
  """Verifies the behavior of linter clean code."""
  code = "\nimport jax.numpy as jnp\ndef f(x):\n    return jnp.abs(x)\n"
  errors = linter.check(code)
  assert len(errors) == 0


def test_linter_detects_import(linter):
  """Verifies the behavior of linter detects import."""
  code = "\nimport torch\nx = torch.abs(y)\n"
  errors = linter.check(code)
  assert len(errors) > 0
  assert "Forbidden Import: 'torch'" in errors[0]


def test_linter_detects_from_import(linter):
  """Verifies the behavior of linter detects from import."""
  code = "from flax import linen as nn"
  errors = linter.check(code)
  assert len(errors) > 0
  assert "Forbidden Import: 'from flax ...'" in errors[0]


def test_linter_detects_aliased_usage(linter):
  """Verifies the behavior of linter detects aliased usage."""
  code = "\nimport torch as t\n# Usage of alias\ny = t.abs(x)\n"
  errors = linter.check(code)
  assert len(errors) >= 1
  usage_errors = [e for e in errors if "Forbidden Usage" in e]
  assert len(usage_errors) > 0
  assert "alias of torch" in usage_errors[0]


def test_facade_flax_inheritance():
  """Verifies the behavior of facade Flax inheritance."""
  mock_adapter = MagicMock()
  mock_adapter.import_alias = ("flax.nnx", "nnx")
  mock_adapter.inherits_from = "jax"
  with patch("ml_switcheroo.testing.linter.get_adapter", return_value=mock_adapter):
    code = "import jax.numpy as jnp"
    (is_valid, errors) = validate_transpilation(code, source_fw="flax_nnx")
    assert not is_valid
    assert "Forbidden Import: 'jax'" in errors[0]


def test_facade_mlx_detection(tmp_path):
  """Verifies the behavior of facade MLX detection."""
  mock_adapter = MagicMock()
  mock_adapter.import_alias = ("mlx.core", "mx")
  mock_adapter.search_modules = ["mlx"]
  mock_adapter.inherits_from = None
  with patch("ml_switcheroo.testing.linter.get_adapter", return_value=mock_adapter):
    code = "\nimport mlx.core as mx\ndef f(x):\n    return mx.abs(x)\n"
    (is_valid, errors) = validate_transpilation(code, source_fw="mlx")
    assert not is_valid
    assert any(("mlx" in e for e in errors))
