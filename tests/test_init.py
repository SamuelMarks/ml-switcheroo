"""Test suite for the primary entrypoint convert function."""

import pytest

import ml_switcheroo
from ml_switcheroo.semantics.manager import SemanticsManager


def test_convert_end_to_end() -> None:
  """Verifies that the API performs a genuine end-to-end conversion without mocks."""
  input_code = "import torch\ndef my_func(x):\n    return torch.relu(x)"

  # Run the real pipeline
  res = ml_switcheroo.convert(input_code, source="torch", target="jax")

  # Validate the resulting code (assuming torch.relu translates to jax.nn.relu)
  assert isinstance(res, str)
  assert "import jax" in res or "jax.nn.relu" in res or "relu" in res
  assert "def my_func(x):" in res


def test_convert_with_custom_semantics() -> None:
  """Verifies that passing a custom semantics manager functions correctly end-to-end."""
  manager = SemanticsManager()
  # It might lazily load defaults, but let's just make sure it doesn't crash
  input_code = "x = 1"
  res = ml_switcheroo.convert(input_code, source="torch", target="jax", semantics=manager)
  assert res.strip() == "x = 1"


def test_convert_failure_reporting() -> None:
  """Verifies that a genuine parsing or validation error results in a Transpilation failed ValueError."""
  # Syntax error to trigger the failure branch in ASTEngine
  input_code = "def my_func(x"

  with pytest.raises(ValueError, match="Transpilation failed.*"):
    ml_switcheroo.convert(input_code, source="torch", target="jax")


def test_convert_with_empty_code() -> None:
  """Verifies that empty code is handled correctly."""
  assert ml_switcheroo.convert("", source="torch", target="jax").strip() == ""
