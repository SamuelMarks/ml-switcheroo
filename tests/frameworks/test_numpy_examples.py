"""Test suite for the Numpy Examples module."""

import ast

import pytest

from ml_switcheroo.frameworks.numpy import NumpyAdapter


@pytest.fixture
def adapter() -> NumpyAdapter:
  """Create a configured `NumpyAdapter` fixture for numpy examples tests.

  Returns:
      Configured `adapter` fixture value (NumpyAdapter).
  """
  return NumpyAdapter()


def test_numpy_examples_structure(adapter: NumpyAdapter) -> None:
  """Verify `adapter.get_tiered_examples` emits `tier1_math` for numpy examples structure.

  Args:
      adapter: FrameworkAdapter instance under test.
  """
  examples: dict[str, str] = adapter.get_tiered_examples()
  assert isinstance(examples, dict)
  assert "tier1_math" in examples
  assert "tier2_neural" in examples
  assert "tier3_extras" in examples


def test_tier1_math_validity(adapter: NumpyAdapter) -> None:
  """Verify `parse` emits `import numpy as np` for tier1 math validity.

  Args:
      adapter: FrameworkAdapter instance under test.
  """
  code: str = adapter.get_tiered_examples()["tier1_math"]
  ast.parse(code)
  assert "import numpy as np" in code
  assert "np.matmul" in code
  assert "np.abs" in code
  assert "np.linalg.norm" in code


def test_tier2_neural_validity(adapter: NumpyAdapter) -> None:
  """Verify `parse` emits `Out of Scope` for tier2 neural validity.

  Args:
      adapter: FrameworkAdapter instance under test.
  """
  code: str = adapter.get_tiered_examples()["tier2_neural"]
  ast.parse(code)
  assert "Out of Scope" in code
  assert "numpy" in code


def test_tier3_extras_validity(adapter: NumpyAdapter) -> None:
  """Verify `parse` emits `np.save` for tier3 extras validity.

  Args:
      adapter: FrameworkAdapter instance under test.
  """
  code: str = adapter.get_tiered_examples()["tier3_extras"]
  ast.parse(code)
  assert "np.save" in code
  assert "np.load" in code
  assert "file=" in code
