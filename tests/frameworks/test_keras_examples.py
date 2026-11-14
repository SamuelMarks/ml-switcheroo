"""Test suite for the Keras Examples module."""

import ast

import pytest

from ml_switcheroo.frameworks.keras import KerasAdapter


@pytest.fixture
def adapter() -> KerasAdapter:
  """Create a configured `KerasAdapter` fixture for keras examples tests.

  Returns:
      Configured `adapter` fixture value (KerasAdapter).
  """
  return KerasAdapter()


def test_keras_examples_structure(adapter: KerasAdapter) -> None:
  """Verify `adapter.get_tiered_examples` emits `tier1_math` for keras examples structure.

  Args:
      adapter: FrameworkAdapter instance under test.
  """
  examples: dict[str, str] = adapter.get_tiered_examples()
  assert isinstance(examples, dict)
  assert "tier1_math" in examples
  assert "tier2_neural_sequential" in examples
  assert "tier3_extras_rng" in examples


def test_tier1_math_validity(adapter: KerasAdapter) -> None:
  """Verify `parse` emits `import keras` for tier1 math validity.

  Args:
      adapter: FrameworkAdapter instance under test.
  """
  code: str = adapter.get_tiered_examples()["tier1_math"]
  ast.parse(code)
  assert "import keras" in code
  assert "keras.ops.abs" in code


def test_tier2_neural_validity(adapter: KerasAdapter) -> None:
  """Verify `parse` emits `keras.Sequential` for tier2 neural validity.

  Args:
      adapter: FrameworkAdapter instance under test.
  """
  code: str = adapter.get_tiered_examples()["tier2_neural_sequential"]
  ast.parse(code)
  assert "keras.Sequential" in code


def test_tier3_extras_validity(adapter: KerasAdapter) -> None:
  """Verify `parse` emits `random.SeedGenerator` for tier3 extras validity.

  Args:
      adapter: FrameworkAdapter instance under test.
  """
  code: str = adapter.get_tiered_examples()["tier3_extras_rng"]
  ast.parse(code)
  assert "random.SeedGenerator" in code
