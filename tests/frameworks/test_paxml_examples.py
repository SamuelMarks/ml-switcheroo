"""Test suite for the Paxml Examples module."""

import ast

import pytest

from ml_switcheroo.frameworks.paxml import PaxmlAdapter


@pytest.fixture
def adapter() -> PaxmlAdapter:
  """Create a configured `PaxmlAdapter` fixture for PaxML examples tests.

  Returns:
      Configured `adapter` fixture value (PaxmlAdapter).
  """
  return PaxmlAdapter()


def test_paxml_examples_structure(adapter: PaxmlAdapter) -> None:
  """Verify `adapter.get_tiered_examples` emits `tier1_math` for PaxML examples structure.

  Args:
      adapter: FrameworkAdapter instance under test.
  """
  examples: dict[str, str] = adapter.get_tiered_examples()
  assert isinstance(examples, dict)
  assert "tier1_math" in examples
  assert "tier2_neural" in examples
  assert "tier3_extras" in examples


def test_tier1_math_validity(adapter: PaxmlAdapter) -> None:
  """Verify `parse` emits `import jax.numpy as jnp` for tier1 math validity.

  Args:
      adapter: FrameworkAdapter instance under test.
  """
  code: str = adapter.get_tiered_examples()["tier1_math"]
  ast.parse(code)
  assert "import jax.numpy as jnp" in code


def test_tier2_neural_validity(adapter: PaxmlAdapter) -> None:
  """Verify `parse` emits `def setup(self):` for tier2 neural validity.

  Args:
      adapter: FrameworkAdapter instance under test.
  """
  code: str = adapter.get_tiered_examples()["tier2_neural"]
  ast.parse(code)
  assert "class SimpleMLP(base_layer.BaseLayer):" in code
  assert "def setup(self):" in code
  assert "pl.Linear" in code


def test_tier3_extras_validity(adapter: PaxmlAdapter) -> None:
  """Verify `parse` emits `pax_fiddle.Config` for tier3 extras validity.

  Args:
      adapter: FrameworkAdapter instance under test.
  """
  code: str = adapter.get_tiered_examples()["tier3_extras"]
  ast.parse(code)
  assert "pax_fiddle.Config" in code
  assert "input_dims" in code
