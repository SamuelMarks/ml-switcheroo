"""
Tests for Type-Aware Test Generation logic.
"""

import pytest
from unittest.mock import MagicMock
from ml_switcheroo.generated_tests.generator import TestGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def gen(tmp_path):
  """Fixture with a generator pointing to temp path."""
  mgr = MagicMock(spec=SemanticsManager)

  # Mock templates for both torch and jax to allow multi-backend generation tests
  def mock_get_template(fw):
    if fw == "torch":
      return {
        "import": "import torch",
        "convert_input": "torch.tensor({np_var})",
        "to_numpy": "{res_var}.numpy()",
      }
    if fw == "jax":
      return {
        "import": "import jax",
        "convert_input": "jnp.array({np_var})",
        "to_numpy": "np.array({res_var})",
      }
    return None

  mgr.get_test_template.side_effect = mock_get_template

  return TestGenerator(semantics_mgr=mgr)


def test_code_gen_str_int(gen):
  """
  Scenario: Type hint "int".
  Expectation: `random.randint(...)`
  """
  code = gen._generate_value_code("dim", "int")
  assert "random.randint" in code


def test_code_gen_str_bool(gen):
  """
  Scenario: Type hint "bool".
  Expectation: `bool(random.getrandbits(1))`
  """
  code = gen._generate_value_code("keepdims", "bool")
  assert "bool(random.getrandbits(1))" in code


def test_code_gen_str_float(gen):
  """
  Scenario: Type hint "float".
  Expectation: `random.uniform`
  """
  code = gen._generate_value_code("alpha", "float")
  assert "random.uniform" in code


def test_code_gen_str_array(gen):
  """
  Scenario: Type hint "Array" or "Tensor".
  Expectation: `np.random.randn`
  """
  code1 = gen._generate_value_code("x", "Array")
  assert "np.random.randn" in code1

  code2 = gen._generate_value_code("x", "Tensor")
  assert "np.random.randn" in code2


def test_code_gen_complex_list(gen):
  """
  Scenario: Type hint "List[int]".
  Expectation: `[1, 2]` stub.
  """
  code = gen._generate_value_code("pads", "List[int]")
  assert "[1, 2]" in code


def test_code_gen_heuristic_fallback(gen):
  """
  Scenario: Type hint "Any" or None.
  Expectation: Name-based heuristic (e.g. 'axis' -> '1').
  """
  # Name 'axis' -> heuristic returns 1
  code_axis = gen._generate_value_code("axis", "Any")
  assert code_axis == "1"

  # Name 'x' -> defaults to array
  code_x = gen._generate_value_code("x", "Any")
  assert "np.random.randn" in code_x


def test_generate_integration_typed_args(gen, tmp_path):
  """
  Integration test ensuring `generate` parses the `std_args` properly
  and emits the typed code into the file.
  """
  # Mock Semantics with typed arguments
  # Must provide at least 2 variants (torch, jax) for the generator to run
  semantics = {
    "randint_op": {
      # Tuple format: (name, type)
      "std_args": [("low", "int"), ("high", "int"), ("shape", "Tuple[int]")],
      "variants": {"torch": {"api": "torch.randint"}, "jax": {"api": "jax.random.randint"}},
    }
  }

  out_file = tmp_path / "test_typed.py"
  gen.generate(semantics, out_file)

  content = out_file.read_text()

  # Check that imports include random
  assert "import random" in content

  # Check input generation
  # Since types are int, we expect random.randint lines
  assert "np_low = random.randint" in content
  assert "np_high = random.randint" in content

  # Check Tuple generation
  assert "np_shape = (1, 2)" in content

  # Ensure no naive array generation happened for these inputs
  assert "np_low = np.random.randn" not in content
