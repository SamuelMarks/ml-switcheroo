"""
Tests for Shape Calculation Verification in Generator.

Verifies that:
1. If `output_shape_calc` is present in ODL, generator emits shape verification logic.
2. The logic correctly calls the lambda with input arguments.
3. Assertions are generated.
"""

import pytest
from unittest.mock import MagicMock
from ml_switcheroo.generated_tests.generator import TestGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


class MockShapeSemantics(SemanticsManager):
  """
  Mock Manager providing templates.
  """

  def __init__(self):
    # Minimal template
    self.templates = {
      "torch": {
        "import": "import torch",
        "convert_input": "torch.tensor({np_var})",
        "to_numpy": "{res_var}.numpy()",
      },
      "jax": {
        "import": "import jax",
        "convert_input": "jax.numpy.array({np_var})",
        "to_numpy": "numpy.array({res_var})",
      },
    }

  def get_test_template(self, framework):
    return self.templates.get(framework)

  def get_framework_config(self, framework):
    return {}


@pytest.fixture
def generator():
  mgr = MockShapeSemantics()
  return TestGenerator(semantics_mgr=mgr)


def test_generator_injects_shape_logic(generator, tmp_path):
  """
  Scenario: Op 'transpose' has output_shape_calc = 'lambda x: (x.shape[1], x.shape[0])'.
  Expectation: Generator emits assertions.
  """
  semantics = {
    "transpose": {
      "std_args": ["x"],
      "output_shape_calc": "lambda x: (x.shape[1], x.shape[0])",
      "variants": {
        "torch": {"api": "torch.t"},
        "jax": {"api": "jnp.transpose"},
      },
    }
  }

  out_file = tmp_path / "test_shape.py"
  generator.generate(semantics, out_file)
  content = out_file.read_text()

  # Verify injection of shape calc
  assert "shape_fn = lambda x: (x.shape[1], x.shape[0])" in content
  # Verify input args passed
  assert "expected_shape = shape_fn(np_x)" in content
  # Verify assertion loop
  assert "assert val.shape == expected_shape" in content


def test_generator_no_shape_logic_if_missing(generator, tmp_path):
  """
  Scenario: Op has no output_shape_calc.
  Expectation: No validation block provided.
  """
  semantics = {
    "abs": {
      "std_args": ["x"],
      # No shape calc
      "variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jnp.abs"}},
    }
  }

  out_file = tmp_path / "test_no_shape.py"
  generator.generate(semantics, out_file)
  content = out_file.read_text()

  assert "expected_shape =" not in content


def test_generator_multi_arg_shape(generator, tmp_path):
  """
  Scenario: Op `matmul(x, y)` has shape calc `lambda x, y: (x.shape[0], y.shape[1])`.
  Expectation: Generator calls `shape_fn(np_x, np_y)`.
  """
  semantics = {
    "matmul": {
      "std_args": ["x", "y"],
      "output_shape_calc": "lambda x, y: (x.shape[0], y.shape[1])",
      "variants": {"torch": {"api": "torch.mm"}, "jax": {"api": "jnp.matmul"}},
    }
  }

  out_file = tmp_path / "test_multi_arg.py"
  generator.generate(semantics, out_file)
  content = out_file.read_text()

  assert "shape_fn = lambda x, y: (x.shape[0], y.shape[1])" in content
  # Verify argument string construction from list
  assert "expected_shape = shape_fn(np_x, np_y)" in content
