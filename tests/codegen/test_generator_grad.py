"""
Tests for Gradient Verification Logic in generator.

Verifies:
1. Logic injection for differentiable operations.
2. Suppression for non-differentiable operations.
3. Usage of default grad templates.
"""

import pytest
from unittest.mock import MagicMock
from ml_switcheroo.generated_tests.generator import TestCaseGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def generator(tmp_path):
  mgr = MagicMock(spec=SemanticsManager)
  # Provide standard templates (without grad keys implies checking defaults)
  templates = {
    "torch": {"import": "import torch", "convert_input": "torch.tensor({np_var})", "to_numpy": "{res_var}.numpy()"},
    "jax": {"import": "import jax", "convert_input": "jnp.array({np_var})", "to_numpy": "{res_var}"},
  }
  mgr.get_test_template.side_effect = lambda fw: templates.get(fw)
  mgr.test_templates = templates
  mgr.get_framework_config.return_value = {}
  return TestCaseGenerator(semantics_mgr=mgr)


def test_grad_injection_enabled(generator, tmp_path):
  """
  Scenario: Differentiable operation (default).
  Expect: results_grad dict and grad checking code.
  """
  semantics = {"sin": {"std_args": ["x"], "variants": {"torch": {"api": "torch.sin"}, "jax": {"api": "jnp.sin"}}}}

  out_file = tmp_path / "test_sin.py"
  generator.generate(semantics, out_file)
  content = out_file.read_text()

  # Check JAX Grad
  assert "jax.grad(lambda a0: jnp.sum(jnp.sin(a0)))(jnp.array(np_x))" in content

  # Check Torch Grad
  assert "torch.func.grad(lambda a0: torch.sum(torch.sin(a0)))(torch.tensor(np_x))" in content

  # Check Comparison
  assert "Gradient Verification" in content
  assert "np.testing.assert_allclose(g_ref, g_val" in content


def test_grad_injection_disabled_flag(generator, tmp_path):
  """
  Scenario: differentiable=False in spec.
  Expect: No gradient code.
  """
  semantics = {
    "argmax": {
      "std_args": ["x"],
      "differentiable": False,
      "variants": {"torch": {"api": "torch.argmax"}, "jax": {"api": "jnp.argmax"}},
    }
  }

  out_file = tmp_path / "test_nodiff.py"
  generator.generate(semantics, out_file)
  content = out_file.read_text()

  assert "jax.grad" not in content
  assert "Gradient Verification" not in content


def test_grad_injection_disabled_primitive(generator, tmp_path):
  """
  Scenario: Input is integer (not Array/Tensor).
  Expect: Skip gradient check because arg0 is not differentiable.
  """
  # Note: Generator _is_primitive("int") returns True -> skips
  semantics = {
    "factorial": {
      "std_args": [{"name": "n", "type": "int"}],
      "variants": {"torch": {"api": "torch.math"}, "jax": {"api": "jax.math"}},
    }
  }

  out_file = tmp_path / "test_int_input.py"
  generator.generate(semantics, out_file)
  content = out_file.read_text()

  assert "jax.grad" not in content


def test_grad_multi_arg(generator, tmp_path):
  """
  Scenario: Binary Op.
  Expect: Lambda signature 'lambda a0, a1: ...'
  """
  semantics = {"add": {"std_args": ["x", "y"], "variants": {"torch": {"api": "torch.add"}, "jax": {"api": "jnp.add"}}}}
  out_file = tmp_path / "test_multi.py"
  generator.generate(semantics, out_file)

  content = out_file.read_text()
  # Check JAX structure
  # lambda with 2 args
  assert "lambda a0, a1:" in content
  # Check call
  assert "jnp.add(a0, a1)" in content
