"""
Tests for Void Return Logic in Test Generator.

Verifies that:
1. Operations with `return_type="None"` skip value assertion.
2. Execution success (try/except safe) is considered passing.
3. Code generation structure omits verify_results call.
"""

import pytest
from unittest.mock import MagicMock
from ml_switcheroo.generated_tests.generator import TestGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def generator(tmp_path):
  mgr = MagicMock(spec=SemanticsManager)

  # FIX: Ensure Templates are returned (needed for generation logic)
  mock_templates = {
    "torch": {"import": "import torch", "convert_input": "{np_var}", "to_numpy": "{res_var}"},
    # We need at least one other framework to avoid "Not enough backends" logic check,
    # though generate() loops over keys in variants map.
    # Let's add 'jax' to ensuring variants loop runs.
    "jax": {"import": "import jax", "convert_input": "{np_var}", "to_numpy": "{res_var}"},
  }

  mgr.get_test_template.side_effect = lambda fw: mock_templates.get(fw)
  # Populate attributes for runtime gen
  mgr.test_templates = mock_templates
  mgr.get_framework_config.return_value = {}
  return TestGenerator(semantics_mgr=mgr)


def test_void_return_logic(generator, tmp_path):
  """
  Scenario: Operation 'Save' has return_type="None".
  Expectation: No `verify_results` call in generated code.
  """
  semantics = {
    "Save": {
      "std_args": ["obj"],
      "return_type": "None",
      "variants": {
        "torch": {"api": "torch.save"},
        "jax": {"api": "jax.save"},  # Need 2 variants for comparison logic block to be valid
      },
    }
  }

  out_file = tmp_path / "test_save.py"
  generator.generate(semantics, out_file)

  content = out_file.read_text()

  # Check logic
  assert "Operation expected to return None / Void" in content
  assert "verify_results" not in content
  assert "assert len(results) >= 2" in content

  # Ensure normal structure exists
  assert "try:" in content
  assert "results['torch'] =" in content


def test_standard_return_logic(generator, tmp_path):
  """
  Scenario: Standard Op (default return type).
  Expectation: Verification logic IS present.
  """
  semantics = {
    "Add": {
      "std_args": ["x"],
      "variants": {"torch": {"api": "torch.add"}, "jax": {"api": "jax.add"}},
    }
  }

  out_file = tmp_path / "test_add.py"
  generator.generate(semantics, out_file)

  content = out_file.read_text()
  assert "verify_results(ref, val" in content
