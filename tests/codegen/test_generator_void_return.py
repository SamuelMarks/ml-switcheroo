"""Test suite for the Generator Void Return module."""

import pytest
from unittest.mock import MagicMock
from ml_switcheroo.generated_tests.generator import TestCaseGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def generator(tmp_path):
  """Provides a mock generator for testing."""
  mgr = MagicMock(spec=SemanticsManager)
  mock_templates = {
    "torch": {"import": "import torch", "convert_input": "{np_var}", "to_numpy": "{res_var}"},
    "jax": {"import": "import jax", "convert_input": "{np_var}", "to_numpy": "{res_var}"},
  }
  mgr.get_test_template.side_effect = lambda fw: mock_templates.get(fw)
  mgr.test_templates = mock_templates
  mgr.get_framework_config.return_value = {}
  return TestCaseGenerator(semantics_mgr=mgr)


def test_void_return_logic(generator, tmp_path):
  """Verifies the behavior of void return logic."""
  semantics = {
    "Save": {
      "std_args": ["obj"],
      "return_type": "None",
      "variants": {"torch": {"api": "torch.save"}, "jax": {"api": "jax.save"}},
    }
  }
  out_file = tmp_path / "test_save.py"
  generator.generate(semantics, out_file)
  content = out_file.read_text()
  assert "Operation expected to return None / Void" in content
  assert "verify_results" not in content
  assert "assert len(results) >= 2" in content
  assert "try:" in content
  assert "results['torch'] =" in content


def test_standard_return_logic(generator, tmp_path):
  """Verifies the behavior of standard return logic."""
  semantics = {"Add": {"std_args": ["x"], "variants": {"torch": {"api": "torch.add"}, "jax": {"api": "jax.add"}}}}
  out_file = tmp_path / "test_add.py"
  generator.generate(semantics, out_file)
  content = out_file.read_text()
  assert "verify_results(ref, val" in content
