"""Test suite for the Generator Tolerances module."""

import pytest
from unittest.mock import MagicMock
from ml_switcheroo.generated_tests.generator import TestCaseGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def generator():
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


def test_tolerance_defaults(generator, tmp_path):
  """Verifies the behavior of tolerance defaults."""
  semantics = {"DefaultOp": {"std_args": ["x"], "variants": {"torch": {"api": "t.op"}, "jax": {"api": "j.op"}}}}
  out_file = tmp_path / "test_default.py"
  generator.generate(semantics, out_file)
  content = out_file.read_text()
  assert "rtol=0.001" in content
  assert "atol=0.0001" in content


def test_tolerance_override(generator, tmp_path):
  """Verifies the behavior of tolerance override."""
  semantics = {
    "LooseOp": {
      "std_args": ["x"],
      "test_rtol": 0.05,
      "test_atol": 1.0,
      "variants": {"torch": {"api": "t.op"}, "jax": {"api": "j.op"}},
    }
  }
  out_file = tmp_path / "test_override.py"
  generator.generate(semantics, out_file)
  content = out_file.read_text()
  assert "rtol=0.05" in content
  assert "atol=1.0" in content


def test_tolerance_scientific_notation(generator, tmp_path):
  """Verifies the behavior of tolerance scientific notation."""
  semantics = {
    "StrictOp": {
      "std_args": ["x"],
      "test_rtol": 1e-07,
      "test_atol": 1e-09,
      "variants": {"torch": {"api": "t.op"}, "jax": {"api": "j.op"}},
    }
  }
  out_file = tmp_path / "test_sci.py"
  generator.generate(semantics, out_file)
  content = out_file.read_text()
  assert "rtol=1e-07" in content or "rtol=1e-7" in content
  assert "atol=1e-09" in content or "atol=1e-9" in content
