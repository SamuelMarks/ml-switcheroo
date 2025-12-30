"""
Tests for Dynamic Tolerance Injection in Test Generator.

Verifies:
1. Default behavior: Uses 1e-3, 1e-4 if not specified.
2. Explicit override: Uses values from `test_rtol`/`test_atol` fields.
3. Robustness: Correctly formats scientific notation floats.
"""

import pytest
from unittest.mock import MagicMock
from ml_switcheroo.generated_tests.generator import TestGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def generator():
  mgr = MagicMock(spec=SemanticsManager)

  # FIX: Ensure Templates are returned so generator produces code
  mock_templates = {
    "torch": {"import": "import torch", "convert_input": "{np_var}", "to_numpy": "{res_var}"},
    "jax": {"import": "import jax", "convert_input": "{np_var}", "to_numpy": "{res_var}"},
  }

  mgr.get_test_template.side_effect = lambda fw: mock_templates.get(fw)
  # Populate attributes
  mgr.test_templates = mock_templates
  mgr.get_framework_config.return_value = {}
  return TestGenerator(semantics_mgr=mgr)


def test_tolerance_defaults(generator, tmp_path):
  """
  Scenario: No constraints in definition.
  Expect: rtol=1e-3, atol=1e-4.
  """
  semantics = {
    "DefaultOp": {
      "std_args": ["x"],
      "variants": {"torch": {"api": "t.op"}, "jax": {"api": "j.op"}},
    }
  }

  out_file = tmp_path / "test_default.py"
  generator.generate(semantics, out_file)

  content = out_file.read_text()
  assert "rtol=0.001" in content
  assert "atol=0.0001" in content


def test_tolerance_override(generator, tmp_path):
  """
  Scenario: Explicit loose tolerances for unstable ops (e.g. fast math).
  """
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
  """
  Scenario: Tiny tolerances using E-notation.
  """
  semantics = {
    "StrictOp": {
      "std_args": ["x"],
      "test_rtol": 1e-7,
      "test_atol": 1e-9,
      "variants": {"torch": {"api": "t.op"}, "jax": {"api": "j.op"}},
    }
  }

  out_file = tmp_path / "test_sci.py"
  generator.generate(semantics, out_file)

  content = out_file.read_text()
  # Python's float.__repr__ generally handles small floats nicely (1e-07)
  assert "rtol=1e-07" in content or "rtol=1e-7" in content
  assert "atol=1e-09" in content or "atol=1e-9" in content
