"""
Tests for Generator Verification Mode Emittance.
"""

import pytest
from unittest.mock import MagicMock
from ml_switcheroo.generated_tests.generator import TestGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def generator(tmp_path):
  mgr = MagicMock(spec=SemanticsManager)
  mgr.get_test_template.return_value = {
    "torch": {"import": "import torch", "convert_input": "{np_var}", "to_numpy": "{res_var}"}
  }
  mgr.get_framework_config.return_value = {}
  return TestGenerator(semantics_mgr=mgr)


def test_emit_approx_default(generator, tmp_path):
  semantics = {
    "op": {
      "std_args": ["x"],
      "variants": {"torch": {"api": "t.op"}, "jax": {"api": "j.op"}},
    }
  }
  out = tmp_path / "test_approx.py"
  generator.generate(semantics, out)

  content = out.read_text()
  assert "exact=False" in content


def test_emit_exact_mode(generator, tmp_path):
  semantics = {
    "op": {
      "std_args": ["x"],
      "verification_mode": "exact",
      "variants": {"torch": {"api": "t.op"}, "jax": {"api": "j.op"}},
    }
  }
  out = tmp_path / "test_exact.py"
  generator.generate(semantics, out)

  content = out.read_text()
  assert "exact=True" in content


def test_emit_custom_tolerances(generator, tmp_path):
  semantics = {
    "op": {
      "std_args": ["x"],
      "test_rtol": 1e-5,
      "test_atol": 1e-8,
      "variants": {"torch": {"api": "t.op"}, "jax": {"api": "j.op"}},
    }
  }
  out = tmp_path / "test_tols.py"
  generator.generate(semantics, out)

  content = out.read_text()
  assert "rtol=1e-05" in content
  assert "atol=1e-08" in content
