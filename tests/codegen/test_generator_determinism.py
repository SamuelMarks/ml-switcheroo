"""
Tests for Determinism Injection.
"""

import pytest
from unittest.mock import MagicMock
from ml_switcheroo.generated_tests.generator import TestCaseGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def generator():
  mgr = MagicMock(spec=SemanticsManager)
  templates = {"torch": {"import": "import torch"}}
  mgr.get_test_template.side_effect = lambda fw: templates.get(fw)
  mgr.test_templates = templates
  mgr.get_framework_config.return_value = {}
  return TestCaseGenerator(semantics_mgr=mgr)


def test_determinism_fixture_injection(generator, tmp_path):
  """
  Verify runtime.py includes the ensure_determinism fixture.
  """
  out_dir = tmp_path / "gen"
  generator._ensure_runtime_module(out_dir)

  content = (out_dir / "runtime.py").read_text()

  assert "@pytest.fixture(autouse=True)" in content
  assert "def ensure_determinism():" in content
  assert "random.seed(42)" in content
  assert "np.random.seed(42)" in content
