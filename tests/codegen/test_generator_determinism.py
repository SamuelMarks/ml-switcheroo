"""Test suite for the Generator Determinism module."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ml_switcheroo.generated_tests.generator import TestCaseGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def generator() -> TestCaseGenerator:
  """Docstring."""
  mgr: SemanticsManager = MagicMock(spec=SemanticsManager)
  templates: dict[str, dict[str, str]] = {"torch": {"import": "import torch"}}
  mgr.get_test_template = MagicMock(side_effect=lambda fw: templates.get(fw))
  setattr(mgr, "test_templates", templates)
  mgr.get_framework_config = MagicMock(return_value={})
  return TestCaseGenerator(semantics_mgr=mgr)


def test_determinism_fixture_injection(generator: TestCaseGenerator, tmp_path: Path) -> None:
  """Verifies the behavior of determinism fixture injection."""
  out_dir: Path = tmp_path / "gen"
  generator._ensure_runtime_module(out_dir)
  content: str = (out_dir / "runtime.py").read_text()
  assert "@pytest.fixture(autouse=True)" in content
  assert "def ensure_determinism()" in content
  assert "random.seed(42)" in content
  assert "np.random.seed(42)" in content
