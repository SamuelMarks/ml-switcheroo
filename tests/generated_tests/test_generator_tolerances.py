"""Test suite for the Generator Tolerances module."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ml_switcheroo.generated_tests.generator import TestCaseGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def generator(monkeypatch: pytest.MonkeyPatch) -> TestCaseGenerator:
  """Create a configured `TestCaseGenerator` fixture for generator tolerances tests.

  Args:
      monkeypatch: Pytest monkeypatch fixture.

  Returns:
      Configured `generator` fixture value (TestCaseGenerator).
  """
  mgr: SemanticsManager = MagicMock(spec=SemanticsManager)
  mock_templates: dict[str, dict[str, str]] = {
    "torch": {"import": "import torch", "convert_input": "{np_var}", "to_numpy": "{res_var}"},
    "jax": {"import": "import jax", "convert_input": "{np_var}", "to_numpy": "{res_var}"},
  }
  monkeypatch.setattr(mgr, "get_test_template", MagicMock(side_effect=lambda fw: mock_templates.get(fw)), raising=False)
  monkeypatch.setattr(mgr, "test_templates", mock_templates, raising=False)
  monkeypatch.setattr(mgr, "get_framework_config", MagicMock(return_value={}), raising=False)
  return TestCaseGenerator(semantics_mgr=mgr)


def test_tolerance_defaults(generator: TestCaseGenerator, tmp_path: Path) -> None:
  """Verify `generator.generate` emits `rtol=0.001` for tolerance defaults.

  Args:
      generator: Code or MLIR generator instance under test.
      tmp_path: Pytest temporary directory path fixture.
  """
  semantics = {"DefaultOp": {"std_args": ["x"], "variants": {"torch": {"api": "t.op"}, "jax": {"api": "j.op"}}}}
  out_file: Path = tmp_path / "test_default.py"
  generator.generate(semantics, out_file)
  content: str = out_file.read_text()
  assert "rtol=0.001" in content
  assert "atol=0.0001" in content


def test_tolerance_override(generator: TestCaseGenerator, tmp_path: Path) -> None:
  """Verify `generator.generate` emits `rtol=0.05` for tolerance override.

  Args:
      generator: Code or MLIR generator instance under test.
      tmp_path: Pytest temporary directory path fixture.
  """
  semantics = {
    "LooseOp": {
      "std_args": ["x"],
      "test_rtol": 0.05,
      "test_atol": 1.0,
      "variants": {"torch": {"api": "t.op"}, "jax": {"api": "j.op"}},
    }
  }
  out_file: Path = tmp_path / "test_override.py"
  generator.generate(semantics, out_file)
  content: str = out_file.read_text()
  assert "rtol=0.05" in content
  assert "atol=1.0" in content


def test_tolerance_scientific_notation(generator: TestCaseGenerator, tmp_path: Path) -> None:
  """Verify `generator.generate` handles tolerance scientific notation as expected.

  Args:
      generator: Code or MLIR generator instance under test.
      tmp_path: Pytest temporary directory path fixture.
  """
  semantics = {
    "StrictOp": {
      "std_args": ["x"],
      "test_rtol": 1e-07,
      "test_atol": 1e-09,
      "variants": {"torch": {"api": "t.op"}, "jax": {"api": "j.op"}},
    }
  }
  out_file: Path = tmp_path / "test_sci.py"
  generator.generate(semantics, out_file)
  content: str = out_file.read_text()
  assert "rtol=1e-07" in content or "rtol=1e-7" in content
  assert "atol=1e-09" in content or "atol=1e-9" in content
