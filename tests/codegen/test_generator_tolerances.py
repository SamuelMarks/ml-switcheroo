"""Test suite for the Generator Tolerances module."""

import typing
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ml_switcheroo.generated_tests.generator import TestCaseGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def generator() -> TestCaseGenerator:
  """Docstring."""
  mgr: SemanticsManager = MagicMock(spec=SemanticsManager)
  mock_templates: dict[str, dict[str, str]] = {
    "torch": {"import": "import torch", "convert_input": "{np_var}", "to_numpy": "{res_var}"},
    "jax": {"import": "import jax", "convert_input": "{np_var}", "to_numpy": "{res_var}"},
  }
  mgr.get_test_template = MagicMock(side_effect=lambda fw: mock_templates.get(fw))
  setattr(mgr, "test_templates", mock_templates)
  mgr.get_framework_config = MagicMock(return_value={})
  return TestCaseGenerator(semantics_mgr=mgr)


def test_tolerance_defaults(generator: TestCaseGenerator, tmp_path: Path) -> None:
  """Verifies the behavior of tolerance defaults."""
  semantics: dict[str, typing.Any] = {
    "DefaultOp": {"std_args": ["x"], "variants": {"torch": {"api": "t.op"}, "jax": {"api": "j.op"}}}
  }
  out_file: Path = tmp_path / "test_default.py"
  generator.generate(semantics, out_file)
  content: str = out_file.read_text()
  assert "rtol=0.001" in content
  assert "atol=0.0001" in content


def test_tolerance_override(generator: TestCaseGenerator, tmp_path: Path) -> None:
  """Verifies the behavior of tolerance override."""
  semantics: dict[str, typing.Any] = {
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
  """Verifies the behavior of tolerance scientific notation."""
  semantics: dict[str, typing.Any] = {
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
