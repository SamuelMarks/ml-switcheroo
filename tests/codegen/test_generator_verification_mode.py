"""Test suite for the Generator Verification Mode module."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ml_switcheroo.generated_tests.generator import TestCaseGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def generator(tmp_path: Path) -> TestCaseGenerator:
  """Execute generator helper logic.

  Args:
      tmp_path: Pytest temporary directory path fixture.

  Returns:
      Configured `generator` fixture value (TestCaseGenerator).
  """
  mgr: SemanticsManager = MagicMock(spec=SemanticsManager)
  setattr(
    mgr,
    "get_test_template",
    MagicMock(return_value={"torch": {"import": "import torch", "convert_input": "{np_var}", "to_numpy": "{res_var}"}}),
  )
  setattr(mgr, "get_framework_config", MagicMock(return_value={}))
  return TestCaseGenerator(semantics_mgr=mgr)


def test_emit_approx_default(generator: TestCaseGenerator, tmp_path: Path) -> None:
  """Emits approx default.

  Args:
      generator: Code or MLIR generator instance under test.
      tmp_path: Pytest temporary directory path fixture.
  """
  semantics = {"op": {"std_args": ["x"], "variants": {"torch": {"api": "t.op"}, "jax": {"api": "j.op"}}}}
  out: Path = tmp_path / "test_approx.py"
  generator.generate(semantics, out)
  content: str = out.read_text()
  assert "exact=False" in content


def test_emit_exact_mode(generator: TestCaseGenerator, tmp_path: Path) -> None:
  """Emits exact mode.

  Args:
      generator: Code or MLIR generator instance under test.
      tmp_path: Pytest temporary directory path fixture.
  """
  semantics = {
    "op": {
      "std_args": ["x"],
      "verification_mode": "exact",
      "variants": {"torch": {"api": "t.op"}, "jax": {"api": "j.op"}},
    }
  }
  out: Path = tmp_path / "test_exact.py"
  generator.generate(semantics, out)
  content: str = out.read_text()
  assert "exact=True" in content


def test_emit_custom_tolerances(generator: TestCaseGenerator, tmp_path: Path) -> None:
  """Emits custom tolerances.

  Args:
      generator: Code or MLIR generator instance under test.
      tmp_path: Pytest temporary directory path fixture.
  """
  semantics = {
    "op": {
      "std_args": ["x"],
      "test_rtol": 1e-05,
      "test_atol": 1e-08,
      "variants": {"torch": {"api": "t.op"}, "jax": {"api": "j.op"}},
    }
  }
  out: Path = tmp_path / "test_tols.py"
  generator.generate(semantics, out)
  content: str = out.read_text()
  assert "rtol=1e-05" in content
  assert "atol=1e-08" in content
