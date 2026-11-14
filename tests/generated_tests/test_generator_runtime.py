"""Test suite for the Generator Runtime module."""

import pathlib

import pytest
from pydantic import JsonValue

from ml_switcheroo.generated_tests.generator import TestCaseGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


class MockRuntimeSemantics(SemanticsManager):
  """Test double `MockRuntimeSemantics` used in generator runtime tests."""

  def __init__(self) -> None:
    """Initializes the MockRuntimeSemantics instance."""
    self.test_templates = {"torch": {"import": "import torch"}}
    self.framework_configs = {}
    self.data = {}

  def get_test_template(self, fw: str) -> dict[str, str] | None:
    """Execute `get` to support get test template in generator runtime tests.

    Args:
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, str] | None` from get test template.
    """
    return self.test_templates.get(fw)

  def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Args:
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Configuration dictionary for the specified framework.
    """
    return {}


@pytest.fixture
def generator(tmp_path: pathlib.Path) -> TestCaseGenerator:
  """Create a configured `TestCaseGenerator` fixture for generator runtime tests.

  Args:
      tmp_path: Pytest temporary directory path fixture.

  Returns:
      Configured `generator` fixture value (TestCaseGenerator).
  """
  mgr = MockRuntimeSemantics()
  return TestCaseGenerator(semantics_mgr=mgr)


def test_runtime_file_creation(generator: TestCaseGenerator, tmp_path: pathlib.Path) -> None:
  """Verify `generator._ensure_runtime_module` emits `def verify_results(ref:` for runtime file creation.

  Args:
      generator: Code or MLIR generator instance under test.
      tmp_path: Pytest temporary directory path fixture.
  """
  tmp_out_dir = tmp_path / "gen_tests"
  generator._ensure_runtime_module(tmp_out_dir)
  runtime_file = tmp_out_dir / "runtime.py"
  assert runtime_file.exists()
  content = runtime_file.read_text(encoding="utf-8")
  assert "def verify_results(ref:" in content
  assert "isinstance(ref, dict)" in content
  assert "np.asanyarray(ref)" in content
  assert "np.array_equal" in content


def test_gen_tests_use_runtime_import(generator: TestCaseGenerator, tmp_path: pathlib.Path) -> None:
  """Verify `generator.generate` emits `from .runtime import *` for gen tests use runtime import.

  Args:
      generator: Code or MLIR generator instance under test.
      tmp_path: Pytest temporary directory path fixture.
  """
  semantics: dict[str, dict[str, JsonValue]] = {
    "abs": {"std_args": ["x"], "variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jnp.abs"}}}
  }
  assert generator.semantics_mgr is not None
  generator.semantics_mgr.test_templates["jax"] = {"import": "import jax"}
  out_file = tmp_path / "gen_tests" / "test_abs.py"
  generator.generate(semantics, out_file)
  assert out_file.exists()
  content = out_file.read_text(encoding="utf-8")
  assert "from .runtime import *" in content
  assert "if TORCH_AVAILABLE:" in content
  assert "verify_results(ref, val" in content
