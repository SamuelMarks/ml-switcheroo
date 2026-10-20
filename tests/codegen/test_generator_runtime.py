"""Test suite for the Generator Runtime module."""

import pathlib
import typing
from pydantic import JsonValue

import pytest

from ml_switcheroo.generated_tests.generator import TestCaseGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


class MockRuntimeSemantics(SemanticsManager):
  """Test helper class MockRuntimeSemantics."""

  def __init__(self) -> None:
    """Initializes the MockRuntimeSemantics instance."""
    self.test_templates = {"torch": {"import": "import torch"}}
    self.framework_configs = {}
    self.data = {}

  def get_test_template(self, fw: str) -> typing.Optional[dict[str, str]]:
    """Execute get test template helper logic.

    Args:
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.test_templates.get(fw)

  def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Args:
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return {}


@pytest.fixture
def generator(tmp_path: pathlib.Path) -> TestCaseGenerator:
  """Execute generator helper logic.

  Args:
      tmp_path: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  mgr = MockRuntimeSemantics()
  return TestCaseGenerator(semantics_mgr=mgr)


def test_runtime_file_creation(generator: TestCaseGenerator, tmp_path: pathlib.Path) -> None:
  """Verifies the behavior of runtime file creation.

  Args:
      generator: Test argument or fixture.
      tmp_path: Test argument or fixture.
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
  """Verify gen tests use runtime import behavior.

  Args:
      generator: Test argument or fixture.
      tmp_path: Test argument or fixture.
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
