"""Test suite for the Generator Runtime module."""

import typing

import pytest

from ml_switcheroo.generated_tests.generator import TestCaseGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


class MockRuntimeSemantics(SemanticsManager):
  """Test helper class MockRuntimeSemantics."""

  def __init__(self) -> None:
    """Initializes the MockRuntimeSemantics instance."""
    self.test_templates: typing.Any = {"torch": {"import": "import torch"}}
    self.framework_configs: typing.Any = {}
    self.data: typing.Any = {}

  def get_test_template(self, fw: typing.Any) -> typing.Any:
    """Execute get test template helper logic.

    Args:
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.test_templates.get(fw)

  def get_framework_config(self, fw: typing.Any) -> typing.Any:
    """Mock implementation of get framework configuration.

    Args:
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return {}


@pytest.fixture
def generator(tmp_path: typing.Any) -> typing.Any:
  """Execute generator helper logic.

  Args:
      tmp_path: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  mgr: typing.Any = MockRuntimeSemantics()
  return TestCaseGenerator(semantics_mgr=mgr)


def test_runtime_file_creation(generator: typing.Any, tmp_path: typing.Any) -> None:
  """Verifies the behavior of runtime file creation.

  Args:
      generator: Test argument or fixture.
      tmp_path: Test argument or fixture.
  """
  tmp_out_dir: typing.Any = tmp_path / "gen_tests"
  generator._ensure_runtime_module(tmp_out_dir)
  runtime_file: typing.Any = tmp_out_dir / "runtime.py"
  assert runtime_file.exists()
  content: typing.Any = runtime_file.read_text(encoding="utf-8")
  assert "def verify_results(ref:" in content
  assert "isinstance(ref, dict)" in content
  assert "np.asanyarray(ref)" in content
  assert "np.array_equal" in content


def test_gen_tests_use_runtime_import(generator: typing.Any, tmp_path: typing.Any) -> None:
  """Verify gen tests use runtime import behavior.

  Args:
      generator: Test argument or fixture.
      tmp_path: Test argument or fixture.
  """
  semantics: typing.Any = {
    "abs": {"std_args": ["x"], "variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jnp.abs"}}}
  }
  generator.semantics_mgr.test_templates["jax"] = {"import": "import jax"}
  out_file: typing.Any = tmp_path / "gen_tests" / "test_abs.py"
  generator.generate(semantics, out_file)
  assert out_file.exists()
  content: typing.Any = out_file.read_text(encoding="utf-8")
  assert "from .runtime import *" in content
  assert "if TORCH_AVAILABLE:" in content
  assert "verify_results(ref, val" in content
