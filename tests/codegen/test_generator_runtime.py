"""Test suite for the Generator Runtime module."""

import typing
from pathlib import Path

import pytest

from ml_switcheroo.generated_tests.generator import TestCaseGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


class MockRuntimeSemantics(SemanticsManager):
  """Docstring."""

  def __init__(self) -> None:
    """Initializes the MockRuntimeSemantics instance."""
    self.test_templates: dict[str, dict[str, str]] = {"torch": {"import": "import torch"}}
    self.framework_configs: dict[str, typing.Any] = {}
    self.data: dict[str, typing.Any] = {}

  def get_test_template(self, fw: str) -> typing.Optional[dict[str, str]]:
    """Docstring."""
    return self.test_templates.get(fw)

  def get_framework_config(self, fw: str) -> dict[str, typing.Any]:
    """Mock implementation of get framework configuration."""
    return {}


@pytest.fixture
def generator(tmp_path: Path) -> TestCaseGenerator:
  """Docstring."""
  mgr = MockRuntimeSemantics()
  return TestCaseGenerator(semantics_mgr=mgr)


def test_runtime_file_creation(generator: TestCaseGenerator, tmp_path: Path) -> None:
  """Verifies the behavior of runtime file creation."""
  tmp_out_dir: Path = tmp_path / "gen_tests"
  generator._ensure_runtime_module(tmp_out_dir)
  runtime_file: Path = tmp_out_dir / "runtime.py"
  assert runtime_file.exists()
  content: str = runtime_file.read_text(encoding="utf-8")
  assert "def verify_results(ref:" in content
  assert "isinstance(ref, dict)" in content
  assert "np.asanyarray(ref)" in content
  assert "np.array_equal" in content


def test_gen_tests_use_runtime_import(generator: TestCaseGenerator, tmp_path: Path) -> None:
  """Docstring."""
  semantics: dict[str, typing.Any] = {
    "abs": {"std_args": ["x"], "variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jnp.abs"}}}
  }
  generator.semantics_mgr.test_templates["jax"] = {"import": "import jax"}
  out_file: Path = tmp_path / "gen_tests" / "test_abs.py"
  generator.generate(semantics, out_file)
  assert out_file.exists()
  content: str = out_file.read_text(encoding="utf-8")
  assert "from .runtime import *" in content
  assert "if TORCH_AVAILABLE:" in content
  assert "verify_results(ref, val" in content
