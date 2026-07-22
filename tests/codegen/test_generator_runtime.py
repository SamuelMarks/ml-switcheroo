"""Test suite for the Generator Runtime module."""

import pytest
from ml_switcheroo.generated_tests.generator import TestCaseGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


class MockRuntimeSemantics(SemanticsManager):
  """Mock Runtime Semantics class for testing purposes."""

  def __init__(self):
    """Initializes the MockRuntimeSemantics instance."""
    self.test_templates = {"torch": {"import": "import torch"}}
    self.framework_configs = {}
    self.data = {}

  def get_test_template(self, fw):
    """Mock implementation of get test template."""
    return self.test_templates.get(fw)

  def get_framework_config(self, fw):
    """Mock implementation of get framework configuration."""
    return {}


@pytest.fixture
def generator(tmp_path):
  """Provides a mock generator for testing."""
  mgr = MockRuntimeSemantics()
  return TestCaseGenerator(semantics_mgr=mgr)


def test_runtime_file_creation(generator, tmp_path):
  """Verifies the behavior of runtime file creation."""
  tmp_out_dir = tmp_path / "gen_tests"
  generator._ensure_runtime_module(tmp_out_dir)
  runtime_file = tmp_out_dir / "runtime.py"
  assert runtime_file.exists()
  content = runtime_file.read_text(encoding="utf-8")
  assert "def verify_results(ref, val" in content
  assert "isinstance(ref, dict)" in content
  assert "np.asanyarray(ref)" in content
  assert "np.array_equal" in content


def test_gen_tests_use_runtime_import(generator, tmp_path):
  """Verifies the behavior of generation tests use runtime import."""
  semantics = {"abs": {"std_args": ["x"], "variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jnp.abs"}}}}
  generator.semantics_mgr.test_templates["jax"] = {"import": "import jax"}
  out_file = tmp_path / "gen_tests" / "test_abs.py"
  generator.generate(semantics, out_file)
  assert out_file.exists()
  content = out_file.read_text(encoding="utf-8")
  assert "from .runtime import *" in content
  assert "if TORCH_AVAILABLE:" in content
  assert "verify_results(ref, val" in content
