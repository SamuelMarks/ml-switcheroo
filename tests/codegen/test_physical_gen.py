"""Test suite for the Physical Gen module."""

import pytest
from ml_switcheroo.generated_tests.generator import TestCaseGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  """Mock Semantics class for testing purposes."""

  def __init__(self):
    """Initializes the MockSemantics instance."""
    self.data = {}
    self.test_templates = {
      "jax": {
        "import": "import jax\nimport jax.numpy as jnp",
        "convert_input": "jnp.array({np_var})",
        "to_numpy": "{res_var}",
        "jit_template": "jax.jit({fn}, static_argnums={static_argnums})",
      },
      "torch": {"import": "import torch", "convert_input": "torch.tensor({np_var})", "to_numpy": "{res_var}.numpy()"},
    }
    self.framework_configs = {"jax": {"traits": {"jit_static_args": ["axis", "keepdims"]}}}

  def get_test_template(self, fw):
    """Mock implementation of get test template."""
    return self.test_templates.get(fw)

  def get_framework_config(self, fw):
    """Mock implementation of get framework configuration."""
    return self.framework_configs.get(fw, {})


@pytest.fixture
def generator():
  """Provides a mock generator for testing."""
  mgr = MockSemantics()
  return TestCaseGenerator(semantics_mgr=mgr)


@pytest.fixture
def sample_spec():
  """Provides a mock sample spec for testing."""
  return {
    "abs": {"std_args": ["x"], "variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jnp.abs"}}},
    "mean": {
      "std_args": [("x", "Array"), ("axis", "int")],
      "variants": {"torch": {"api": "torch.mean", "args": {"axis": "dim"}}, "jax": {"api": "jnp.mean"}},
    },
  }


def test_generator_writes_file_and_runtime(generator, sample_spec, tmp_path):
  """Verifies the behavior of generator writes file and runtime."""
  out_file = tmp_path / "test_generated.py"
  generator.generate(sample_spec, out_file)
  assert out_file.exists()
  content = out_file.read_text()
  assert "from .runtime import *" in content
  assert "verify_results(ref, val" in content
  runtime_file = tmp_path / "runtime.py"
  assert runtime_file.exists()
  runtime_content = runtime_file.read_text()
  assert 'find_spec("torch")' in runtime_content
  assert "TORCH_AVAILABLE" in runtime_content
  assert 'find_spec("jax")' in runtime_content
  assert "chex_mod.assert_trees_all_close" in runtime_content
  assert "isinstance(ref, dict)" in runtime_content


def test_jit_static_argnums(generator, sample_spec, tmp_path):
  """Verifies the behavior of jit static argnums."""
  out_file = tmp_path / "test_jit.py"
  generator.generate(sample_spec, out_file)
  content = out_file.read_text()
  start = content.find("test_gen_mean")
  block = content[start:]
  assert "static_argnums=(1,)" in block
  assert "jax.jit(fn, static_argnums=(1,))" in block


def test_generated_file_is_valid_python(generator, sample_spec, tmp_path):
  """Verifies the behavior of generated file is valid python."""
  out_file = tmp_path / "test_valid.py"
  generator.generate(sample_spec, out_file)
  try:
    compile(out_file.read_text(), out_file.name, "exec")
  except SyntaxError as e:
    pytest.fail(f"Generated Invalid Python: {e}")


def test_overwrite_behavior(generator, sample_spec, tmp_path):
  """Verifies the behavior of overwrite behavior."""
  out_file = tmp_path / "test_overwrite.py"
  out_file.write_text("OLD CONTENT")
  generator.generate(sample_spec, out_file)
  content = out_file.read_text()
  assert "OLD CONTENT" not in content
  assert "def test_gen_abs" in content


def test_skip_existing_manual_test(generator, sample_spec, tmp_path):
  """Verifies the behavior of skip existing manual test."""
  out_file = tmp_path / "test_manual.py"
  out_file.write_text("def test_gen_abs(): pass")
  generator.generate(sample_spec, out_file)
  content = out_file.read_text()
  assert "def test_gen_abs" not in content
  assert "def test_gen_mean" in content
