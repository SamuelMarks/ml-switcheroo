"""Test suite for the Generator Template module."""

import pytest
from ml_switcheroo.generated_tests.generator import TestCaseGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


class MockTemplateSemantics(SemanticsManager):
  """Mock Template Semantics class for testing purposes."""

  def __init__(self, templates=None):
    """Initializes the MockTemplateSemantics instance."""
    self.test_templates = templates or {}
    self.data = {}
    self.framework_configs = {}

  def get_test_template(self, framework):
    """Mock implementation of get test template."""
    return self.test_templates.get(framework)

  def get_framework_config(self, framework):
    """Mock implementation of get framework configuration."""
    return {}


@pytest.fixture
def semantics_data():
  """Provides a mock semantics data for testing."""
  return {"abs": {"std_args": ["x"], "variants": {"torch": {"api": "torch.abs"}, "tinygrad": {"api": "tinygrad.abs"}}}}


def test_default_template_fallback(tmp_path, semantics_data):
  """Verifies the behavior of default template fallback."""
  defaults = {
    "torch": {"import": "import torch", "convert_input": "torch.tensor({np_var})", "to_numpy": "{res_var}.numpy()"},
    "jax": {"import": "import jax", "convert_input": "jnp.array({np_var})", "to_numpy": "{res_var}"},
  }
  mgr = MockTemplateSemantics(templates=defaults)
  gen = TestCaseGenerator(semantics_mgr=mgr)
  out_file = tmp_path / "test_defaults.py"
  semantics_data["abs"]["variants"]["jax"] = {"api": "jnp.abs"}
  gen.generate(semantics_data, out_file)
  content = out_file.read_text()
  runtime_content = (out_file.parent / "runtime.py").read_text()
  assert 'find_spec("torch")' in runtime_content
  assert 'find_spec("jax")' in runtime_content
  assert "Framework: tinygrad" not in content


def test_custom_backend_template(tmp_path, semantics_data):
  """Verifies the behavior of custom backend template."""
  custom_templates = {
    "torch": {"import": "import torch", "convert_input": "torch.tensor({np_var})", "to_numpy": "{res_var}.numpy()"},
    "tinygrad": {
      "import": "from tinygrad.tensor import Tensor",
      "convert_input": "Tensor({np_var})",
      "to_numpy": "{res_var}.numpy()",
    },
  }
  mgr = MockTemplateSemantics(templates=custom_templates)
  gen = TestCaseGenerator(semantics_mgr=mgr)
  out_file = tmp_path / "test_tinygrad.py"
  gen.generate(semantics_data, out_file)
  content = out_file.read_text()
  runtime_content = (out_file.parent / "runtime.py").read_text()
  assert "Framework: tinygrad" in content
  assert 'find_spec("tinygrad")' in runtime_content
  assert "Tensor(np_x)" in content


def test_jit_config_via_template(tmp_path):
  """Verifies the behavior of jit configuration via template."""
  data = {"add": {"std_args": ["x"], "variants": {"custom_jit": {"api": "lib.add"}, "other": {"api": "other.add"}}}}
  templates = {
    "custom_jit": {
      "import": "import lib",
      "convert_input": "{np_var}",
      "to_numpy": "{res_var}",
      "jit_template": "jax.jit({fn})",
    },
    "other": {"import": "import other", "convert_input": "{np_var}", "to_numpy": "{res_var}"},
  }
  mgr = MockTemplateSemantics(templates=templates)
  gen = TestCaseGenerator(semantics_mgr=mgr)
  out_file = tmp_path / "test_jit_flag.py"
  gen.generate(data, out_file)
  content = out_file.read_text()
  assert "Framework: custom_jit" in content
  assert "jax.jit(fn)" in content


def test_invalid_framework_skipped(tmp_path, semantics_data):
  """Verifies the behavior of invalid framework skipped."""
  semantics_data["abs"]["variants"]["ghost_fw"] = {"api": "ghost.abs"}
  semantics_data["abs"]["variants"]["jax"] = {"api": "jnp.abs"}
  templates = {"torch": {"import": "import torch"}, "jax": {"import": "import jax"}}
  mgr = MockTemplateSemantics(templates=templates)
  gen = TestCaseGenerator(semantics_mgr=mgr)
  out_file = tmp_path / "test_skip.py"
  gen.generate(semantics_data, out_file)
  content = out_file.read_text()
  assert "Framework: jax" in content
  assert "Framework: ghost_fw" not in content
