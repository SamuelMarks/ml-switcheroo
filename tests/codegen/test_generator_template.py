"""
Tests for Template-Based Test Generation.

Verifies:
1. Generator uses default templates (from Registry) when no external JSON is provided.
2. Generator loads templates from `SemanticsManager`.
3. Generated code reflects custom backend configs (e.g. TinyGrad).
"""

import pytest
from ml_switcheroo.generated_tests.generator import TestGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


class MockTemplateSemantics(SemanticsManager):
  """Mocks SemanticsManager with custom templates provided at init."""

  def __init__(self, templates=None):
    # We purposely skip the real init to avoid loading files or registry defaults
    # for controlled unit testing.
    self.test_templates = templates or {}
    # Ensure minimal state for other methods if called
    self.data = {}
    self.framework_configs = {}

  def get_test_template(self, framework):
    return self.test_templates.get(framework)


@pytest.fixture
def semantics_data():
  """Basic operation data for generation."""
  return {
    "abs": {
      "std_args": ["x"],
      "variants": {
        "torch": {"api": "torch.abs"},
        "tinygrad": {"api": "tinygrad.abs"},
      },
    }
  }


def test_default_template_fallback(tmp_path, semantics_data):
  """
  Scenario: Use default 'torch' template (sourced from Adapter Registry).
  """
  # NOTE: We use the REAL SemanticsManager here to verify registry integration
  mgr = SemanticsManager()

  gen = TestGenerator(semantics_mgr=mgr)
  out_file = tmp_path / "test_defaults.py"

  # Must add a valid second one (JAX) to force generation logic (needs >=2 variants)
  semantics_data["abs"]["variants"]["jax"] = {"api": "jnp.abs"}

  gen.generate(semantics_data, out_file)

  content = out_file.read_text()
  assert "import torch" in content
  assert "import jax" in content
  # Should skip tinygrad because no template exists for it in defaults
  assert "Framework: tinygrad" not in content


def test_custom_backend_template(tmp_path, semantics_data):
  """
  Scenario: Add templates for 'tinygrad'.
  Expect: Generated code includes TinyGrad block.
  """
  custom_templates = {
    "torch": {
      "import": "import torch",
      "convert_input": "torch.tensor({np_var})",
      "to_numpy": "{res_var}.numpy()",
    },
    "tinygrad": {
      "import": "from tinygrad.tensor import Tensor",
      "convert_input": "Tensor({np_var})",
      "to_numpy": "{res_var}.numpy()",
    },
  }

  mgr = MockTemplateSemantics(templates=custom_templates)
  gen = TestGenerator(semantics_mgr=mgr)
  out_file = tmp_path / "test_tinygrad.py"

  gen.generate(semantics_data, out_file)

  content = out_file.read_text()

  # check structure
  assert "Framework: tinygrad" in content
  assert "from tinygrad.tensor import Tensor" in content
  assert "Tensor(np_x)" in content


def test_jit_config_via_template(tmp_path):
  """
  Scenario: Enable JIT via template flag "jit_wrap": "True".
  """
  data = {
    "add": {
      "std_args": ["x"],
      "variants": {
        "custom_jit": {"api": "lib.add"},
        "other": {"api": "other.add"},
      },
    }
  }

  templates = {
    "custom_jit": {
      "import": "import lib",
      "convert_input": "{np_var}",
      "to_numpy": "{res_var}",
      "jit_wrap": "True",  # String or Bool
    },
    "other": {
      "import": "import other",
      "convert_input": "{np_var}",
      "to_numpy": "{res_var}",
    },
  }

  mgr = MockTemplateSemantics(templates=templates)
  gen = TestGenerator(semantics_mgr=mgr)
  out_file = tmp_path / "test_jit_flag.py"

  gen.generate(data, out_file)
  content = out_file.read_text()

  assert "Framework: custom_jit" in content
  assert "jax.jit(fn" in content  # Currently hardcoded to use jax.jit if flag is true


def test_invalid_framework_skipped(tmp_path, semantics_data):
  """
  Scenario: Data has variant 'ghost_fw', but no template exists.
  Expect: 'ghost_fw' is excluded from generation.
  """
  semantics_data["abs"]["variants"]["ghost_fw"] = {"api": "ghost.abs"}
  # Must add a valid second one to force generation at all
  semantics_data["abs"]["variants"]["jax"] = {"api": "jnp.abs"}

  # Use Real Manager to ensure default behavior (missing ghost)
  mgr = SemanticsManager()
  gen = TestGenerator(semantics_mgr=mgr)
  out_file = tmp_path / "test_skip.py"

  gen.generate(semantics_data, out_file)
  content = out_file.read_text()

  assert "Framework: jax" in content
  assert "Framework: ghost_fw" not in content
