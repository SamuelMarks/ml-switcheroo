"""
Tests for Runtime Module Generation (DRY Fix).

Verifies that:
1. `_ensure_runtime_module` constructs a valid `runtime.py`.
2. The runtime file contains try/except import blocks for configured frameworks.
3. The runtime file generates the `verify_results` helper.
4. Generated test code imports from `.runtime`.
"""

import sys
import pytest
from unittest.mock import MagicMock
from ml_switcheroo.generated_tests.generator import TestCaseGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


class MockRuntimeSemantics(SemanticsManager):
  """Mocks logic for runtime generation tests."""

  def __init__(self):
    self.test_templates = {
      "torch": {"import": "import torch"},
    }
    # Empty configs for other calls
    self.framework_configs = {}
    self.data = {}

  def get_test_template(self, fw):
    return self.test_templates.get(fw)

  def get_framework_config(self, fw):
    return {}


@pytest.fixture
def generator(tmp_path):
  mgr = MockRuntimeSemantics()
  return TestCaseGenerator(semantics_mgr=mgr)


def test_runtime_file_creation(generator, tmp_path):
  """
  Verify `runtime.py` is created with correct content and helpers.
  """
  tmp_out_dir = tmp_path / "gen_tests"
  generator._ensure_runtime_module(tmp_out_dir)

  runtime_file = tmp_out_dir / "runtime.py"
  assert runtime_file.exists()

  content = runtime_file.read_text(encoding="utf-8")

  # Check Helper Function Injection
  assert "def verify_results(ref, val" in content

  # Check for recursive logic (the new feature)
  assert "isinstance(ref, dict)" in content
  assert "np.asanyarray(ref)" in content
  # Verify handling of non-numeric types
  assert "np.array_equal" in content


def test_gen_tests_use_runtime_import(generator, tmp_path):
  """
  Verify generated tests import the flags from .runtime.
  """
  # Update semantics to have 2 frameworks with templates in the Mock
  semantics = {"abs": {"std_args": ["x"], "variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jnp.abs"}}}}

  # Add JAX template to mock for this specific test
  generator.semantics_mgr.test_templates["jax"] = {"import": "import jax"}

  out_file = tmp_path / "gen_tests" / "test_abs.py"
  generator.generate(semantics, out_file)

  assert out_file.exists()
  content = out_file.read_text(encoding="utf-8")

  # 1. Imports check
  assert "from .runtime import *" in content

  # 2. Logic Check (should assume flags exist)
  assert "if TORCH_AVAILABLE:" in content

  # 3. Check for presence of verification call
  assert "verify_results(ref, val" in content
