"""
Tests for Physical Test File Generation.

Verifies:
1. Generator can create new files.
2. Runtime modules are created.
3. JIT logic is applied when templates request it.
4. Output files syntax is valid.
5. Comparison logic uses `verify_results`.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock

from ml_switcheroo.generated_tests.generator import TestCaseGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  """
  Mock Manager providing predefined templates and specs.
  Skips filesystem logic.
  """

  def __init__(self):
    self.data = {}
    # Templates for JAX (with JIT) and Torch (Standard)
    # Updated: Explicit JIT Template required for JAX in decoupled architecture
    self.test_templates = {
      "jax": {
        "import": "import jax\nimport jax.numpy as jnp",
        "convert_input": "jnp.array({np_var})",
        # Chex support uses identity logic here
        "to_numpy": "{res_var}",
        "jit_template": "jax.jit({fn}, static_argnums={static_argnums})",
      },
      "torch": {"import": "import torch", "convert_input": "torch.tensor({np_var})", "to_numpy": "{res_var}.numpy()"},
    }
    # Config for static args
    self.framework_configs = {"jax": {"traits": {"jit_static_args": ["axis", "keepdims"]}}}

  def get_test_template(self, fw):
    return self.test_templates.get(fw)

  def get_framework_config(self, fw):
    return self.framework_configs.get(fw, {})


@pytest.fixture
def generator():
  mgr = MockSemantics()
  return TestCaseGenerator(semantics_mgr=mgr)


@pytest.fixture
def sample_spec():
  """Returns a simple Semantics dictionary."""
  return {
    "abs": {"std_args": ["x"], "variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jnp.abs"}}},
    "mean": {
      "std_args": [("x", "Array"), ("axis", "int")],
      "variants": {"torch": {"api": "torch.mean", "args": {"axis": "dim"}}, "jax": {"api": "jnp.mean"}},
    },
  }


def test_generator_writes_file_and_runtime(generator, sample_spec, tmp_path):
  """Verify physical file creation and runtime injection."""
  out_file = tmp_path / "test_generated.py"

  # We must ensure runtime can be generated next to output
  # TestCaseGenerator._ensure_runtime_module does this
  generator.generate(sample_spec, out_file)

  assert out_file.exists()
  content = out_file.read_text()

  # Check Imports
  assert "from .runtime import *" in content

  # Check verify helper logic
  assert "verify_results(ref, val" in content

  # Check runtime file existence
  runtime_file = tmp_path / "runtime.py"
  assert runtime_file.exists()
  runtime_content = runtime_file.read_text()

  assert "import torch" in runtime_content
  assert "TORCH_AVAILABLE" in runtime_content
  assert "import jax" in runtime_content
  # Updated check for substring logic rather than full match
  assert "chex_mod.assert_trees_all_close" in runtime_content
  assert "isinstance(ref, dict)" in runtime_content


def test_jit_static_argnums(generator, sample_spec, tmp_path):
  """Verify JIT wrapper includes static_argnums for 'axis'."""
  out_file = tmp_path / "test_jit.py"
  generator.generate(sample_spec, out_file)

  content = out_file.read_text()

  # Extract mean test block
  start = content.find("test_gen_mean")
  block = content[start:]

  # Check for static_argnums
  # 'axis' is index 1 in [x, axis]
  assert "static_argnums=(1,)" in block
  assert "jax.jit(fn, static_argnums=(1,))" in block


def test_generated_file_is_valid_python(generator, sample_spec, tmp_path):
  """Verify syntax of generated file by importing it."""
  out_file = tmp_path / "test_valid.py"
  generator.generate(sample_spec, out_file)

  # Try compiling it
  try:
    compile(out_file.read_text(), out_file.name, "exec")
  except SyntaxError as e:
    pytest.fail(f"Generated Invalid Python: {e}")


def test_overwrite_behavior(generator, sample_spec, tmp_path):
  """Verify generator overwrites file cleanly."""
  out_file = tmp_path / "test_overwrite.py"
  out_file.write_text("OLD CONTENT")

  generator.generate(sample_spec, out_file)
  content = out_file.read_text()

  assert "OLD CONTENT" not in content
  assert "def test_gen_abs" in content


def test_skip_existing_manual_test(generator, sample_spec, tmp_path):
  """Verify existing manual tests are respected."""
  out_file = tmp_path / "test_manual.py"
  out_file.write_text("def test_gen_abs(): pass")  # Pretend it exists

  generator.generate(sample_spec, out_file)

  content = out_file.read_text()
  # manually defined 'test_gen_abs' should NOT be overwritten in the sense
  # that the generator should filter it out from the NEW content list.
  # However, the file itself is overwritten by the generation process logic which
  # recreates the file. So unless we parse and merge (which we don't, we just skip generating logic),
  # the old content is lost, but the *Generated* content is suppressed.

  assert "def test_gen_abs" not in content
  assert "def test_gen_mean" in content
