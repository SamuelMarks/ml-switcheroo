"""
Tests for Physical Test File Generation.

Verifies:
1. Generator can create new files.
2. Imports are deduplicated and aggregated.
3. JIT logic is applied when templates request it.
4. Output files are executable (valid python).
"""

import pytest
import sys
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

from ml_switcheroo.generated_tests.generator import TestGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  """
  Mock Manager providing predefined templates and specs.
  Skips filesystem logic.
  """

  def __init__(self):
    self.data = {}
    # Templates for JAX (with JIT) and Torch (Standard)
    self.test_templates = {
      "jax": {
        "import": "import jax\nimport jax.numpy as jnp",
        "convert_input": "jnp.array({np_var})",
        "to_numpy": "np.array({res_var})",
        "jit_wrap": "True",
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
  return TestGenerator(semantics_mgr=mgr)


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


def test_generator_writes_file(generator, sample_spec, tmp_path):
  """Verify physical file creation."""
  out_file = tmp_path / "test_generated.py"

  generator.generate(sample_spec, out_file)

  assert out_file.exists()
  content = out_file.read_text()

  # Check Imports
  assert "import torch" in content
  assert "import jax" in content

  # Check Functions
  assert "def test_gen_abs():" in content
  assert "def test_gen_mean():" in content


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
  """Verify existing manual tests are respected if parsing existing fails? No, check name collision."""
  # TestGenerator _parse_existing_tests reads the target file.
  # But since we use mode="w", we overwrite it anyway.
  # The logic in generate() reads *before* opening for write.

  out_file = tmp_path / "test_manual.py"
  out_file.write_text("def test_gen_abs(): pass")  # Pretend it exists

  generator.generate(sample_spec, out_file)

  content = out_file.read_text()
  # It should effectively have cleared the file (mode='w')
  # BUT, the loop skips adding 'test_gen_abs' to the *new* content list.
  # So the file will just contain imports + test_gen_mean.

  assert "def test_gen_abs" not in content
  assert "def test_gen_mean" in content
