"""
Tests for Variable Backend Test Generation.

Verifies:
1.  Generator handles 'torch' and 'jax' correctly (standard case).
2.  Generator handles 'tensorflow' or 'mlx' if present in Semantics (dynamic case).
3.  Generator skips operations with only 1 variant.
4.  Generator respects manually existing tests.
5.  **NEW**: Runtime import usage.
"""

import pytest
from unittest.mock import MagicMock
from ml_switcheroo.generated_tests.generator import TestGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def mock_mgr():
  """
  Provides a SemanticsManager with pre-loaded templates for all frameworks
  used in these tests. This ensures tests run independently of the filesystem/bootstrap state.
  """
  mgr = MagicMock(spec=SemanticsManager)

  templates = {
    "torch": {
      "import": "import torch",
      "convert_input": "torch.tensor({np_var})",
      "to_numpy": "{res_var}.numpy()",
    },
    "jax": {
      "import": "import jax\nimport jax.numpy as jnp",
      "convert_input": "jnp.array({np_var})",
      "to_numpy": "np.array({res_var})",
    },
    "tensorflow": {
      "import": "import tensorflow as tf",
      "convert_input": "tf.convert_to_tensor({np_var})",
      "to_numpy": "{res_var}.numpy()",
    },
    "numpy": {
      "import": "import numpy as np",
      "convert_input": "{np_var}",
      "to_numpy": "{res_var}",
    },
  }

  mgr.get_test_template.side_effect = lambda fw: templates.get(fw)
  # Safe fallback for other calls
  mgr.get_framework_config.return_value = {}
  # Ensure test_templates attribute exists for runtime generator
  mgr.test_templates = templates
  return mgr


def test_generation_runtime_import(tmp_path, mock_mgr):
  """
  Verify `from .runtime import *` is generated.
  """
  semantics = {"add": {"std_args": ["x"], "variants": {"torch": {"api": "torch.add"}, "jax": {"api": "jnp.add"}}}}

  out_file = tmp_path / "test_structure.py"
  gen = TestGenerator(semantics_mgr=mock_mgr)
  gen.generate(semantics, out_file)

  content = out_file.read_text()
  assert "from .runtime import *" in content


def test_generation_safety(tmp_path, mock_mgr):
  """Verify manual overrides are respected."""
  # 1. Mock Semantics
  semantics = {"abs": {"variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jnp.abs"}}}}

  # 2. Setup existing file with a manual test
  out_file = tmp_path / "test_generated.py"
  out_file.write_text(""" 
def test_gen_abs(): 
    # Manual override
    assert True
""")

  # 3. Run Generator
  gen = TestGenerator(semantics_mgr=mock_mgr)
  gen.generate(semantics, out_file)

  # 4. Verify Content
  content = out_file.read_text()
  # Should NOT contain the generated blocks
  assert "np.random.randn" not in content
  assert "results = {}" not in content


def test_generation_multi_backend(tmp_path, mock_mgr):
  """
  Scenario: Semantics includes Torch, JAX, and TensorFlow.
  Expect: Generated code contains execution blocks for all three.
  """
  # 1. Mock Semantics
  semantics = {
    "add": {
      "std_args": ["x", "y"],  # Binary
      "variants": {"torch": {"api": "torch.add"}, "jax": {"api": "jnp.add"}, "tensorflow": {"api": "tf.math.add"}},
    }
  }

  out_file = tmp_path / "test_multi.py"

  # 2. Run
  gen = TestGenerator(semantics_mgr=mock_mgr)
  gen.generate(semantics, out_file)

  # 3. Verify Imports via Runtime
  content = out_file.read_text()

  # 4. Verify Execution Blocks
  assert "if TORCH_AVAILABLE:" in content
  assert "if TENSORFLOW_AVAILABLE:" in content
  assert "tf.convert_to_tensor(np_x)" in content
  assert "res.numpy()" in content  # TF specific normalization


def test_excludes_single_variant(tmp_path, mock_mgr):
  """
  Scenario: Operation only defined for Torch.
  Expect: No test generated (cannot compare).
  """
  semantics = {"unique_op": {"variants": {"torch": {"api": "torch.unique_thing"}}}}

  out_file = tmp_path / "test_empty.py"

  gen = TestGenerator(semantics_mgr=mock_mgr)
  gen.generate(semantics, out_file)

  content = out_file.read_text() if out_file.exists() else ""
  # Header might be written if mode 'w', but no function body
  assert "def test_gen_unique_op" not in content


def test_generation_unary_vs_binary(tmp_path, mock_mgr):
  """
  Verify argument count logic.
  """
  semantics = {
    "neg": {
      "std_args": ["x"],  # Unary
      "variants": {"torch": {"api": "torch.neg"}, "numpy": {"api": "np.negative"}},
    }
  }

  out_file = tmp_path / "test_unary.py"
  gen = TestGenerator(semantics_mgr=mock_mgr)
  gen.generate(semantics, out_file)

  content = out_file.read_text()
  # Should not generate np_y
  assert "np_y" not in content
  # Call should be unary
  assert "torch.neg(torch.tensor(np_x))" in content
