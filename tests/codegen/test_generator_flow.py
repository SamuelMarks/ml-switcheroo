"""Test suite for the Generator Flow module."""

import pytest
from unittest.mock import MagicMock
from ml_switcheroo.generated_tests.generator import TestCaseGenerator
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def mock_mgr():
  """Provides a mock mgr for testing."""
  mgr = MagicMock(spec=SemanticsManager)
  templates = {
    "torch": {"import": "import torch", "convert_input": "torch.tensor({np_var})", "to_numpy": "{res_var}.numpy()"},
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
    "numpy": {"import": "import numpy as np", "convert_input": "{np_var}", "to_numpy": "{res_var}"},
  }
  mgr.get_test_template.side_effect = lambda fw: templates.get(fw)
  mgr.get_framework_config.return_value = {}
  mgr.test_templates = templates
  return mgr


def test_generation_runtime_import(tmp_path, mock_mgr):
  """Verifies the behavior of generation runtime import."""
  semantics = {"add": {"std_args": ["x"], "variants": {"torch": {"api": "torch.add"}, "jax": {"api": "jnp.add"}}}}
  out_file = tmp_path / "test_structure.py"
  gen = TestCaseGenerator(semantics_mgr=mock_mgr)
  gen.generate(semantics, out_file)
  content = out_file.read_text()
  assert "from .runtime import *" in content


def test_generation_safety(tmp_path, mock_mgr):
  """Verifies the behavior of generation safety."""
  semantics = {"abs": {"variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jnp.abs"}}}}
  out_file = tmp_path / "test_generated.py"
  out_file.write_text("\ndef test_gen_abs():\n    # Manual override\n    assert True\n")
  gen = TestCaseGenerator(semantics_mgr=mock_mgr)
  gen.generate(semantics, out_file)
  content = out_file.read_text()
  assert "np.random.randn" not in content
  assert "results = {}" not in content


def test_generation_multi_backend(tmp_path, mock_mgr):
  """Verifies the behavior of generation multi backend."""
  semantics = {
    "add": {
      "std_args": ["x", "y"],
      "variants": {"torch": {"api": "torch.add"}, "jax": {"api": "jnp.add"}, "tensorflow": {"api": "tf.math.add"}},
    }
  }
  out_file = tmp_path / "test_multi.py"
  gen = TestCaseGenerator(semantics_mgr=mock_mgr)
  gen.generate(semantics, out_file)
  content = out_file.read_text()
  assert "if TORCH_AVAILABLE:" in content
  assert "if TENSORFLOW_AVAILABLE:" in content
  assert "tf.convert_to_tensor(np_x)" in content
  assert "res.numpy()" in content


def test_excludes_single_variant(tmp_path, mock_mgr):
  """Verifies the behavior of excludes single variant."""
  semantics = {"unique_op": {"variants": {"torch": {"api": "torch.unique_thing"}}}}
  out_file = tmp_path / "test_empty.py"
  gen = TestCaseGenerator(semantics_mgr=mock_mgr)
  gen.generate(semantics, out_file)
  content = out_file.read_text() if out_file.exists() else ""
  assert "def test_gen_unique_op" not in content


def test_generation_unary_vs_binary(tmp_path, mock_mgr):
  """Verifies the behavior of generation unary vs binary."""
  semantics = {"neg": {"std_args": ["x"], "variants": {"torch": {"api": "torch.neg"}, "numpy": {"api": "np.negative"}}}}
  out_file = tmp_path / "test_unary.py"
  gen = TestCaseGenerator(semantics_mgr=mock_mgr)
  gen.generate(semantics, out_file)
  content = out_file.read_text()
  assert "np_y" not in content
  assert "torch.neg(torch.tensor(np_x))" in content
