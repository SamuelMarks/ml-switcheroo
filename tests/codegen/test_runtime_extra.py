"""Module docstring."""

import pathlib
from unittest import mock
from ml_switcheroo.generated_tests.runtime_builder import ensure_runtime_module, get_required_packages


def test_get_required_packages():
  """Test get required packages."""
  assert get_required_packages("import os") == ["os"]
  assert get_required_packages("import os.path") == ["os"]
  assert get_required_packages("from os import path") == ["os"]
  assert get_required_packages("import a, b") == ["a", "b"]
  assert get_required_packages("=invalid syntax=") == []


def test_ensure_runtime_module_tensorflow_and_jax(tmp_path: pathlib.Path):
  """Test ensure runtime module tensorflow and jax."""
  mgr = mock.MagicMock()

  def mock_get_template(m, fw):
    """Mock get template."""
    if fw == "tensorflow":
      return {"import": "import tensorflow as tf"}
    if fw == "jax":
      return {"import": "import jax.numpy as jnp\nimport jax"}
    if fw == "torch":
      return {"import": "import torch"}
    return None

  with mock.patch("ml_switcheroo.generated_tests.runtime_builder.get_template", side_effect=mock_get_template):
    ensure_runtime_module(tmp_path, frameworks=["tensorflow", "jax", "unknown"], mgr=mgr)

  content = (tmp_path / "runtime.py").read_text()
  assert "TENSORFLOW_AVAILABLE" in content
  assert "JAX_AVAILABLE" in content
  assert "TORCH_AVAILABLE" in content
  assert "tf" in content
