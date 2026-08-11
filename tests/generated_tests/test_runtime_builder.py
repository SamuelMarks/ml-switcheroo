"""Tests for the Runtime Module Logic Generator."""

import pathlib
from unittest import mock

from ml_switcheroo.generated_tests.runtime_builder import get_required_packages, ensure_runtime_module


def test_get_required_packages_simple():
  """Test getting required packages for simple imports."""
  packages = get_required_packages("import torch\nimport jax")
  assert packages == ["torch", "jax"]


def test_get_required_packages_from():
  """Test getting required packages for from imports."""
  packages = get_required_packages("from torch import nn\nfrom jax.numpy import abs")
  assert packages == ["torch", "jax"]


def test_get_required_packages_aliases():
  """Test getting required packages with aliases."""
  packages = get_required_packages("import torch as t\nimport jax.numpy as jnp")
  assert packages == ["torch", "jax"]


def test_get_required_packages_deduplication():
  """Test deduplication of required packages."""
  packages = get_required_packages("import torch\nfrom torch import nn")
  assert packages == ["torch"]


def test_get_required_packages_syntax_error():
  """Test handling of invalid import syntax."""
  packages = get_required_packages("import class")
  assert packages == []


def test_ensure_runtime_module(tmp_path: pathlib.Path):
  """Test generating the runtime module."""
  out_dir = tmp_path / "tests"

  # We mock get_template to supply predictable test data
  def mock_get_template(mgr, fw):
    """Mock get template."""
    if fw == "torch":
      return {"import": "import torch"}
    elif fw == "jax":
      return {"import": "import jax.numpy as jnp"}
    elif fw == "tensorflow":
      return {"import": "import tensorflow as tf"}
    return None

  with mock.patch("ml_switcheroo.generated_tests.runtime_builder.get_template", side_effect=mock_get_template):
    ensure_runtime_module(out_dir, frameworks=["torch", "jax", "tensorflow", "unknown"])

  runtime_file = out_dir / "runtime.py"
  assert runtime_file.exists()

  content = runtime_file.read_text(encoding="utf-8")

  # Check imports
  assert "import sys" in content
  assert "import pytest" in content

  # Check framework blocks
  assert "# --- torch ---" in content
  assert "TORCH_AVAILABLE =" in content

  assert "# --- jax ---" in content
  assert "JAX_AVAILABLE =" in content

  assert "# --- tensorflow ---" in content
  assert "TENSORFLOW_AVAILABLE =" in content

  # Check that it doesn't include the unknown framework
  assert "# --- unknown ---" not in content

  # Check the shared logic is included
  assert "def verify_results" in content
  assert "def ensure_determinism" in content


def test_ensure_runtime_module_no_frameworks(tmp_path: pathlib.Path):
  """Test generating with no explicit frameworks requested."""
  out_dir = tmp_path / "tests"

  def mock_get_template(mgr, fw):
    """Mock get template."""
    if fw == "torch":
      return {"import": "import torch"}
    elif fw == "jax":
      return {"import": "import jax.numpy as jnp"}
    return {"import": f"import {fw}"}

  with mock.patch("ml_switcheroo.generated_tests.runtime_builder.get_template", side_effect=mock_get_template):
    ensure_runtime_module(out_dir)

  content = (out_dir / "runtime.py").read_text(encoding="utf-8")

  # It should still include torch and jax by default
  assert "# --- torch ---" in content
  assert "# --- jax ---" in content
  assert "def verify_results" in content


def test_ensure_runtime_module_empty_template_import(tmp_path: pathlib.Path):
  """Test generating when a template has no specific import logic (uses default) or evaluates to empty packages."""
  out_dir = tmp_path / "tests"

  def mock_get_template(mgr, fw):
    """Mock get template."""
    if fw == "torch":
      # Give it an import that parses to empty required_packages, e.g., a SyntaxError string or just comments
      return {"import": "class  # Syntax error to trigger empty req pkgs"}
    return None

  with mock.patch("ml_switcheroo.generated_tests.runtime_builder.get_template", side_effect=mock_get_template):
    ensure_runtime_module(out_dir, frameworks=["torch"])

  content = (out_dir / "runtime.py").read_text(encoding="utf-8")
  assert "TORCH_AVAILABLE = True" in content
