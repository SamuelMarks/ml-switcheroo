"""Tests for the Runtime Module Logic Generator."""

import pathlib
from unittest import mock

from ml_switcheroo.generated_tests.runtime_builder import ensure_runtime_module, get_required_packages
from ml_switcheroo.semantics.manager import SemanticsManager


def test_get_required_packages_simple() -> None:
  """Verify `get_required_packages` handles simple as expected."""
  packages = get_required_packages("import torch\nimport jax")
  assert packages == ["torch", "jax"]


def test_get_required_packages_from() -> None:
  """Verify `get_required_packages` handles from as expected."""
  packages = get_required_packages("from torch import nn\nfrom jax.numpy import abs")
  assert packages == ["torch", "jax"]


def test_get_required_packages_aliases() -> None:
  """Verify `get_required_packages` handles aliases as expected."""
  packages = get_required_packages("import torch as t\nimport jax.numpy as jnp")
  assert packages == ["torch", "jax"]


def test_get_required_packages_deduplication() -> None:
  """Verify `get_required_packages` handles deduplication as expected."""
  packages = get_required_packages("import torch\nfrom torch import nn")
  assert packages == ["torch"]


def test_get_required_packages_syntax_error() -> None:
  """Verify `get_required_packages` handles syntax error as expected."""
  packages = get_required_packages("import class")
  assert packages == []
  assert get_required_packages("import a, b\x0ax = 1\x0afrom . import sibling\x0a") == ["a", "b"]


def test_ensure_runtime_module_custom_semantics_no_req_pkgs(tmp_path: pathlib.Path) -> None:
  """Verify ensure_runtime_module handles custom SemanticsManager returning non-package imports.

  Args:
      tmp_path: Pytest temporary directory path fixture.
  """
  sem = SemanticsManager()
  with mock.patch.object(sem, "get_test_template", return_value={"import": "import"}):
    ensure_runtime_module(tmp_path, frameworks=["dummy"], mgr=sem)
  content = (tmp_path / "runtime.py").read_text(encoding="utf-8")
  assert "DUMMY_AVAILABLE = True" in content


def test_ensure_runtime_module(tmp_path: pathlib.Path) -> None:
  """Verify `runtime_file.exists` emits `import sys` for ensure runtime module.

  Args:
      tmp_path: Pytest temporary directory path fixture.
  """
  out_dir = tmp_path / "tests"

  # We mock get_template to supply predictable test data
  def mock_get_template(mgr: SemanticsManager, fw: str) -> dict[str, str] | None:
    """Mock get template.

    Args:
        mgr: Semantics manager test instance (`SemanticsManager`).
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, str] | None` from mock get template.
    """
    if fw == "torch":
      return {"import": "import torch"}
    if fw == "jax":
      return {"import": "import jax.numpy as jnp"}
    if fw == "tensorflow":
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


def test_ensure_runtime_module_no_frameworks(tmp_path: pathlib.Path) -> None:
  """Verify `ensure_runtime_module` emits `# --- torch ---` for no frameworks.

  Args:
      tmp_path: Pytest temporary directory path fixture.
  """
  out_dir = tmp_path / "tests"

  def mock_get_template(mgr: SemanticsManager, fw: str) -> dict[str, str] | None:
    """Mock get template.

    Args:
        mgr: Semantics manager test instance (`SemanticsManager`).
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, str] | None` from mock get template.
    """
    if fw == "torch":
      return {"import": "import torch"}
    if fw == "jax":
      return {"import": "import jax.numpy as jnp"}
    return {"import": f"import {fw}"}

  with mock.patch("ml_switcheroo.generated_tests.runtime_builder.get_template", side_effect=mock_get_template):
    ensure_runtime_module(out_dir)

  content = (out_dir / "runtime.py").read_text(encoding="utf-8")

  # It should still include torch and jax by default
  assert "# --- torch ---" in content
  assert "# --- jax ---" in content
  assert "def verify_results" in content


def test_ensure_runtime_module_empty_template_import(tmp_path: pathlib.Path) -> None:
  """Verify `ensure_runtime_module` emits `TORCH_AVAILABLE = True` for empty template import.

  Args:
      tmp_path: Pytest temporary directory path fixture.
  """
  out_dir = tmp_path / "tests"

  def mock_get_template(mgr: SemanticsManager, fw: str) -> dict[str, str] | None:
    """Mock get template.

    Args:
        mgr: Semantics manager test instance (`SemanticsManager`).
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, str] | None` from mock get template.
    """
    if fw == "torch":
      # Give it an import that parses to empty required_packages, e.g., a SyntaxError string or just comments
      return {"import": "class  # Syntax error to trigger empty req pkgs"}
    return None

  with mock.patch("ml_switcheroo.generated_tests.runtime_builder.get_template", side_effect=mock_get_template):
    ensure_runtime_module(out_dir, frameworks=["torch"])

  content = (out_dir / "runtime.py").read_text(encoding="utf-8")
  assert "TORCH_AVAILABLE = True" in content
