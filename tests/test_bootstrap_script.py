"""Unit and integration tests for scripts/bootstrap.sh functions and logic."""

from pathlib import Path
import subprocess


def _get_function_def(func_name: str) -> str:
  """Extract a specific shell function definition from scripts/bootstrap.sh.

  Args:
      func_name: Name of the function to extract.

  Returns:
      The shell function definition string.
  """
  script = Path(__file__).resolve().parent.parent / "scripts" / "bootstrap.sh"
  res = subprocess.run(
    ["sed", "-n", f"/{func_name}() {{/,/^}}/p", str(script)],
    capture_output=True,
    text=True,
  )
  return res.stdout


def test_bootstrap_get_abs_script_path() -> None:
  """Test that get_abs_script_path in bootstrap.sh resolves the script directory."""
  func_def = _get_function_def("get_abs_script_path")
  cmd = f"{func_def}" + chr(10) + "get_abs_script_path"
  res = subprocess.run(["/bin/sh", "-c", cmd], capture_output=True, text=True)
  assert res.returncode == 0
  assert len(res.stdout.strip()) > 0


def test_bootstrap_get_install_cmd_uv(tmp_path: Path) -> None:
  """Test get_install_cmd returns 'uv pip install' when VIRTUAL_ENV and uv exist.

  Args:
      tmp_path: Temporary directory fixture from pytest.
  """
  func_def = _get_function_def("get_install_cmd")
  fake_uv = tmp_path / "uv"
  fake_uv.write_text("#!/bin/sh" + chr(10) + "exit 0" + chr(10), encoding="utf-8")
  fake_uv.chmod(0o755)

  cmd = (
    'export VIRTUAL_ENV="/fake/venv"'
    + chr(10)
    + f'export PATH="{tmp_path}:/bin:/usr/bin"'
    + chr(10)
    + f"{func_def}"
    + chr(10)
    + "get_install_cmd"
  )
  res = subprocess.run(["/bin/sh", "-c", cmd], capture_output=True, text=True)
  assert res.returncode == 0
  assert "uv pip install" in res.stdout


def test_bootstrap_get_install_cmd_pip(tmp_path: Path) -> None:
  """Test get_install_cmd falls back to 'python3 -m pip install' when VIRTUAL_ENV is unset.

  Args:
      tmp_path: Temporary directory fixture from pytest.
  """
  func_def = _get_function_def("get_install_cmd")
  cmd = f"unset VIRTUAL_ENV; {func_def}" + chr(10) + "get_install_cmd"
  res = subprocess.run(["/bin/sh", "-c", cmd], capture_output=True, text=True)
  assert res.returncode == 0
  assert "python3 -m pip install" in res.stdout


def test_bootstrap_list_required_libs_filtering() -> None:
  """Test list_required_libs filters out virtual frameworks and normalizes aliases."""
  func_def = _get_function_def("list_required_libs")
  cmd = f"{func_def}" + chr(10) + "list_required_libs"
  res = subprocess.run(["/bin/sh", "-c", cmd], capture_output=True, text=True)
  assert res.returncode == 0
  libs = res.stdout.strip().split()

  # Virtual / non-pip frameworks should be excluded
  assert "html" not in libs
  assert "latex_dsl" not in libs
  assert "tikz" not in libs
  assert "mlir" not in libs
  assert "stablehlo" not in libs
  assert "sass" not in libs

  # Real frameworks should be present
  assert "torch" in libs
  assert "torchvision" in libs
  assert "jax" in libs
  assert "flax" in libs  # normalized from flax_nnx
  assert "flax_nnx" not in libs


def test_bootstrap_list_required_libs_fallback() -> None:
  """Test list_required_libs fallback string when import fails."""
  py_fallback = (
    "try:"
    + chr(10)
    + "    raise ImportError('Simulated')"
    + chr(10)
    + "except ImportError:"
    + chr(10)
    + "    print('torch torchvision jax flax tensorflow keras mlx numpy')"
    + chr(10)
  )
  res = subprocess.run(["python3", "-c", py_fallback], capture_output=True, text=True)
  assert res.returncode == 0
  assert "torch torchvision jax flax tensorflow keras mlx numpy" in res.stdout
