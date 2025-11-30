"""
Tests for Packaging and Entry Points.

Verifies that:
1. The package exposes a valid entry point script.
2. The main function is callable as a console script.
"""

import sys
import subprocess
import pytest


def test_entry_point_importable():
  """
  Verify that 'ml_switcheroo.cli.__main__:main' is resolvable.
  This ensures the pyproject.toml string points to valid code.
  """
  try:
    from ml_switcheroo.cli.__main__ import main

    assert callable(main)
  except ImportError as e:
    pytest.fail(f"Could not import entry point: {e}")


def test_command_execution_via_python_m(tmp_path):
  """
  Verify that `python -m ml_switcheroo` works (module execution).
  """
  # Simply asking for help is enough to prove it runs
  result = subprocess.run([sys.executable, "-m", "ml_switcheroo", "--help"], capture_output=True, text=True)

  if result.returncode != 0:
    pytest.fail(f"Execution failed: {result.stderr}")

  assert result.returncode == 0
  assert "usage:" in result.stdout
  assert "ml-switcheroo" in result.stdout


def test_module_structure_compliance():
  """
  Verify src layout allows finding the package.
  """
  import ml_switcheroo

  assert ml_switcheroo.__file__ is not None

  # Check if plugins are discoverable relative to package
  import ml_switcheroo.plugins

  assert ml_switcheroo.plugins.__file__ is not None
