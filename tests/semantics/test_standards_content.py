"""
Content Tests for Standards.

Verifies that key operations are present in the loaded SemanticsManager.
Replaces legacy checks against standards_internal.py.
"""

import pytest
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture(scope="module")
def mgr():
  return SemanticsManager()


def test_functional_math_ops(mgr):
  """Verify Array API defaults are loaded."""
  data = mgr.get_known_apis()
  # Use get to avoid KeyError if files missing during bootstrap
  if not data:
    pytest.skip("No semantics loaded (Bootstrap needed)")

  assert "Abs" in data
  assert "Add" in data
  assert "Mean" in data

  abs_op = data["Abs"]
  # Handle list of dicts (new format) or strings
  args = []
  for arg in abs_op.get("std_args", []):
    if isinstance(arg, dict):
      args.append(arg.get("name"))
    elif isinstance(arg, str):
      args.append(arg)
    elif isinstance(arg, (tuple, list)):
      args.append(arg[0])

  assert "x" in args


def test_neural_ops(mgr):
  """Verify Neural Network layers are loaded."""
  data = mgr.get_known_apis()
  assert "Conv2d" in data
  assert "Linear" in data
  assert "MultiheadAttention" in data

  conv = data["Conv2d"]
  args = []
  for arg in conv.get("std_args", []):
    if isinstance(arg, dict):
      args.append(arg.get("name"))
    elif isinstance(arg, str):
      args.append(arg)

  assert "in_channels" in args
  assert "kernel_size" in args


def test_optimizer_standards(mgr):
  """Verify Optimizers are present."""
  data = mgr.get_known_apis()
  assert "Adam" in data
  assert "SGD" in data

  adam = data["Adam"]
  args = []
  for arg in adam.get("std_args", []):
    if isinstance(arg, dict):
      args.append(arg.get("name"))
    elif isinstance(arg, str):
      args.append(arg)

  assert "lr" in args


def test_io_constants(mgr):
  """Verify Extras/IO."""
  data = mgr.get_known_apis()
  assert "Save" in data
  assert "Load" in data
