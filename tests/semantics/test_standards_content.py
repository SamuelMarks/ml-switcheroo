"""Test suite for the Standards Content module."""

import pytest
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture(scope="module")
def mgr():
  """Helper to mgr."""
  return SemanticsManager()


def test_functional_math_ops(mgr):
  """Verifies the behavior of functional math ops."""
  data = mgr.get_known_apis()
  if not data:
    pytest.skip("No semantics loaded (Bootstrap needed)")
  assert "abs" in data
  assert "add" in data
  assert "Mean" in data
  abs_op = data["abs"]
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
  """Verifies the behavior of neural ops."""
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
  assert "in_channels" in args or "input" in args
  assert "kernel_size" in args or "weight" in args


def test_optimizer_standards(mgr):
  """Verifies the behavior of optimizer standards."""
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
  assert "lr" in args or "learning_rate" in args or "params" in args


def test_io_constants(mgr):
  """Verifies the behavior of I/O constants."""
  data = mgr.get_known_apis()
  assert "Save" in data
  assert "Load" in data
