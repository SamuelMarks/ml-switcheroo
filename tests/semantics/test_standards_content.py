"""Test suite for the Standards Content module."""

import typing


import pytest

from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture(scope="module")
def mgr() -> typing.Any:
  """Helper to mgr.

  Returns:
      SemanticsManager: A semantics manager instance.
  """
  return SemanticsManager()


def test_functional_math_ops(mgr: typing.Any) -> None:
  """Verifies the behavior of functional math ops.

  Args:
      mgr (SemanticsManager): Manager fixture.
  """
  data: typing.Any = mgr.get_known_apis()
  if not data:
    pytest.skip("No semantics loaded (Bootstrap needed)")
  assert "abs" in data
  assert "add" in data
  assert "mean" in data or "Mean" in data
  abs_op: typing.Any = data["abs"]
  args: typing.Any = []
  for arg in abs_op.get("std_args", []):
    if isinstance(arg, dict):
      args.append(arg.get("name"))
    elif isinstance(arg, str):
      args.append(arg)
    elif isinstance(arg, (tuple, list)):
      args.append(arg[0])
  assert "x" in args or "input" in args


def test_neural_ops(mgr: typing.Any) -> None:
  """Verifies the behavior of neural ops.

  Args:
      mgr (SemanticsManager): Manager fixture.
  """
  data: typing.Any = mgr.get_known_apis()
  assert "Conv2d" in data
  assert "Linear" in data
  assert "MultiHeadAttention" in data or "MultiheadAttention" in data
  conv: typing.Any = data["Conv2d"]
  args: typing.Any = []
  for arg in conv.get("std_args", []):
    if isinstance(arg, dict):
      args.append(arg.get("name"))
    elif isinstance(arg, str):
      args.append(arg)
  assert "in_channels" in args or "input" in args
  assert "kernel_size" in args or "weight" in args


def test_optimizer_standards(mgr: typing.Any) -> None:
  """Verifies the behavior of optimizer standards.

  Args:
      mgr (SemanticsManager): Manager fixture.
  """
  data: typing.Any = mgr.get_known_apis()
  assert "Adam" in data
  assert "SGD" in data
  adam: typing.Any = data["Adam"]
  args: typing.Any = []
  for arg in adam.get("std_args", []):
    if isinstance(arg, dict):
      args.append(arg.get("name"))
    elif isinstance(arg, str):
      args.append(arg)
  assert "lr" in args or "learning_rate" in args or "params" in args


def test_io_constants(mgr: typing.Any) -> None:
  """Verifies the behavior of I/O constants.

  Args:
      mgr (SemanticsManager): Manager fixture.
  """
  data: typing.Any = mgr.get_known_apis()
  assert "Save" in data or "save" in data
  assert "Load" in data or "load" in data
