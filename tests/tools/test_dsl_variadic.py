"""Test suite for the Dsl Variadic module."""

from typing import Any

from ml_switcheroo.core.dsl import FrameworkVariant, OperationDef, ParameterDef


def test_parameter_variadic_defaults() -> None:
  """Verifies the behavior of parameter variadic defaults."""
  p: ParameterDef = ParameterDef(name="x")
  assert getattr(p, "name") == "x"
  assert getattr(p, "is_variadic") is False
  assert getattr(p, "kind") == "positional_or_keyword"


def test_parameter_variadic_explicit() -> None:
  """Verifies the behavior of parameter variadic explicit."""
  p: ParameterDef = ParameterDef(name="tensors", is_variadic=True)
  assert getattr(p, "is_variadic") is True
  assert getattr(p, "name") == "tensors"


def test_variadic_integration_in_op_def() -> None:
  """Verifies the behavior of variadic integration in op def."""
  op: OperationDef = OperationDef(
    operation="MaxVariadic",
    description="Elementwise max of variable number of tensors",
    std_args=[
      ParameterDef(name="args", is_variadic=True, type="Tensor"),
      ParameterDef(name="out", type="Tensor", default="None"),
    ],
    variants={"torch": FrameworkVariant(api="torch.maximum")},
  )
  assert len(getattr(op, "std_args")) == 2
  v_param: Any = getattr(op, "std_args")[0]
  assert getattr(v_param, "name") == "args"
  assert getattr(v_param, "is_variadic") is True
  n_param: Any = getattr(op, "std_args")[1]
  assert getattr(n_param, "name") == "out"
  assert getattr(n_param, "is_variadic") is False


def test_parameter_kind_field() -> None:
  """Verifies the behavior of parameter kind field."""
  p: ParameterDef = ParameterDef(name="x", kind="positional_only")
  assert getattr(p, "kind") == "positional_only"
