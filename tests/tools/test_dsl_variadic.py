"""Test suite for the Dsl Variadic module."""

from ml_switcheroo.core.dsl import FrameworkVariant, OperationDef, ParameterDef


def test_parameter_variadic_defaults() -> None:
  """Verify `ParameterDef` handles parameter variadic defaults as expected."""
  p: ParameterDef = ParameterDef(name="x")
  assert p.name == "x"
  assert p.is_variadic is False
  assert p.kind == "positional_or_keyword"


def test_parameter_variadic_explicit() -> None:
  """Verify `ParameterDef` handles parameter variadic explicit as expected."""
  p: ParameterDef = ParameterDef(name="tensors", is_variadic=True)
  assert p.is_variadic is True
  assert p.name == "tensors"


def test_variadic_integration_in_op_def() -> None:
  """Verify `OperationDef` handles variadic integration in operation def as expected."""
  op: OperationDef = OperationDef(
    operation="MaxVariadic",
    description="Elementwise max of variable number of tensors",
    std_args=[
      ParameterDef(name="args", is_variadic=True, type="Tensor"),
      ParameterDef(name="out", type="Tensor", default="None"),
    ],
    variants={"torch": FrameworkVariant(api="torch.maximum")},
  )
  assert len(op.std_args) == 2
  v_param = op.std_args[0]
  assert isinstance(v_param, ParameterDef)
  assert v_param.name == "args"
  assert v_param.is_variadic is True
  n_param = op.std_args[1]
  assert isinstance(n_param, ParameterDef)
  assert n_param.name == "out"
  assert n_param.is_variadic is False


def test_parameter_kind_field() -> None:
  """Verify `ParameterDef` handles parameter kind field as expected."""
  p: ParameterDef = ParameterDef(name="x", kind="positional_only")
  assert p.kind == "positional_only"
