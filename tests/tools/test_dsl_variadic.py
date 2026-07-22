"""Test suite for the Dsl Variadic module."""

from ml_switcheroo.core.dsl import ParameterDef, OperationDef, FrameworkVariant


def test_parameter_variadic_defaults():
  """Verifies the behavior of parameter variadic defaults."""
  p = ParameterDef(name="x")
  assert p.name == "x"
  assert p.is_variadic is False
  assert p.kind == "positional_or_keyword"


def test_parameter_variadic_explicit():
  """Verifies the behavior of parameter variadic explicit."""
  p = ParameterDef(name="tensors", is_variadic=True)
  assert p.is_variadic is True
  assert p.name == "tensors"


def test_variadic_integration_in_op_def():
  """Verifies the behavior of variadic integration in op def."""
  op = OperationDef(
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
  assert v_param.name == "args"
  assert v_param.is_variadic is True
  n_param = op.std_args[1]
  assert n_param.name == "out"
  assert n_param.is_variadic is False


def test_parameter_kind_field():
  """Verifies the behavior of parameter kind field."""
  p = ParameterDef(name="x", kind="positional_only")
  assert p.kind == "positional_only"
