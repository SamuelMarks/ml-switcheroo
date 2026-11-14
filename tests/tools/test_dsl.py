"""Test suite for the Dsl module."""

from ml_switcheroo.core.dsl import FrameworkVariant, OperationDef, ParameterDef


def test_parameter_def_rich_defaults() -> None:
  """Verify `ParameterDef` returns `1` for parameter def rich defaults."""
  p1: ParameterDef = ParameterDef(name="d", default=1)
  assert p1.default == 1
  p2: ParameterDef = ParameterDef(name="flag", default=False)
  assert p2.default is False
  p3: ParameterDef = ParameterDef(name="eps", default=1e-05)
  assert p3.default == 1e-05
  p4: ParameterDef = ParameterDef(name="pads", default=[0, 0])
  assert p4.default == [0, 0]


def test_framework_variant_inject_args_rich_types() -> None:
  """Verify `FrameworkVariant` handles framework variant inject args rich types as expected."""
  v: FrameworkVariant = FrameworkVariant(
    api="foo", inject_args={"val": 1.5, "flag": False, "dims": [1, 2], "data": {"a": 1}}
  )
  assert v.inject_args["dims"] == [1, 2]


def test_operation_def_structure() -> None:
  """Verify `OperationDef.model_validate` returns `0` for operation def structure."""
  data = {
    "operation": "TestOp",
    "description": "A test op",
    "std_args": [{"name": "x", "type": "int", "default": 0}],
    "variants": {"torch": {"api": "torch.test"}},
  }
  op: OperationDef = OperationDef.model_validate(data)
  first_arg = op.std_args[0]
  assert isinstance(first_arg, ParameterDef)
  assert first_arg.default == 0
