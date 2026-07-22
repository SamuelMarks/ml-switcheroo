"""Test suite for the Dsl module."""

from ml_switcheroo.core.dsl import OperationDef, ParameterDef, FrameworkVariant


def test_parameter_def_rich_defaults():
  """Verifies the behavior of parameter def rich defaults."""
  p1 = ParameterDef(name="d", default=1)
  assert p1.default == 1
  p2 = ParameterDef(name="flag", default=False)
  assert p2.default is False
  p3 = ParameterDef(name="eps", default=1e-05)
  assert p3.default == 1e-05
  p4 = ParameterDef(name="pads", default=[0, 0])
  assert p4.default == [0, 0]


def test_framework_variant_inject_args_rich_types():
  """Verifies the behavior of framework variant inject arguments rich types."""
  v = FrameworkVariant(api="foo", inject_args={"val": 1.5, "flag": False, "dims": [1, 2], "data": {"a": 1}})
  assert v.inject_args["dims"] == [1, 2]


def test_operation_def_structure():
  """Verifies the behavior of operation def structure."""
  data = {
    "operation": "TestOp",
    "description": "A test op",
    "std_args": [{"name": "x", "type": "int", "default": 0}],
    "variants": {"torch": {"api": "torch.test"}},
  }
  op = OperationDef(**data)
  assert op.std_args[0].default == 0
