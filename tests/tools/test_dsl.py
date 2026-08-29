"""Test suite for the Dsl module."""

from typing import Any, Dict

from ml_switcheroo.core.dsl import FrameworkVariant, OperationDef, ParameterDef


def test_parameter_def_rich_defaults() -> None:
  """Verifies the behavior of parameter def rich defaults."""
  p1: ParameterDef = ParameterDef(name="d", default=1)
  assert getattr(p1, "default") == 1
  p2: ParameterDef = ParameterDef(name="flag", default=False)
  assert getattr(p2, "default") is False
  p3: ParameterDef = ParameterDef(name="eps", default=1e-05)
  assert getattr(p3, "default") == 1e-05
  p4: ParameterDef = ParameterDef(name="pads", default=[0, 0])
  assert getattr(p4, "default") == [0, 0]


def test_framework_variant_inject_args_rich_types() -> None:
  """Verifies the behavior of framework variant inject arguments rich types."""
  v: FrameworkVariant = FrameworkVariant(
    api="foo", inject_args={"val": 1.5, "flag": False, "dims": [1, 2], "data": {"a": 1}}
  )
  assert getattr(v, "inject_args")["dims"] == [1, 2]


def test_operation_def_structure() -> None:
  """Verifies the behavior of operation def structure."""
  data: Dict[str, Any] = {
    "operation": "TestOp",
    "description": "A test op",
    "std_args": [{"name": "x", "type": "int", "default": 0}],
    "variants": {"torch": {"api": "torch.test"}},
  }
  op: OperationDef = OperationDef(**data)
  assert getattr(getattr(op, "std_args")[0], "default") == 0
