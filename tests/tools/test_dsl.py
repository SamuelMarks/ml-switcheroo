"""
Tests for ODL DSL Schema.
"""

import pytest
from pydantic import ValidationError
from ml_switcheroo.core.dsl import (
  OperationDef,
  ParameterDef,
  FrameworkVariant,
)


def test_parameter_def_rich_defaults():
  """
  Verify 'default' field accepts various types (int, float, list, bool).
  """
  p1 = ParameterDef(name="d", default=1)
  assert p1.default == 1

  p2 = ParameterDef(name="flag", default=False)
  assert p2.default is False

  p3 = ParameterDef(name="eps", default=1e-5)
  assert p3.default == 1e-5

  p4 = ParameterDef(name="pads", default=[0, 0])
  assert p4.default == [0, 0]


def test_framework_variant_inject_args_rich_types():
  """Verify 'inject_args' accepts complex types."""
  v = FrameworkVariant(api="foo", inject_args={"val": 1.5, "flag": False, "dims": [1, 2], "data": {"a": 1}})
  assert v.inject_args["dims"] == [1, 2]


def test_operation_def_structure():
  data = {
    "operation": "TestOp",
    "description": "A test op",
    "std_args": [{"name": "x", "type": "int", "default": 0}],
    "variants": {"torch": {"api": "torch.test"}},
  }
  op = OperationDef(**data)
  assert op.std_args[0].default == 0
