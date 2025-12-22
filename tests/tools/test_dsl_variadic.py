"""
Tests for ODL Schema Extension: Variadic Arguments support.
Corresponds to Limitation #8 in the Architectural roadmap.

Verifies:
1. ParameterDef supports `is_variadic` field.
2. Defaults are False.
3. Integration within `OperationDef` for defining concepts like `max(*args)`.
"""

import pytest
from ml_switcheroo.core.dsl import ParameterDef, OperationDef, FrameworkVariant


def test_parameter_variadic_defaults():
  """
  Verify 'is_variadic' defaults to False for standard arguments.
  """
  p = ParameterDef(name="x")
  assert p.name == "x"
  assert p.is_variadic is False
  assert p.kind == "positional_or_keyword"


def test_parameter_variadic_explicit():
  """
  Verify 'is_variadic' can be set to True.
  """
  p = ParameterDef(name="tensors", is_variadic=True)
  assert p.is_variadic is True
  assert p.name == "tensors"


def test_variadic_integration_in_op_def():
  """
  Verify that an operation like `max` or `cat` can define variadic inputs.
  """
  # Simulating: torch.cat(tensors, dim=0) -> mapped as cat(*tensors, dim=0)?
  # Actually torch.cat takes a list, but max(*args) takes varargs.

  op = OperationDef(
    operation="MaxVariadic",
    description="Elementwise max of variable number of tensors",
    std_args=[
      ParameterDef(name="args", is_variadic=True, type="Tensor"),
      ParameterDef(name="out", type="Tensor", default="None"),
    ],
    variants={
      "torch": FrameworkVariant(api="torch.maximum")  # Hypothetical mapping
    },
  )

  assert len(op.std_args) == 2

  # Check variadic param
  v_param = op.std_args[0]
  assert v_param.name == "args"
  assert v_param.is_variadic is True

  # Check normal param
  n_param = op.std_args[1]
  assert n_param.name == "out"
  assert n_param.is_variadic is False


def test_parameter_kind_field():
  """
  Verify the new 'kind' field allows defining positional-only args.
  """
  p = ParameterDef(name="x", kind="positional_only")
  assert p.kind == "positional_only"
