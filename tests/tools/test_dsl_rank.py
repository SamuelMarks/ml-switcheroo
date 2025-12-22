"""
Tests for ODL Schema Extension: Tensor Rank Constraints.
Corresponds to Limitation #1 in the Architectural roadmap.
"""

import pytest
from pydantic import ValidationError
from ml_switcheroo.core.dsl import ParameterDef, OperationDef, FrameworkVariant


def test_param_rank_field_initialization():
  """
  Verify that the 'rank' field can be initialized explicitly.
  This enables the Fuzzer to generate tensors with fixed dimensions (e.g. 4 for images).
  """
  # Case: Explicit Rank
  p = ParameterDef(name="input_tensor", rank=4)
  assert p.rank == 4
  assert p.name == "input_tensor"


def test_param_rank_default_is_none():
  """
  Verify 'rank' defaults to None (arbitrary rank).
  This ensures backward compatibility with existing definitions.
  """
  p = ParameterDef(name="x")
  assert p.rank is None


def test_param_rank_type_validation():
  """
  Verify that 'rank' enforces integer types.
  """
  # Valid string casting
  p = ParameterDef(name="x", rank="3")
  assert p.rank == 3

  # Invalid type
  with pytest.raises(ValidationError) as excinfo:
    ParameterDef(name="x", rank="four")

  assert "rank" in str(excinfo.value)


def test_integration_in_operation_def():
  """
  Verify that ParameterDef with rank integrates correctly into the top-level OperationDef.
  Simulates defining a Conv2d operation which requires 4D input.
  """
  conv_op = OperationDef(
    operation="Conv2d",
    description="2D Convolution",
    std_args=[ParameterDef(name="input", type="Tensor", rank=4), ParameterDef(name="weight", type="Tensor", rank=4)],
    variants={"torch": FrameworkVariant(api="torch.nn.functional.conv2d")},
  )

  # Verify persistence
  assert len(conv_op.std_args) == 2
  assert conv_op.std_args[0].name == "input"
  assert conv_op.std_args[0].rank == 4
  assert conv_op.std_args[1].rank == 4


def test_rank_serialization_roundtrip():
  """
  Verify that rank metadata survives JSON serialization/deserialization cycles.
  This is critical for saving/loading the Knowledge Base.
  """
  original = ParameterDef(name="x", rank=5)

  # 1. Serialize
  json_str = original.model_dump_json()

  # 2. Deserialize
  restored = ParameterDef.model_validate_json(json_str)

  assert restored.rank == 5
  assert restored.name == "x"
