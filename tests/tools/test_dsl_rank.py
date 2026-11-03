"""Test suite for the Dsl Rank module."""

from tests.conftest import _test_getattr

import pytest
from pydantic import ValidationError

from ml_switcheroo.core.dsl import FrameworkVariant, OperationDef, ParameterDef


def test_param_rank_field_initialization() -> None:
  """Verifies the behavior of parameter rank field initialization."""
  p = ParameterDef(name="input_tensor", rank=4)
  assert _test_getattr(p, "rank") == 4
  assert _test_getattr(p, "name") == "input_tensor"


def test_param_rank_default_is_none() -> None:
  """Verifies the behavior of parameter rank default is none."""
  p = ParameterDef(name="x")
  assert _test_getattr(p, "rank") is None


def test_param_rank_type_validation() -> None:
  """Verifies the behavior of parameter rank type validation."""
  p = ParameterDef.model_validate({"name": "x", "rank": "3"})
  assert p.rank == 3
  with pytest.raises(ValidationError) as excinfo:
    ParameterDef.model_validate({"name": "x", "rank": "four"})
  assert "rank" in str(excinfo.value)


def test_integration_in_operation_def() -> None:
  """Verifies the behavior of integration in operation def."""
  conv_op = OperationDef(
    operation="Conv2d",
    description="2D Convolution",
    std_args=[ParameterDef(name="input", type="Tensor", rank=4), ParameterDef(name="weight", type="Tensor", rank=4)],
    variants={"torch": FrameworkVariant(api="torch.nn.functional.conv2d")},
  )
  assert len(_test_getattr(conv_op, "std_args")) == 2
  assert _test_getattr(_test_getattr(conv_op, "std_args")[0], "name") == "input"
  assert _test_getattr(_test_getattr(conv_op, "std_args")[0], "rank") == 4
  assert _test_getattr(_test_getattr(conv_op, "std_args")[1], "rank") == 4


def test_rank_serialization_roundtrip() -> None:
  """Verifies the behavior of rank serialization roundtrip."""
  original = ParameterDef(name="x", rank=5)
  json_str = original.model_dump_json()
  restored = ParameterDef.model_validate_json(json_str)
  assert _test_getattr(restored, "rank") == 5
  assert _test_getattr(restored, "name") == "x"
