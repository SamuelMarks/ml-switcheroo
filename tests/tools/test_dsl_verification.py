"""Test suite for the Dsl Verification module."""

import pytest
from pydantic import ValidationError

from ml_switcheroo.core.dsl import OperationDef


def test_verification_mode_default() -> None:
  """Verify `OperationDef` handles verification mode default as expected."""
  op = OperationDef(operation="Add", description="Addition", std_args=[], variants={})
  assert op.verification_mode == "approx"


def test_verification_mode_exact() -> None:
  """Verify `OperationDef` handles verification mode exact as expected."""
  op = OperationDef(operation="IsNan", description="Check nan", std_args=[], variants={}, verification_mode="exact")
  assert op.verification_mode == "exact"


def test_verification_mode_invalid() -> None:
  """Verify `OperationDef.model_validate` raises `ValidationError` during verification mode invalid."""
  with pytest.raises(ValidationError):
    OperationDef.model_validate(
      {"operation": "Bad", "description": "Bad", "std_args": [], "variants": {}, "verification_mode": "loose"}
    )
