"""Test suite for the Dsl Verification module."""

import pytest
from pydantic import ValidationError

from ml_switcheroo.core.dsl import OperationDef


def test_verification_mode_default() -> None:
  """Verifies the behavior of verification mode default."""
  op: OperationDef = OperationDef(operation="Add", description="Addition", std_args=[], variants={})
  assert getattr(op, "verification_mode") == "approx"


def test_verification_mode_exact() -> None:
  """Verifies the behavior of verification mode exact."""
  op: OperationDef = OperationDef(
    operation="IsNan", description="Check nan", std_args=[], variants={}, verification_mode="exact"
  )
  assert getattr(op, "verification_mode") == "exact"


def test_verification_mode_invalid() -> None:
  """Verifies the behavior of verification mode invalid."""
  with pytest.raises(ValidationError):
    OperationDef(operation="Bad", description="Bad", std_args=[], variants={}, verification_mode="loose")
