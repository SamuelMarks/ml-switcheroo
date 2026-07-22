"""Test suite for the Dsl Verification module."""

import pytest
from pydantic import ValidationError
from ml_switcheroo.core.dsl import OperationDef


def test_verification_mode_default():
  """Verifies the behavior of verification mode default."""
  op = OperationDef(operation="Add", description="Addition", std_args=[], variants={})
  assert op.verification_mode == "approx"


def test_verification_mode_exact():
  """Verifies the behavior of verification mode exact."""
  op = OperationDef(operation="IsNan", description="Check nan", std_args=[], variants={}, verification_mode="exact")
  assert op.verification_mode == "exact"


def test_verification_mode_invalid():
  """Verifies the behavior of verification mode invalid."""
  with pytest.raises(ValidationError):
    OperationDef(operation="Bad", description="Bad", std_args=[], variants={}, verification_mode="loose")
