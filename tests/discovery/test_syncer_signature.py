"""
Tests for Signature Verification in FrameworkSyncer.

Ensures that the Syncer does not blindly link functions with incompatible
signatures (e.g., mismatching argument counts).
"""

import pytest
from ml_switcheroo.discovery.syncer import FrameworkSyncer


class MockSyncer(FrameworkSyncer):
  """Subclass to expose protected method for direct testing."""

  def check(self, func, std_args):
    return self._is_compatible(func, std_args)


@pytest.fixture
def syncer():
  return MockSyncer()


def test_compat_exact_match(syncer):
  """Spec: [x, y], Func: (a, b) -> Compatible."""

  def candidate(_a, _b):
    pass

  assert syncer.check(candidate, ["x", "y"]) is True


def test_compat_var_args(syncer):
  """Spec: [x, y, z], Func: (*args) -> Compatible."""

  def candidate(*_args):
    pass

  assert syncer.check(candidate, ["x", "y", "z"]) is True


def test_compat_extra_defaults(syncer):
  """Spec: [x], Func: (a, b=1) -> Compatible (b is optional)."""

  def candidate(_a, _b=1):
    pass

  assert syncer.check(candidate, ["x"]) is True


def test_incompatible_too_many_mandatory(syncer):
  """
  Spec: [x] (1 arg provided)
  Func: (a, b) (2 mandatory args required)
  result: Incompatible.
  """

  def candidate(_a, _b):
    pass

  assert syncer.check(candidate, ["x"]) is False


def test_incompatible_capacity_too_low(syncer):
  """
  Spec: [x, y] (2 args provided)
  Func: (a)    (Accepts max 1 arg)
  result: Incompatible.
  """

  def candidate(_a):
    pass

  assert syncer.check(candidate, ["x", "y"]) is False


def test_compat_builtins_fallback(syncer):
  """
  Some builtins raise ValueError on inspect.signature.
  We should assume compatibility to be safe (fail-open).
  """

  # Create a mock that raises ValueError on inspect
  class TrickyBuiltin:
    pass

  # We rely on inspect.signature(TrickyBuiltin) raising or failing.
  # range() is a class, inspect.signature(range) works in 3.x,
  # but some C-extension functions like 'math.uops' might fail.
  # We simulate the validation logic's exception handling by mocking?
  # Actually, let's use a lambda that takes parameters but we pass a non-function?

  # inspect.signature(1) raises TypeError
  assert syncer.check(1, ["x"]) is True
