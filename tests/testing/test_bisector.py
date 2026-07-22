"""Test suite for the Bisector module."""

from unittest.mock import MagicMock
from ml_switcheroo.testing.bisector import SemanticsBisector
from ml_switcheroo.testing.runner import EquivalenceRunner


def test_propose_fix_relaxes_tolerances():
  """Verifies the behavior of propose fix relaxes tolerances."""
  runner = MagicMock(spec=EquivalenceRunner)
  runner.verify.side_effect = [(False, "Fail"), (False, "Fail"), (True, "Pass")]
  bisector = SemanticsBisector(runner)
  op_def = {"std_args": ["x", {"name": "y"}], "variants": {"a": {}}, "test_rtol": 1e-05}
  patch = bisector.propose_fix("MyOp", op_def)
  assert patch is not None
  assert patch["test_rtol"] == 0.01
  assert patch["test_atol"] == 0.001


def test_propose_fix_returns_none_if_no_relaxation_helps():
  """Verifies the behavior of propose fix returns none if no relaxation helps."""
  runner = MagicMock(spec=EquivalenceRunner)
  runner.verify.return_value = (False, "Fail")
  bisector = SemanticsBisector(runner)
  op_def = {"std_args": [("x", "int"), {"name": "z", "min": 0}], "variants": {"a": {}}}
  patch = bisector.propose_fix("MyOp", op_def)
  assert patch is None


def test_propose_fix_returns_none_if_matches_original():
  """Verifies the behavior of propose fix returns none if matches original."""
  runner = MagicMock(spec=EquivalenceRunner)
  runner.verify.return_value = (True, "Pass")
  bisector = SemanticsBisector(runner)
  op_def = {"std_args": ["x"], "variants": {"a": {}}, "test_rtol": 0.001, "test_atol": 0.0001}
  patch = bisector.propose_fix("MyOp", op_def)
  assert patch is None
