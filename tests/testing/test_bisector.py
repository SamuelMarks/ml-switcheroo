"""Test suite for the Bisector module."""

from typing import Any, Dict, Optional, Tuple
from unittest.mock import MagicMock

from ml_switcheroo.testing.bisector import SemanticsBisector
from ml_switcheroo.testing.runner import EquivalenceRunner


def test_propose_fix_relaxes_tolerances() -> None:
  """Verifies the behavior of propose fix relaxes tolerances."""
  runner: MagicMock = MagicMock(spec=EquivalenceRunner)
  runner.verify.side_effect = [(False, "Fail"), (False, "Fail"), (True, "Pass")]
  bisector: SemanticsBisector = SemanticsBisector(runner)
  op_def: Dict[str, Any] = {"std_args": ["x", {"name": "y"}], "variants": {"a": {}}, "test_rtol": 1e-05}
  patch: Optional[Dict[str, Any]] = bisector.propose_fix("MyOp", op_def)
  assert patch is not None
  assert patch["test_rtol"] == 0.01
  assert patch["test_atol"] == 0.001


def test_propose_fix_returns_none_if_no_relaxation_helps() -> None:
  """Verifies the behavior of propose fix returns none if no relaxation helps."""
  runner: MagicMock = MagicMock(spec=EquivalenceRunner)
  runner.verify.return_value = (False, "Fail")
  bisector: SemanticsBisector = SemanticsBisector(runner)
  op_def: Dict[str, Any] = {"std_args": [("x", "int"), {"name": "z", "min": 0}], "variants": {"a": {}}}
  patch: Optional[Dict[str, Any]] = bisector.propose_fix("MyOp", op_def)
  assert patch is None


def test_propose_fix_returns_none_if_matches_original() -> None:
  """Verifies the behavior of propose fix returns none if matches original."""
  runner: MagicMock = MagicMock(spec=EquivalenceRunner)
  runner.verify.return_value = (True, "Pass")
  bisector: SemanticsBisector = SemanticsBisector(runner)
  op_def: Dict[str, Any] = {"std_args": ["x"], "variants": {"a": {}}, "test_rtol": 0.001, "test_atol": 0.0001}
  patch: Optional[Dict[str, Any]] = bisector.propose_fix("MyOp", op_def)
  assert patch is None


# --- Merged from test_bisector_missing.py ---


def test_bisector_extract_params() -> None:
  """Docstring."""
  from ml_switcheroo.testing.bisector import SemanticsBisector

  bisector: SemanticsBisector = SemanticsBisector(None)
  op_def: Dict[str, Any] = {
    "std_args": [{"name": "a", "type": "int", "min": 0}, ["b", "float"], ["c"], "d"],
    "test_rtol": 1e-10,
  }
  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.testing.bisector.EquivalenceRunner.verify", return_value=(True, "OK")
  ):

    class MockRunner:
      def verify(self, *args: Any, **kwargs: Any) -> Tuple[bool, str]:
        """Mock implementation of verify."""
        assert kwargs.get("hints") == {"a": "int", "b": "float"}
        return (True, "OK")

    bisector.runner = MockRunner()
    res: Optional[Dict[str, Any]] = bisector.propose_fix("foo", op_def)
    assert res is not None


def test_bisector_runner_exception() -> None:
  """Verifies the behavior of bisector runner correctly handling an exception."""
  from ml_switcheroo.testing.bisector import SemanticsBisector

  bisector: SemanticsBisector = SemanticsBisector(None)
  op_def: Dict[str, Any] = {"std_args": ["a"]}

  class MockRunnerEx:
    def verify(self, *args: Any, **kwargs: Any) -> Tuple[bool, str]:
      """Mock implementation of verify."""
      raise ValueError("Runner died")

  bisector.runner = MockRunnerEx()
  res: Optional[Dict[str, Any]] = bisector.propose_fix("foo", op_def)
  assert res is None


def test_bisector_no_fix_needed() -> None:
  """Verifies the behavior of bisector no fix needed."""
  from ml_switcheroo.testing.bisector import SemanticsBisector

  bisector: SemanticsBisector = SemanticsBisector(None)
  op_def: Dict[str, Any] = {"std_args": ["a"]}

  class MockRunnerOk:
    def verify(self, *args: Any, **kwargs: Any) -> Tuple[bool, str]:
      """Mock implementation of verify."""
      return (True, "OK")

  bisector.runner = MockRunnerOk()
  res: Optional[Dict[str, Any]] = bisector.propose_fix("foo", op_def)
  assert res is None


def test_bisector_fix_found() -> None:
  """Verifies the behavior of bisector fix found."""
  from ml_switcheroo.testing.bisector import SemanticsBisector

  bisector: SemanticsBisector = SemanticsBisector(None)
  op_def: Dict[str, Any] = {"std_args": ["a"]}

  class MockRunner:
    def verify(self, *args: Any, **kwargs: Any) -> Tuple[bool, str]:
      """Mock implementation of verify."""
      return (True, "OK")

  bisector.runner = MockRunner()
  op_def["test_rtol"] = 1e-09
  res: Optional[Dict[str, Any]] = bisector.propose_fix("foo", op_def)
  assert res is not None
  assert res["test_rtol"] == 0.001


def test_bisector_exception() -> None:
  """Verifies the behavior of bisector correctly handling an exception."""
  from ml_switcheroo.testing.bisector import SemanticsBisector

  bisector: SemanticsBisector = SemanticsBisector(None)
  op_def: Dict[str, Any] = {"std_args": ["a"]}

  class MockRunner:
    def verify(self, *args: Any, **kwargs: Any) -> Tuple[bool, str]:
      """Mock implementation of verify."""
      raise Exception("fail")

  bisector.runner = MockRunner()
  res: Optional[Dict[str, Any]] = bisector.propose_fix("foo", op_def)
  assert res is None
