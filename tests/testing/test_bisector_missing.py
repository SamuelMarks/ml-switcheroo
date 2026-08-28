"""Test suite for the Bisector Missing module."""

from typing import Dict, Any, Optional, Tuple


def test_bisector_extract_params() -> None:
  """Verifies the behavior of bisector extract parameters."""
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
      """Mock Runner class for testing purposes."""

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
    """Mock Runner Ex class for testing purposes."""

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
    """Mock Runner Ok class for testing purposes."""

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
    """Mock Runner class for testing purposes."""

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
    """Mock Runner class for testing purposes."""

    def verify(self, *args: Any, **kwargs: Any) -> Tuple[bool, str]:
      """Mock implementation of verify."""
      raise Exception("fail")

  bisector.runner = MockRunner()
  res: Optional[Dict[str, Any]] = bisector.propose_fix("foo", op_def)
  assert res is None
