"""Test suite for the Bisector module."""

from pydantic import JsonValue
from unittest.mock import MagicMock

from ml_switcheroo.testing.bisector import SemanticsBisector
from ml_switcheroo.testing.runner import EquivalenceRunner


def test_propose_fix_relaxes_tolerances() -> None:
  """Verifies the behavior of propose fix relaxes tolerances."""
  runner: MagicMock = MagicMock(spec=EquivalenceRunner)
  runner.verify.side_effect = [(False, "Fail"), (False, "Fail"), (True, "Pass")]
  bisector: SemanticsBisector = SemanticsBisector(runner)
  op_def = {"std_args": ["x", {"name": "y"}], "variants": {"a": {}}, "test_rtol": 1e-05}
  patch = getattr(
    {
      "v": getattr({"v": getattr({"v": bisector}, "__getitem__")("v").propose_fix}, "__getitem__")("v")(
        getattr({"v": "MyOp"}, "__getitem__")("v"), getattr({"v": op_def}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert patch is not None
  assert patch["test_rtol"] == 0.01
  assert patch["test_atol"] == 0.001


def test_propose_fix_returns_none_if_no_relaxation_helps() -> None:
  """Verifies the behavior of propose fix returns none if no relaxation helps."""
  runner: MagicMock = MagicMock(spec=EquivalenceRunner)
  runner.verify.return_value = (False, "Fail")
  bisector: SemanticsBisector = SemanticsBisector(runner)
  op_def = {"std_args": [("x", "int"), {"name": "z", "min": 0}], "variants": {"a": {}}}
  patch = getattr(
    {
      "v": getattr({"v": getattr({"v": bisector}, "__getitem__")("v").propose_fix}, "__getitem__")("v")(
        getattr({"v": "MyOp"}, "__getitem__")("v"), getattr({"v": op_def}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert patch is None


def test_propose_fix_returns_none_if_matches_original() -> None:
  """Verifies the behavior of propose fix returns none if matches original."""
  runner: MagicMock = MagicMock(spec=EquivalenceRunner)
  runner.verify.return_value = (True, "Pass")
  bisector: SemanticsBisector = SemanticsBisector(runner)
  op_def = {"std_args": ["x"], "variants": {"a": {}}, "test_rtol": 0.001, "test_atol": 0.0001}
  patch = getattr(
    {
      "v": getattr({"v": getattr({"v": bisector}, "__getitem__")("v").propose_fix}, "__getitem__")("v")(
        getattr({"v": "MyOp"}, "__getitem__")("v"), getattr({"v": op_def}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert patch is None


def test_bisector_extract_params() -> None:
  """Verify bisector extract params behavior."""
  from ml_switcheroo.testing.bisector import SemanticsBisector

  bisector = SemanticsBisector(None)
  op_def = {
    "std_args": [{"name": "a", "type": "int", "min": 0}, ["b", "float"], ["c"], "d"],
    "test_rtol": 1e-10,
  }
  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.testing.bisector.EquivalenceRunner.verify", return_value=(True, "OK")
  ):

    class MockRunner:
      """Test helper class MockRunner."""

      def verify(self, *args: JsonValue, **kwargs: JsonValue) -> tuple[bool, str]:
        """Mock implementation of verify.

        Args:
            args: The args (JsonValue) for verify.
            kwargs: The kwargs (JsonValue) for verify.

        Returns:
            Resulting `tuple[bool, str]` from verify.
        """
        assert kwargs.get("hints") == {"a": "int", "b": "float"}
        return (True, "OK")

    setattr(
      getattr({"v": bisector}, "__getitem__")("v"),
      "runner",
      getattr({"v": getattr({"v": MockRunner}, "__getitem__")("v")()}, "__getitem__")("v"),
    )
    res = getattr(
      {
        "v": getattr({"v": getattr({"v": bisector}, "__getitem__")("v").propose_fix}, "__getitem__")("v")(
          getattr({"v": "foo"}, "__getitem__")("v"), getattr({"v": op_def}, "__getitem__")("v")
        )
      },
      "__getitem__",
    )("v")
    assert res is not None


def test_bisector_runner_exception() -> None:
  """Verifies the behavior of bisector runner correctly handling an exception."""
  from ml_switcheroo.testing.bisector import SemanticsBisector

  bisector = SemanticsBisector(None)
  op_def = {"std_args": ["a"]}

  class MockRunnerEx:
    """Test helper class MockRunnerEx."""

    def verify(self, *args: JsonValue, **kwargs: JsonValue) -> tuple[bool, str]:
      """Mock implementation of verify.

      Args:
          args: The args (JsonValue) for verify.
          kwargs: The kwargs (JsonValue) for verify.

      Returns:
          Resulting `tuple[bool, str]` from verify.
      """
      raise ValueError("Runner died")

  setattr(
    getattr({"v": bisector}, "__getitem__")("v"),
    "runner",
    getattr({"v": getattr({"v": MockRunnerEx}, "__getitem__")("v")()}, "__getitem__")("v"),
  )
  res = getattr(
    {
      "v": getattr({"v": getattr({"v": bisector}, "__getitem__")("v").propose_fix}, "__getitem__")("v")(
        getattr({"v": "foo"}, "__getitem__")("v"), getattr({"v": op_def}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert res is None


def test_bisector_no_fix_needed() -> None:
  """Verifies the behavior of bisector no fix needed."""
  from ml_switcheroo.testing.bisector import SemanticsBisector

  bisector = SemanticsBisector(None)
  op_def = {"std_args": ["a"]}

  class MockRunnerOk:
    """Test helper class MockRunnerOk."""

    def verify(self, *args: JsonValue, **kwargs: JsonValue) -> tuple[bool, str]:
      """Mock implementation of verify.

      Args:
          args: The args (JsonValue) for verify.
          kwargs: The kwargs (JsonValue) for verify.

      Returns:
          Resulting `tuple[bool, str]` from verify.
      """
      return (True, "OK")

  setattr(
    getattr({"v": bisector}, "__getitem__")("v"),
    "runner",
    getattr({"v": getattr({"v": MockRunnerOk}, "__getitem__")("v")()}, "__getitem__")("v"),
  )
  res = getattr(
    {
      "v": getattr({"v": getattr({"v": bisector}, "__getitem__")("v").propose_fix}, "__getitem__")("v")(
        getattr({"v": "foo"}, "__getitem__")("v"), getattr({"v": op_def}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert res is None


def test_bisector_fix_found() -> None:
  """Verifies the behavior of bisector fix found."""
  from ml_switcheroo.testing.bisector import SemanticsBisector

  bisector = SemanticsBisector(None)
  op_def = {"std_args": ["a"]}

  class MockRunner:
    """Test helper class MockRunner."""

    def verify(self, *args: JsonValue, **kwargs: JsonValue) -> tuple[bool, str]:
      """Mock implementation of verify.

      Args:
          args: The args (JsonValue) for verify.
          kwargs: The kwargs (JsonValue) for verify.

      Returns:
          Resulting `tuple[bool, str]` from verify.
      """
      return (True, "OK")

  setattr(
    getattr({"v": bisector}, "__getitem__")("v"),
    "runner",
    getattr({"v": getattr({"v": MockRunner}, "__getitem__")("v")()}, "__getitem__")("v"),
  )
  getattr(getattr({"v": op_def}, "__getitem__")("v"), "__setitem__")(
    "test_rtol", getattr({"v": 1e-09}, "__getitem__")("v")
  )
  res = getattr(
    {
      "v": getattr({"v": getattr({"v": bisector}, "__getitem__")("v").propose_fix}, "__getitem__")("v")(
        getattr({"v": "foo"}, "__getitem__")("v"), getattr({"v": op_def}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert res is not None
  assert res["test_rtol"] == 0.001


def test_bisector_exception() -> None:
  """Verifies the behavior of bisector correctly handling an exception."""
  from ml_switcheroo.testing.bisector import SemanticsBisector

  bisector = SemanticsBisector(None)
  op_def = {"std_args": ["a"]}

  class MockRunner:
    """Test helper class MockRunner."""

    def verify(self, *args: JsonValue, **kwargs: JsonValue) -> tuple[bool, str]:
      """Mock implementation of verify.

      Args:
          args: The args (JsonValue) for verify.
          kwargs: The kwargs (JsonValue) for verify.

      Returns:
          Resulting `tuple[bool, str]` from verify.
      """
      raise Exception("fail")

  setattr(
    getattr({"v": bisector}, "__getitem__")("v"),
    "runner",
    getattr({"v": getattr({"v": MockRunner}, "__getitem__")("v")()}, "__getitem__")("v"),
  )
  res = getattr(
    {
      "v": getattr({"v": getattr({"v": bisector}, "__getitem__")("v").propose_fix}, "__getitem__")("v")(
        getattr({"v": "foo"}, "__getitem__")("v"), getattr({"v": op_def}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert res is None


def test_bisector_unsupported_arg_type() -> None:
  """Test propose_fix with unsupported argument type in std_args to cover branch 96->76."""
  runner: MagicMock = MagicMock(spec=EquivalenceRunner)
  runner.verify.return_value = (True, "Pass")
  bisector: SemanticsBisector = SemanticsBisector(runner)
  op_def = {"std_args": [12345], "variants": {"a": {}}, "test_rtol": 0.001, "test_atol": 0.0001}
  patch = getattr(
    {
      "v": getattr({"v": getattr({"v": bisector}, "__getitem__")("v").propose_fix}, "__getitem__")("v")(
        getattr({"v": "MyOp"}, "__getitem__")("v"), getattr({"v": op_def}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert patch is None
