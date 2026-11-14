"""Test suite for the Bisector module."""

from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue

from ml_switcheroo.testing.bisector import SemanticsBisector
from ml_switcheroo.testing.runner import EquivalenceRunner


def test_propose_fix_relaxes_tolerances() -> None:
  """Verify `SemanticsBisector` handles propose fix relaxes tolerances as expected."""
  runner: MagicMock = MagicMock(spec=EquivalenceRunner)
  runner.verify.side_effect = [(False, "Fail"), (False, "Fail"), (True, "Pass")]
  bisector: SemanticsBisector = SemanticsBisector(runner)
  op_def: dict[str, JsonValue] = {"std_args": ["x", {"name": "y"}], "variants": {"a": {}}, "test_rtol": 1e-05}
  patch = bisector.propose_fix("MyOp", op_def)
  assert patch is not None
  assert patch["test_rtol"] == 0.01
  assert patch["test_atol"] == 0.001


def test_propose_fix_returns_none_if_no_relaxation_helps() -> None:
  """Verify `SemanticsBisector` handles propose fix returns none if no relaxation helps as expected."""
  runner: MagicMock = MagicMock(spec=EquivalenceRunner)
  runner.verify.return_value = (False, "Fail")
  bisector: SemanticsBisector = SemanticsBisector(runner)
  op_def: dict[str, JsonValue] = {"std_args": [["x", "int"], {"name": "z", "min": 0}], "variants": {"a": {}}}
  patch = bisector.propose_fix("MyOp", op_def)
  assert patch is None


def test_propose_fix_returns_none_if_matches_original() -> None:
  """Verify `SemanticsBisector` handles propose fix returns none if matches original as expected."""
  runner: MagicMock = MagicMock(spec=EquivalenceRunner)
  runner.verify.return_value = (True, "Pass")
  bisector: SemanticsBisector = SemanticsBisector(runner)
  op_def: dict[str, JsonValue] = {"std_args": ["x"], "variants": {"a": {}}, "test_rtol": 0.001, "test_atol": 0.0001}
  patch = bisector.propose_fix("MyOp", op_def)
  assert patch is None


def test_bisector_extract_params(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `SemanticsBisector` handles bisector extract params as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  from ml_switcheroo.testing.bisector import SemanticsBisector

  bisector = SemanticsBisector(None)
  op_def: dict[str, JsonValue] = {
    "std_args": [{"name": "a", "type": "int", "min": 0}, ["b", "float"], ["c"], "d"],
    "test_rtol": 1e-10,
  }
  from unittest import mock

  with mock.patch("ml_switcheroo.testing.bisector.EquivalenceRunner.verify", return_value=(True, "OK")):

    class MockRunner:
      """Test double `MockRunner` implementing `verify` for bisector tests."""

      def verify(self, *args: JsonValue, **kwargs: JsonValue) -> tuple[bool, str]:
        """Mock implementation of verify.

        Args:
            args: Positional argument specifications (`JsonValue`).
            kwargs: Keyword argument bindings (`JsonValue`).

        Returns:
            Resulting `tuple[bool, str]` from verify.
        """
        assert kwargs.get("hints") == {"a": "int", "b": "float"}
        return (True, "OK")

    monkeypatch.setattr(
      bisector,
      "runner",
      MockRunner(),
      raising=False,
    )
    res = bisector.propose_fix("foo", op_def)
    assert res is not None


def test_bisector_runner_exception(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `SemanticsBisector` handles bisector runner exception as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  from ml_switcheroo.testing.bisector import SemanticsBisector

  bisector = SemanticsBisector(None)
  op_def: dict[str, JsonValue] = {"std_args": ["a"]}

  class MockRunnerEx:
    """Test double `MockRunnerEx` implementing `verify` for bisector tests."""

    def verify(self, *args: JsonValue, **kwargs: JsonValue) -> tuple[bool, str]:
      """Mock implementation of verify.

      Args:
          args: Positional argument specifications (`JsonValue`).
          kwargs: Keyword argument bindings (`JsonValue`).

      Returns:
          Resulting `tuple[bool, str]` from verify.
      """
      raise ValueError("Runner died")

  monkeypatch.setattr(
    bisector,
    "runner",
    MockRunnerEx(),
    raising=False,
  )
  res = bisector.propose_fix("foo", op_def)
  assert res is None


def test_bisector_no_fix_needed(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `SemanticsBisector` handles bisector no fix needed as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  from ml_switcheroo.testing.bisector import SemanticsBisector

  bisector = SemanticsBisector(None)
  op_def: dict[str, JsonValue] = {"std_args": ["a"]}

  class MockRunnerOk:
    """Test double `MockRunnerOk` implementing `verify` for bisector tests."""

    def verify(self, *args: JsonValue, **kwargs: JsonValue) -> tuple[bool, str]:
      """Mock implementation of verify.

      Args:
          args: Positional argument specifications (`JsonValue`).
          kwargs: Keyword argument bindings (`JsonValue`).

      Returns:
          Resulting `tuple[bool, str]` from verify.
      """
      return (True, "OK")

  monkeypatch.setattr(
    bisector,
    "runner",
    MockRunnerOk(),
    raising=False,
  )
  res = bisector.propose_fix("foo", op_def)
  assert res is None


def test_bisector_fix_found(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `SemanticsBisector` handles bisector fix found as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  from ml_switcheroo.testing.bisector import SemanticsBisector

  bisector = SemanticsBisector(None)
  op_def: dict[str, JsonValue] = {"std_args": ["a"]}

  class MockRunner:
    """Test double `MockRunner` implementing `verify` for bisector tests."""

    def verify(self, *args: JsonValue, **kwargs: JsonValue) -> tuple[bool, str]:
      """Mock implementation of verify.

      Args:
          args: Positional argument specifications (`JsonValue`).
          kwargs: Keyword argument bindings (`JsonValue`).

      Returns:
          Resulting `tuple[bool, str]` from verify.
      """
      return (True, "OK")

  monkeypatch.setattr(
    bisector,
    "runner",
    MockRunner(),
    raising=False,
  )
  op_def["test_rtol"] = 1e-09
  res = bisector.propose_fix("foo", op_def)
  assert res is not None
  assert res["test_rtol"] == 0.001


def test_bisector_exception(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `SemanticsBisector` handles bisector exception as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  from ml_switcheroo.testing.bisector import SemanticsBisector

  bisector = SemanticsBisector(None)
  op_def: dict[str, JsonValue] = {"std_args": ["a"]}

  class MockRunner:
    """Test double `MockRunner` implementing `verify` for bisector tests."""

    def verify(self, *args: JsonValue, **kwargs: JsonValue) -> tuple[bool, str]:
      """Mock implementation of verify.

      Args:
          args: Positional argument specifications (`JsonValue`).
          kwargs: Keyword argument bindings (`JsonValue`).

      Returns:
          Resulting `tuple[bool, str]` from verify.
      """
      raise RuntimeError("fail")

  monkeypatch.setattr(
    bisector,
    "runner",
    MockRunner(),
    raising=False,
  )
  res = bisector.propose_fix("foo", op_def)
  assert res is None


def test_bisector_unsupported_arg_type() -> None:
  """Test propose_fix with unsupported argument type in std_args to cover branch 96->76."""
  runner: MagicMock = MagicMock(spec=EquivalenceRunner)
  runner.verify.return_value = (True, "Pass")
  bisector: SemanticsBisector = SemanticsBisector(runner)
  op_def: dict[str, JsonValue] = {"std_args": [12345], "variants": {"a": {}}, "test_rtol": 0.001, "test_atol": 0.0001}
  patch = bisector.propose_fix("MyOp", op_def)
  assert patch is None
