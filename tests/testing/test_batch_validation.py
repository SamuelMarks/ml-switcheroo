"""Test suite for the Batch Validation module."""

import typing
from pydantic import JsonValue
import pathlib
from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.testing.batch_runner import BatchValidator


@pytest.fixture
def mock_semantics() -> MagicMock:
  """Execute mock semantics helper logic.

  Returns:
      Computed test result or fixture value.
  """
  mgr: MagicMock = MagicMock(spec=SemanticsManager)
  kb = {
    "auto_op": {"std_args": [("x", "int")], "variants": {"torch": {"api": "t.op"}}},
    "broken_op": {"std_args": ["x"], "variants": {"torch": {"api": "t.break"}}},
    "manual_op": {"std_args": ["x"], "variants": {}},
    "shape_op": {"std_args": ["x"], "output_shape_calc": "lambda x: x.shape", "variants": {"torch": {"api": "shape.op"}}},
  }
  mgr.get_known_apis.return_value = kb
  return mgr


@pytest.fixture
def validator(mock_semantics: MagicMock) -> BatchValidator:
  """Execute validator helper logic.

  Args:
      mock_semantics: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return BatchValidator(mock_semantics)


def test_batch_execution_flow(validator: BatchValidator) -> None:
  """Verifies the behavior of batch execution flow.

  Args:
      validator: Test argument or fixture.
  """

  def mock_verify(
    variants: dict[str, JsonValue],
    params: typing.List[str],
    hints: typing.Optional[typing.Dict[str, str]] = None,
    constraints: typing.Optional[typing.Dict[str, typing.Dict[str, JsonValue]]] = None,
    shape_calc: typing.Optional[str] = None,
  ) -> typing.Tuple[bool, str]:
    """Execute mock verify helper logic.

    Args:
        variants: Test argument or fixture.
        params: Test argument or fixture.
        hints: Test argument or fixture.
        constraints: Test argument or fixture.
        shape_calc: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if not variants:
      return (True, "Skipped")
    api: str = list(variants.values())[0]["api"]  # type: ignore[call-overload, index, assignment]
    if api == "t.op":
      return (True, "OK")
    if api == "t.break":
      return (False, "Fail")
    if api == "shape.op":
      if shape_calc == "lambda x: x.shape":
        return (True, "Shape OK")
      return (False, "Missing Shape Calc")
    return (False, "Unknown")

  with patch.object(validator.runner, "verify", side_effect=mock_verify) as mock_run:
    results: typing.Dict[str, bool] = validator.run_all()
    assert results["auto_op"] is True
    assert results["broken_op"] is False
    assert results["shape_op"] is True
    assert results["manual_op"] is True
    assert mock_run.call_count == 4


def test_extraction_of_shape_calc(validator: BatchValidator) -> None:
  """Verify extraction of shape calc behavior.

  Args:
      validator: Test argument or fixture.
  """
  with patch.object(validator.runner, "verify", return_value=(True, "OK")) as mock_run:
    validator.run_all()
    found_shape_call: bool = False
    for call in mock_run.call_args_list:
      args, kwargs = call
      variants = args[0]
      if not variants:
        continue
      if variants["torch"]["api"] == "shape.op":
        assert kwargs["shape_calc"] == "lambda x: x.shape"
        found_shape_call = True
    assert found_shape_call


def test_manual_override_priority(validator: BatchValidator, tmp_path: pathlib.Path) -> None:
  """Verifies the behavior of manual override priority.

  Args:
      validator: Test argument or fixture.
      tmp_path: Test argument or fixture.
  """
  test_dir: pathlib.Path = tmp_path / "tests"
  test_dir.mkdir()
  (test_dir / "test_manual.py").write_text("def test_manual_op(): pass")

  def mock_verify(*args: JsonValue, **kwargs: JsonValue) -> typing.Tuple[bool, str]:
    """Execute mock verify helper logic.

    Args:
        args: Test argument or fixture.
        kwargs: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return (True, "OK")

  with patch.object(validator.runner, "verify", side_effect=mock_verify) as mock_run:
    results: typing.Dict[str, bool] = validator.run_all(manual_test_dir=tmp_path)
    assert results["manual_op"] is True
    assert mock_run.call_count == 3


def test_ignore_generated_tests(validator: BatchValidator, tmp_path: pathlib.Path) -> None:
  """Verify ignore generated tests behavior.

  Args:
      validator: Test argument or fixture.
      tmp_path: Test argument or fixture.
  """
  gen_dir: pathlib.Path = tmp_path / "generated"
  gen_dir.mkdir()
  (gen_dir / "test_gen_auto_op.py").write_text("def test_gen_auto_op(): pass")
  with patch.object(validator.runner, "verify", return_value=(True, "OK")) as mock_run:
    validator.run_all(manual_test_dir=tmp_path)
    assert mock_run.call_count == 4


def test_unpack_args_logic(validator: BatchValidator) -> None:
  """Verifies the behavior of unpack arguments logic.

  Args:
      validator: Test argument or fixture.
  """
  raw = [("x", "Array"), "axis", ("dims", "Tuple[int]")]
  params: typing.List[str]
  hints: typing.Dict[str, str]
  params, hints, constraints = validator._unpack_args(raw)
  assert params == ["x", "axis", "dims"]
  assert hints["x"] == "Array"
  assert hints["dims"] == "Tuple[int]"
