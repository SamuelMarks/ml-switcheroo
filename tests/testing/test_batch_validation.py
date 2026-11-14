"""Test suite for the Batch Validation module."""

import pathlib
from unittest.mock import MagicMock, patch

import pytest
from pydantic import JsonValue

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.testing.batch_runner import BatchValidator


@pytest.fixture
def mock_semantics() -> MagicMock:
  """Create a configured `MagicMock` fixture for batch validation tests.

  Returns:
      Configured `mock_semantics` fixture value (MagicMock).
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
  """Create a configured `BatchValidator` fixture for batch validation tests.

  Args:
      mock_semantics: Input `mock_semantics` parameter value (`MagicMock`).

  Returns:
      Configured `validator` fixture value (BatchValidator).
  """
  return BatchValidator(mock_semantics)


def test_batch_execution_flow(validator: BatchValidator) -> None:
  """Verify `validator.run_all` returns `4` for batch execution flow.

  Args:
      validator: Input `validator` parameter value (`BatchValidator`).
  """

  def mock_verify(
    variants: dict[str, JsonValue],
    params: list[str],
    hints: dict[str, str] | None = None,
    constraints: dict[str, dict[str, JsonValue]] | None = None,
    shape_calc: str | None = None,
  ) -> tuple[bool, str]:
    """Execute `mock_verify` helper in batch validation tests.

    Args:
        variants: Per-framework operation variant mappings (`dict[str, JsonValue]`).
        params: Input `params` parameter value (`list[str]`).
        hints: Input `hints` parameter value (`dict[str, str] | None`).
        constraints: Input `constraints` parameter value (`dict[str, dict[str, JsonValue]] | None`).
        shape_calc: Input `shape_calc` parameter value (`str | None`).

    Returns:
        Resulting `tuple[bool, str]` from mock verify.
    """
    if not variants:
      return (True, "Skipped")
    first_var = next(iter(variants.values()))
    assert isinstance(first_var, dict)
    api = first_var.get("api")
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
    results: dict[str, bool] = validator.run_all()
    assert results["auto_op"] is True
    assert results["broken_op"] is False
    assert results["shape_op"] is True
    assert results["manual_op"] is True
    assert mock_run.call_count == 4


def test_extraction_of_shape_calc(validator: BatchValidator) -> None:
  """Verify `validator.run_all` handles extraction of shape calc as expected.

  Args:
      validator: Input `validator` parameter value (`BatchValidator`).
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
  """Verify `validator.run_all` returns `3` for manual override priority.

  Args:
      validator: Input `validator` parameter value (`BatchValidator`).
      tmp_path: Pytest temporary directory path fixture.
  """
  test_dir: pathlib.Path = tmp_path / "tests"
  test_dir.mkdir()
  (test_dir / "test_manual.py").write_text("def test_manual_op(): pass")

  def mock_verify(*args: JsonValue, **kwargs: JsonValue) -> tuple[bool, str]:
    """Execute `mock_verify` helper in batch validation tests.

    Args:
        args: Positional argument specifications (`JsonValue`).
        kwargs: Keyword argument bindings (`JsonValue`).

    Returns:
        Resulting `tuple[bool, str]` from mock verify.
    """
    return (True, "OK")

  with patch.object(validator.runner, "verify", side_effect=mock_verify) as mock_run:
    results: dict[str, bool] = validator.run_all(manual_test_dir=tmp_path)
    assert results["manual_op"] is True
    assert mock_run.call_count == 3


def test_ignore_generated_tests(validator: BatchValidator, tmp_path: pathlib.Path) -> None:
  """Verify `validator.run_all` returns `4` for ignore generated tests.

  Args:
      validator: Input `validator` parameter value (`BatchValidator`).
      tmp_path: Pytest temporary directory path fixture.
  """
  gen_dir: pathlib.Path = tmp_path / "generated"
  gen_dir.mkdir()
  (gen_dir / "test_gen_auto_op.py").write_text("def test_gen_auto_op(): pass")
  with patch.object(validator.runner, "verify", return_value=(True, "OK")) as mock_run:
    validator.run_all(manual_test_dir=tmp_path)
    assert mock_run.call_count == 4


def test_unpack_args_logic(validator: BatchValidator) -> None:
  """Verify `validator._unpack_args` handles logic as expected.

  Args:
      validator: Input `validator` parameter value (`BatchValidator`).
  """
  raw = [("x", "Array"), "axis", ("dims", "Tuple[int]")]
  params: list[str]
  hints: dict[str, str]
  params, hints, _constraints = validator._unpack_args(raw)
  assert params == ["x", "axis", "dims"]
  assert hints["x"] == "Array"
  assert hints["dims"] == "Tuple[int]"
