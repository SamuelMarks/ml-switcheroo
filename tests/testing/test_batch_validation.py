"""Test suite for the Batch Validation module."""

import pytest
from unittest.mock import MagicMock, patch
from ml_switcheroo.testing.batch_runner import BatchValidator
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def mock_semantics():
  """Provides a mock semantics for testing."""
  mgr = MagicMock(spec=SemanticsManager)
  kb = {
    "auto_op": {"std_args": [("x", "int")], "variants": {"torch": {"api": "t.op"}}},
    "broken_op": {"std_args": ["x"], "variants": {"torch": {"api": "t.break"}}},
    "manual_op": {"std_args": ["x"], "variants": {}},
    "shape_op": {"std_args": ["x"], "output_shape_calc": "lambda x: x.shape", "variants": {"torch": {"api": "shape.op"}}},
  }
  mgr.get_known_apis.return_value = kb
  return mgr


@pytest.fixture
def validator(mock_semantics):
  """Provides a mock validator for testing."""
  return BatchValidator(mock_semantics)


def test_batch_execution_flow(validator):
  """Verifies the behavior of batch execution flow."""

  def mock_verify(variants, params, hints=None, constraints=None, shape_calc=None):
    """Provides a mock verify for testing."""
    if not variants:
      return (True, "Skipped")
    api = list(variants.values())[0]["api"]
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
    results = validator.run_all()
    assert results["auto_op"] is True
    assert results["broken_op"] is False
    assert results["shape_op"] is True
    assert results["manual_op"] is True
    assert mock_run.call_count == 4


def test_extraction_of_shape_calc(validator):
  """Verifies the behavior of extraction of shape calculation."""
  with patch.object(validator.runner, "verify", return_value=(True, "OK")) as mock_run:
    validator.run_all()
    found_shape_call = False
    for call in mock_run.call_args_list:
      (args, kwargs) = call
      variants = args[0]
      if not variants:
        continue
      if variants["torch"]["api"] == "shape.op":
        assert kwargs["shape_calc"] == "lambda x: x.shape"
        found_shape_call = True
    assert found_shape_call


def test_manual_override_priority(validator, tmp_path):
  """Verifies the behavior of manual override priority."""
  test_dir = tmp_path / "tests"
  test_dir.mkdir()
  (test_dir / "test_manual.py").write_text("def test_manual_op(): pass")

  def mock_verify(*args, **kwargs):
    """Provides a mock verify for testing."""
    return (True, "OK")

  with patch.object(validator.runner, "verify", side_effect=mock_verify) as mock_run:
    results = validator.run_all(manual_test_dir=tmp_path)
    assert results["manual_op"] is True
    assert mock_run.call_count == 3


def test_ignore_generated_tests(validator, tmp_path):
  """Verifies the behavior of ignore generated tests."""
  gen_dir = tmp_path / "generated"
  gen_dir.mkdir()
  (gen_dir / "test_gen_auto_op.py").write_text("def test_gen_auto_op(): pass")
  with patch.object(validator.runner, "verify", return_value=(True, "OK")) as mock_run:
    validator.run_all(manual_test_dir=tmp_path)
    assert mock_run.call_count == 4


def test_unpack_args_logic(validator):
  """Verifies the behavior of unpack arguments logic."""
  raw = [("x", "Array"), "axis", ("dims", "Tuple[int]")]
  (params, hints, constraints) = validator._unpack_args(raw)
  assert params == ["x", "axis", "dims"]
  assert hints["x"] == "Array"
  assert hints["dims"] == "Tuple[int]"
