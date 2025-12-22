"""
Tests for Batch Validation Logic.

Verifies:
1. Automated fuzzing execution flow.
2. Manual Test prioritization (Human Override).
3. Correct unpacking of type hints from specs.
4. Filtering of generated test files during scan.
5. Passing output_shape_calc to runner.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ml_switcheroo.testing.batch_runner import BatchValidator
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def mock_semantics():
  """Returns a mocked SemanticsManager with sample operations."""
  mgr = MagicMock(spec=SemanticsManager)

  # Define 3 operations:
  # 1. 'auto_op': Will be fuzzed (Passes)
  # 2. 'broken_op': Will be fuzzed (Fails)
  # 3. 'manual_op': Has a manual test file (Skipped fuzzing, marked True)
  # 4. 'shape_op': Has output_shape_calc

  kb = {
    "auto_op": {"std_args": [("x", "int")], "variants": {"torch": {"api": "t.op"}}},
    "broken_op": {"std_args": ["x"], "variants": {"torch": {"api": "t.break"}}},
    "manual_op": {"std_args": ["x"], "variants": {}},
    "shape_op": {
      "std_args": ["x"],
      "output_shape_calc": "lambda x: x.shape",
      "variants": {"torch": {"api": "shape.op"}},
    },
  }

  mgr.get_known_apis.return_value = kb
  return mgr


@pytest.fixture
def validator(mock_semantics):
  return BatchValidator(mock_semantics)


def test_batch_execution_flow(validator):
  """
  Verify that runner.verify is called for operations without manual tests.
  """

  # Mock Runner behavior
  def mock_verify(variants, params, hints=None, constraints=None, shape_calc=None):
    # Handle case where variants is empty (manual_op in this fixture)
    if not variants:
      return True, "Skipped"

    api = list(variants.values())[0]["api"]
    if api == "t.op":
      return True, "OK"
    if api == "t.break":
      return False, "Fail"
    if api == "shape.op":
      # Verify shape calc passed
      if shape_calc == "lambda x: x.shape":
        return True, "Shape OK"
      return False, "Missing Shape Calc"
    return False, "Unknown"

  with patch.object(validator.runner, "verify", side_effect=mock_verify) as mock_run:
    results = validator.run_all()

    # Verify Results dictionary
    assert results["auto_op"] is True
    assert results["broken_op"] is False
    assert results["shape_op"] is True
    assert results["manual_op"] is True

    assert mock_run.call_count == 4


def test_extraction_of_shape_calc(validator):
  """Verify output_shape_calc is extracted from details."""
  with patch.object(validator.runner, "verify", return_value=(True, "OK")) as mock_run:
    validator.run_all()

    # Find call for shape_op
    # args[0] is variants, find the one with shape.op api
    found_shape_call = False
    for call in mock_run.call_args_list:
      args, kwargs = call
      # variants passed as first positional arg
      variants = args[0]

      # If manual_op (empty variants), skip check logic
      if not variants:
        continue

      if variants["torch"]["api"] == "shape.op":
        # Check shape_calc keyword argument
        assert kwargs["shape_calc"] == "lambda x: x.shape"
        found_shape_call = True

    assert found_shape_call


# ... (Existing tests: manual_override, ignore_generated, unpack_args remain identical) ...


def test_manual_override_priority(validator, tmp_path):
  test_dir = tmp_path / "tests"
  test_dir.mkdir()
  (test_dir / "test_manual.py").write_text("def test_manual_op(): pass")

  def mock_verify(*args, **kwargs):
    return True, "OK"

  with patch.object(validator.runner, "verify", side_effect=mock_verify) as mock_run:
    results = validator.run_all(manual_test_dir=tmp_path)
    assert results["manual_op"] is True
    assert mock_run.call_count == 3


def test_ignore_generated_tests(validator, tmp_path):
  gen_dir = tmp_path / "generated"
  gen_dir.mkdir()
  (gen_dir / "test_gen_auto_op.py").write_text("def test_gen_auto_op(): pass")
  with patch.object(validator.runner, "verify", return_value=(True, "OK")) as mock_run:
    validator.run_all(manual_test_dir=tmp_path)
    assert mock_run.call_count == 4


def test_unpack_args_logic(validator):
  raw = [("x", "Array"), "axis", ("dims", "Tuple[int]")]
  params, hints, constraints = validator._unpack_args(raw)
  assert params == ["x", "axis", "dims"]
  assert hints["x"] == "Array"
  assert hints["dims"] == "Tuple[int]"
