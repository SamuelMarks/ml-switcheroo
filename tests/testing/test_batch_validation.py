"""
Tests for Batch Validation Logic.

Verifies:
1. Automated fuzzing execution flow.
2. Manual Test prioritization (Human Override).
3. Correct unpacking of type hints from specs.
4. Filtering of generated test files during scan.
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

  kb = {
    "auto_op": {"std_args": [("x", "int")], "variants": {"torch": {"api": "t.op"}}},
    "broken_op": {"std_args": ["x"], "variants": {"torch": {"api": "t.break"}}},
    "manual_op": {"std_args": ["x"], "variants": {}},
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
  # auto_op -> True, broken_op -> False
  def mock_verify(variants, params, hints=None):
    # Handle case where variants is empty (manual_op in this fixture)
    if not variants:
      return True, "Skipped"

    # Safe access now that we checked for empty
    api = list(variants.values())[0]["api"]
    if api == "t.op":
      return True, "OK"
    if api == "t.break":
      return False, "Fail"
    return False, "Unknown"

  with patch.object(validator.runner, "verify", side_effect=mock_verify) as mock_run:
    results = validator.run_all()

    # Verify Results dictionary
    assert results["auto_op"] is True
    assert results["broken_op"] is False

    # Verify Manual op result (True because logic maps empty/skipped to True usually)
    assert results["manual_op"] is True

    assert mock_run.call_count == 3


def test_manual_override_priority(validator, tmp_path):
  """
  Scenario: 'manual_op' has a test file `tests/test_manual.py`.
  Expectation: Fuzzer is NOT called for 'manual_op', result is True.
  """
  # Create manual test file
  test_dir = tmp_path / "tests"
  test_dir.mkdir()
  (test_dir / "test_manual.py").write_text("def test_manual_op(): pass")

  # Mock Runner to crash if called for manual_op
  def mock_verify(variants, params, hints=None):
    return True, "OK"

  with patch.object(validator.runner, "verify", side_effect=mock_verify) as mock_run:
    results = validator.run_all(manual_test_dir=tmp_path)

    # Assert Result
    assert results["manual_op"] is True

    # Assert Runner NOT called for manual_op
    # Logic: 3 total ops. manual_op skipped. run called 2 times.
    assert mock_run.call_count == 2


def test_ignore_generated_tests(validator, tmp_path):
  """
  Scenario: 'auto_op' has a test in `generated/test_gen_auto_op.py`.
  Expectation: This is NOT counted as manual verification. Fuzzer runs.
  """
  gen_dir = tmp_path / "generated"
  gen_dir.mkdir()
  (gen_dir / "test_gen_auto_op.py").write_text("def test_gen_auto_op(): pass")

  with patch.object(validator.runner, "verify", return_value=(True, "OK")) as mock_run:
    validator.run_all(manual_test_dir=tmp_path)

    # Should run 3 times because generated test is ignored
    assert mock_run.call_count == 3


def test_unpack_args_logic(validator):
  """
  Verify conversion of Spec definitions to Fuzzer inputs.
  """
  raw = [("x", "Array"), "axis", ("dims", "Tuple[int]")]

  params, hints = validator._unpack_args(raw)

  assert params == ["x", "axis", "dims"]
  assert hints["x"] == "Array"
  assert "axis" not in hints
  assert hints["dims"] == "Tuple[int]"
