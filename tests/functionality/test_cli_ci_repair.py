"""
Integration test for CI Auto-Repair functionality.
"""

import pytest
from unittest.mock import MagicMock, patch
from ml_switcheroo.cli.handlers.verify import handle_ci
from ml_switcheroo.testing.batch_runner import BatchValidator
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.testing.runner import EquivalenceRunner


@pytest.fixture
def mock_validator_repair():
  """Patches BatchValidator to simulate failures and repairable logic."""
  with patch("ml_switcheroo.cli.handlers.verify.BatchValidator") as mock_cls:
    instance = MagicMock(spec=BatchValidator)
    # Simulate initial failure
    instance.run_all.return_value = {"FailingOp": False}
    instance.runner = MagicMock(spec=EquivalenceRunner)
    mock_cls.return_value = instance
    yield instance


@pytest.fixture
def mock_semantics():
  with patch("ml_switcheroo.cli.handlers.verify.SemanticsManager") as mock_cls:
    instance = MagicMock(spec=SemanticsManager)
    # Return a definition that can be relaxed
    op_def = {"std_args": ["x"], "variants": {}, "test_rtol": 1e-5}
    instance.get_definition_by_id.return_value = op_def
    mock_cls.return_value = instance
    yield instance


def test_ci_repair_triggers_bisection_and_update(mock_validator_repair, mock_semantics, capsys):
  """
  Scenario: 'ci --repair' is called. Op fails initially. Bisector finds a relaxed tolerance.
  Expectation: Bisector.propose_fix is called, and semantics.update_definition is called.
  """
  # Fix the bisector logic using a mock side_effect to simulate success on relaxed params
  # Bisector instantiates with validator.runner.
  # We need to mock the runner's verify method to fail first, then succeed with looser tols.

  # 1. Setup Runner behavior for Bisection Loop
  runner = mock_validator_repair.runner
  # Sequence:
  # 1. Standard (Fail)
  # 2. Loose Abs (Fail)
  # 3. Loose Rel (True) -> Success!
  runner.verify.side_effect = [(False, "Err"), (False, "Err"), (True, "OK")]

  # 2. Run CI with Repair=True
  ret = handle_ci(update_readme=False, readme_path=None, json_report=None, repair=True)

  # 3. Assertions
  assert ret == 0

  # Verify bisector interaction logic flow
  assert mock_semantics.get_definition_by_id.called
  mock_semantics.update_definition.assert_called_once()
  args = mock_semantics.update_definition.call_args
  assert args[0][0] == "FailingOp"
  # Ensure patched values
  assert args[0][1]["test_rtol"] == 0.01  # 1e-2 from 3rd step in bisector

  captured = capsys.readouterr()
  assert "Repaired 'FailingOp'" in captured.out
