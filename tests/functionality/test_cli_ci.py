"""
Tests for CI/Validation Command CLI logic.

Verifies:
1.  CI command execution logic.
2.  --json-report flag behavior (Lockfile generation).
3.  Integration with BatchValidator.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from ml_switcheroo.cli.__main__ import main


@pytest.fixture
def mock_validator_cls():
  """Patches BatchValidator class to avoid real execution."""
  # FIX: Patch in verify handler
  with patch("ml_switcheroo.cli.handlers.verify.BatchValidator") as mock_cls:
    # Defaults for the instance
    mock_instance = MagicMock()
    mock_instance.run_all.return_value = {"abs": True, "broken_op": False, "untested": False}
    mock_cls.return_value = mock_instance
    yield mock_cls


def test_ci_execution_default(mock_validator_cls, capsys):
  """
  Scenario: Run `ci` with no specific output flags.
  Expect: Validator runs, summary printed to stdout.
  """
  args = ["ci"]

  # Run
  ret_code = main(args)

  assert ret_code == 0

  captured = capsys.readouterr()
  # Check for summary log
  assert "Results: 1/3 mappings verified" in captured.out

  # Check validator call
  mock_validator_cls.return_value.run_all.assert_called_once()


def test_ci_json_report_generation(mock_validator_cls, tmp_path, capsys):
  """
  Scenario: Run `ci --json-report lock.json`.
  Expect: A JSON file created with the validation results dict.
  """
  report_file = tmp_path / "lock.json"
  args = ["ci", "--json-report", str(report_file)]

  ret_code = main(args)

  assert ret_code == 0
  assert report_file.exists()

  # Validate content
  content = json.loads(report_file.read_text())
  assert content["abs"] is True
  assert content["broken_op"] is False
  assert len(content) == 3

  captured = capsys.readouterr()
  assert "Verification report saved to" in captured.out


def test_ci_json_report_failure_handling(mock_validator_cls, capsys, tmp_path):
  """
  Scenario: Path is invalid (e.g., directory matching file name).
  Expect: Log error and return non-zero exit code.
  """
  # Create an invalid path: A directory where we expect a file.
  invalid_path = tmp_path / "protected_dir"
  invalid_path.mkdir()

  # We rely on OS-level errors by trying to open a directory for writing.

  args = ["ci", "--json-report", str(invalid_path)]
  ret_code = main(args)

  assert ret_code == 1
  captured = capsys.readouterr()
  assert "Failed to save report" in captured.out


def test_ci_update_readme_flag(mock_validator_cls, tmp_path):
  """
  Scenario: Run `ci --update-readme`.
  Expect: ReadmeEditor is invoked with correct path.
  """
  readme_path = tmp_path / "README.md"
  readme_path.touch()  # Create dummy

  # FIX: Patch ReadmeEditor in verify handler
  with patch("ml_switcheroo.cli.handlers.verify.ReadmeEditor") as mock_editor_cls:
    args = ["ci", "--update-readme", "--readme-path", str(readme_path)]

    main(args)

    # Verify Editor instantiation
    assert mock_editor_cls.call_count == 1
    call_args = mock_editor_cls.call_args
    assert call_args[0][1] == readme_path  # Second arg is readme_path

    # Verify method call
    mock_editor_cls.return_value.update_matrix.assert_called_once()
