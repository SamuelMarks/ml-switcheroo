"""
Tests for CLI 'Audit' Command argument handling.

Verifies that:
1.  If no `--roots` provided, the CLI defaults to scanning all registered frameworks.
2.  If `--roots` IS provided, the CLI respects the user list.
"""

from unittest.mock import patch
from ml_switcheroo.cli.__main__ import main


@patch("ml_switcheroo.cli.__main__.available_frameworks")
@patch("ml_switcheroo.cli.commands.handle_audit")
def test_audit_defaults_dynamic(mock_handle, mock_avail):
  """
  Scenario: User runs `ml_switcheroo audit src/`.
  Expectation: Roots defaults to result of `available_frameworks()`.
  """
  # Mock the registry returning a dynamic customized list
  mock_avail.return_value = ["custom_fw", "torch"]

  # Run CLI with no roots arg
  main(["audit", "src/"])

  # Ensure available_frameworks was queried
  mock_avail.assert_called_once()

  # Ensure handle_audit called with dynamic list
  mock_handle.assert_called_once()
  args = mock_handle.call_args[0]
  # arg 0 is path, arg 1 is roots list
  assert args[1] == ["custom_fw", "torch"]


@patch("ml_switcheroo.cli.commands.handle_audit")
def test_audit_explicit_roots(mock_handle):
  """
  Scenario: User runs `ml_switcheroo audit src/ --roots torch tensorflow`.
  Expectation: Roots uses cli value. `available_frameworks` NOT called (implicit via patch behavior).
  """
  main(["audit", "src/", "--roots", "torch", "tensorflow"])

  mock_handle.assert_called_once()
  args = mock_handle.call_args[0]
  assert args[1] == ["torch", "tensorflow"]
