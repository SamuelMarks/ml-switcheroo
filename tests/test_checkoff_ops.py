"""Tests for checkoff_ops script."""

from unittest.mock import patch, MagicMock


import scripts.checkoff_ops


@patch("builtins.open")
def test_run(mock_open: MagicMock) -> None:
  """Test run function checks off ops correctly."""
  # Setup mock file content
  input_content = "# TODO PLAN\n- [ ] `some_dialect.some_op`\n- [ ] `another.op`\n- [x] `already.done`\n"
  expected_output = "# TODO PLAN\n- [x] `some_dialect.some_op`\n- [x] `another.op`\n- [x] `already.done`\n"

  # We need to mock open for both read and write
  mock_file = MagicMock()
  mock_file.read.return_value = input_content
  mock_open.return_value.__enter__.return_value = mock_file

  scripts.checkoff_ops.run()

  mock_file.write.assert_called_once_with(expected_output)


def test_main_execution() -> None:
  """Test module execution block."""
  source_code = open("scripts/checkoff_ops.py").read()

  with patch("builtins.open") as mock_open:
    mock_file = MagicMock()
    mock_file.read.return_value = ""
    mock_open.return_value.__enter__.return_value = mock_file

    with patch.object(scripts.checkoff_ops, "__name__", "__main__"):
      code = compile(source_code, "scripts/checkoff_ops.py", "exec")
      exec(code, scripts.checkoff_ops.__dict__)

    # Verify open was called by run()
    mock_open.assert_called_with("TODO_PLAN.md", "w")
