"""Test suite for scaffold."""

import typing

from pathlib import Path
from unittest.mock import MagicMock, patch

from ml_switcheroo.cli.handlers.scaffold import handle_scaffold


def test_handle_scaffold(tmp_path: Path) -> None:
  """Verify handle scaffold behavior.

  Args:
      tmp_path: Test argument or fixture.
  """
  args: MagicMock = MagicMock()
  args.framework = "test_fw"

  with patch("ml_switcheroo.cli.handlers.scaffold.ConsensusEngine") as MockEngine:
    engine: MagicMock = MockEngine.return_value
    engine.cluster.return_value = {"std_name": ["test_fw.path"]}

    # We need to mock open to not write to current dir
    import builtins

    mock_open: MagicMock = MagicMock()
    mock_file: MagicMock = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file

    with patch.object(builtins, "open", mock_open):
      with patch("ml_switcheroo.cli.handlers.scaffold.json.dump") as mock_dump:
        handle_scaffold(args)
        mock_dump.assert_called_once()


def test_handle_scaffold_empty_paths() -> None:
  """Verify handle scaffold empty paths behavior."""
  args: typing.Any = MagicMock()
  args.framework = "test_fw_empty"

  with patch("ml_switcheroo.cli.handlers.scaffold.ConsensusEngine") as MockEngine:
    engine: typing.Any = MockEngine.return_value
    engine.cluster.return_value = {"std_name": []}

    import builtins

    mock_open: typing.Any = MagicMock()
    with patch.object(builtins, "open", mock_open):
      with patch("ml_switcheroo.cli.handlers.scaffold.json.dump"):
        handle_scaffold(args)
        res: typing.Any = None
  assert res is None
