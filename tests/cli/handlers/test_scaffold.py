"""Docstring."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from ml_switcheroo.cli.handlers.scaffold import handle_scaffold


def test_handle_scaffold(tmp_path: Path) -> None:
  """Docstring."""
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
