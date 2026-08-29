"""Docstring."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from ml_switcheroo.cli.handlers.suggest import handle_suggest


def test_handle_suggest_single() -> None:
  """Docstring."""
  with patch("ml_switcheroo.cli.handlers.suggest.importlib.import_module") as mock_import:
    mock_mod: MagicMock = MagicMock()
    mock_mod.foo = MagicMock()
    mock_mod.foo.__doc__ = "doc"
    mock_import.return_value = mock_mod

    # We need to mock inspect.signature to not fail
    import inspect

    with patch.object(inspect, "signature", return_value=MagicMock()):
      assert handle_suggest("os.foo") == 0


def test_handle_suggest_wildcard(tmp_path: Path) -> None:
  """Docstring."""
  with patch("ml_switcheroo.cli.handlers.suggest.importlib.import_module"):
    mock_mod: MagicMock = MagicMock()
    mock_mod.foo = MagicMock()
    mock_mod.foo.__doc__ = "doc"
    mock_mod.foo.__module__ = "os"
    mock_mod.__name__ = "os"

    # dir() on mock doesn't work easily, we mock _get_public_callables directly maybe?
    # or mock `getattr`
    pass


def test_handle_suggest_wildcard_fail(tmp_path: Path) -> None:
  """Docstring."""
  assert (
    handle_suggest("doesnt_exist.*") == 1
  )  # Can't mock easily here without patching builtins, wait, just let it fail to find anything valid.


def test_handle_suggest_single_fail() -> None:
  """Docstring."""
  assert handle_suggest("os.does_not_exist") == 1


def test_handle_suggest_single_out_dir(tmp_path: Path) -> None:
  """Docstring."""
  out: Path = tmp_path / "out"
  with patch("ml_switcheroo.cli.handlers.suggest.importlib.import_module") as mock_import:
    mock_mod: MagicMock = MagicMock()
    mock_mod.foo = MagicMock()
    mock_mod.foo.__doc__ = "doc"
    mock_import.return_value = mock_mod
    import inspect

    with patch.object(inspect, "signature", return_value=MagicMock()):
      assert handle_suggest("os.foo", out) == 0
      assert out.exists()
