"""Test suite for suggest."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from ml_switcheroo.cli.handlers.suggest import handle_suggest


def test_handle_suggest_single() -> None:
  """Verify handle suggest single behavior."""
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
  """Verify handle suggest wildcard behavior.

  Args:
      tmp_path: Test argument or fixture.
  """
  out_dir: Path = tmp_path / "wildcard_out"
  res = handle_suggest("math.*", out_dir, 5)
  assert res == 0
  assert out_dir.exists()


def test_handle_suggest_wildcard_fail(tmp_path: Path) -> None:
  """Verify handle suggest wildcard fail behavior.

  Args:
      tmp_path: Test argument or fixture.
  """
  assert (
    handle_suggest("doesnt_exist.*") == 1
  )  # Can't mock easily here without patching builtins, wait, just let it fail to find anything valid.


def test_handle_suggest_single_fail() -> None:
  """Verify handle suggest single fail behavior."""
  assert handle_suggest("os.does_not_exist") == 1


def test_handle_suggest_single_out_dir(tmp_path: Path) -> None:
  """Verify handle suggest single out dir behavior.

  Args:
      tmp_path: Test argument or fixture.
  """
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


def test_suggest_missing_branches(tmp_path: Path) -> None:
  """Verify suggest missing branches behavior.

  Args:
      tmp_path: Test argument or fixture.
  """
  from unittest.mock import MagicMock, patch

  from ml_switcheroo.cli.handlers.suggest import handle_suggest

  # 1. glob mode (.*) with some members and no members
  out_dir = tmp_path / "suggest_out"
  out_dir.mkdir(parents=True, exist_ok=True)

  assert handle_suggest("math.*", out_dir, 10) == 0  # 101->104 (out_dir exists), 54-65 (glob members)

  # 2. No targets
  import inspect

  with patch("ml_switcheroo.cli.handlers.suggest.importlib.import_module"):
    # no members
    with patch.object(inspect, "getmembers", return_value=[]):
      assert handle_suggest("empty_module.*", None, 10) == 1  # 79-80 (not targets)

  # 3. Invalid api_path (no dot)
  assert handle_suggest("nodots", None, 10) == 1  # 169 (invalid path format)

  # 4. inspect.signature ValueError/TypeError (144-146)
  assert handle_suggest("math.sqrt", None, 10) == 0

  # 5. Extract metadata with C-extension Exception
  with patch("ml_switcheroo.cli.handlers.suggest.importlib.import_module"):
    with patch.object(inspect, "getmembers", return_value=[("func", MagicMock())]):
      with patch.object(inspect, "getdoc", side_effect=Exception("Failed")):
        assert handle_suggest("test.*", None, 10) == 1  # 64-65 (except Exception: continue)


def test_suggest_ismodule(tmp_path: Path) -> None:
  """Verify suggest ismodule behavior.

  Args:
      tmp_path: Test argument or fixture.
  """
  import inspect
  from unittest.mock import MagicMock, patch

  from ml_switcheroo.cli.handlers.suggest import handle_suggest

  with patch("ml_switcheroo.cli.handlers.suggest.importlib.import_module"):
    with patch.object(inspect, "getmembers", return_value=[("submod", MagicMock())]):
      with patch.object(inspect, "ismodule", return_value=True):
        assert handle_suggest("test.*", None, 10) == 1
