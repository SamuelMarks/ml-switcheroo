"""Test suite for utils."""

import pathlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo.utils.code_extractor import CodeExtractor
from ml_switcheroo.utils.readme_editor import ReadmeEditor, _guess_category


def test_code_extractor_oserror_1() -> None:
  """Verify `CodeExtractor.extract_class` raises `OSError` during code extractor oserror 1."""

  class DummyClass:
    """Sentinel `DummyClass` fixture class used in utils tests."""

  with (
    patch("ml_switcheroo.utils.code_extractor.inspect.getsource", side_effect=OSError("Test error")),
    pytest.raises(OSError, match="Could not get source"),
  ):
    CodeExtractor.extract_class(DummyClass)


def test_readme_editor_no_header_1(tmp_path: Path) -> None:
  """Verify `ReadmeEditor` handles readme editor no header 1 as expected.

  Args:
      tmp_path: Pytest temporary directory path fixture.
  """
  readme: Path = tmp_path / "README.md"
  readme.write_text("Just some text", encoding="utf-8")
  semantics: MagicMock = MagicMock()
  editor: ReadmeEditor = ReadmeEditor(semantics, readme)
  result: bool = editor.update_matrix({})
  assert result is False


def test_readme_editor_no_mapping_1(tmp_path: Path) -> None:
  """Verify `ReadmeEditor` handles readme editor no mapping 1 as expected.

  Args:
      tmp_path: Pytest temporary directory path fixture.
  """
  readme: Path = tmp_path / "README.md"
  readme.write_text("# ✅ Compatibility Matrix\n\nSome text", encoding="utf-8")
  semantics: MagicMock = MagicMock()
  editor: ReadmeEditor = ReadmeEditor(semantics, readme)

  # Force map to be None to cover line 87
  with patch("markdown_it.MarkdownIt.parse") as mock_parse:
    mock_token: MagicMock = MagicMock()
    mock_token.type = "heading_open"
    mock_token.map = None
    mock_inline: MagicMock = MagicMock()
    mock_inline.type = "inline"
    mock_inline.content = "✅ Compatibility Matrix"
    mock_parse.return_value = [mock_token, mock_inline]
    result: bool = editor.update_matrix({})
    assert result is False


def test_readme_editor_write_error_1(tmp_path: Path) -> None:
  """Verify `ReadmeEditor` handles readme editor write error 1 as expected.

  Args:
      tmp_path: Pytest temporary directory path fixture.
  """
  readme: Path = tmp_path / "README.md"
  readme.write_text("# ✅ Compatibility Matrix\n\nSome text\n\n## Next", encoding="utf-8")
  semantics: MagicMock = MagicMock()
  semantics.get_known_apis.return_value = {}
  editor: ReadmeEditor = ReadmeEditor(semantics, readme)
  with patch.object(Path, "write_text", side_effect=OSError("Mocked error")):
    result: bool = editor.update_matrix({})
    assert result is False


def test_readme_editor_generate_table_none_variant_1() -> None:
  """Verify `ReadmeEditor` handles readme editor generate table none variant 1 as expected."""
  semantics: MagicMock = MagicMock()
  semantics.get_known_apis.return_value = {"op_1": {"variants": {"torch": {"api": "torch.foo"}, "jax": None}}}
  editor: ReadmeEditor = ReadmeEditor(semantics, Path("dummy"))
  table: str = editor._generate_markdown_table({"op_1": False})
  assert "⚠️ Untested/Fail" in table


def test_readme_editor_plugin_fail_1() -> None:
  """Verify `ReadmeEditor` emits `🧩 Plugin (Complex)` for readme editor plugin fail 1."""
  semantics: MagicMock = MagicMock()
  semantics.get_known_apis.return_value = {
    "op_1": {"variants": {"torch": {"api": "torch.foo"}, "jax": {"api": "jax.foo", "requires_plugin": True}}}
  }
  editor: ReadmeEditor = ReadmeEditor(semantics, Path("dummy"))
  table: str = editor._generate_markdown_table({"op_1": False})
  assert "🧩 Plugin (Complex)" in table


def test_guess_category_special_1() -> None:
  """Verify `_guess_category` handles special 1 as expected."""
  assert _guess_category("torch.foo", {"requires_plugin": True}) == "Special"


def test_console_missing_export() -> None:
  """Verify `_ConsoleProxy` handles console missing export as expected."""
  import unittest.mock

  from ml_switcheroo.utils.console import _ConsoleProxy

  p: _ConsoleProxy = _ConsoleProxy()
  mock_console = unittest.mock.MagicMock()
  p._backend = mock_console
  p.export_text()
  p.export_html()
  p.export_svg()
  mock_console.export_text.assert_called_once()
  mock_console.export_html.assert_called_once()
  mock_console.export_svg.assert_called_once()
  is_term = p.is_terminal
  assert is_term is not None


def test_console_missing_more() -> None:
  """Verify `logging.getLogger` handles console missing more as expected."""
  import logging

  from rich.console import Console

  from ml_switcheroo.utils.console import (
    _SuccessLogger,
    get_console,
    log_error,
    log_info,
    log_success,
    log_warning,
    reset_console,
    set_console,
  )

  log: logging.Logger = logging.getLogger("test_success")
  log.setLevel(logging.INFO)
  assert isinstance(log, _SuccessLogger)
  assert log.name == "test_success"
  assert log.level == logging.INFO
  log.success("It works")
  custom = Console(record=True, file=None)
  set_console(custom)
  assert get_console() is custom
  log_success("recorded_success")
  exported = custom.export_text(clear=False)
  assert "recorded_success" in exported
  assert "✅" in exported
  reset_console()
  log_info("i")
  log_warning("w")
  log_error("e")
  log_success("s")
  _res = get_console()
  assert isinstance(_res, Console)
  assert _res is not custom
  assert _res.width > 0


def test_console_missing_export_again() -> None:
  """Verify `_ConsoleProxy` handles console missing export again as expected."""
  import unittest.mock

  from ml_switcheroo.utils.console import _ConsoleProxy

  p: _ConsoleProxy = _ConsoleProxy()
  p._backend = unittest.mock.MagicMock()
  p.get_style("bold")
  p.print("hello")
  _res = None
  assert _res is None
  p._backend.print.assert_called_once_with("hello")


def test_doc_renderer_missing() -> None:
  """Verify `OpPageRenderer` emits `No implementations mapped` for doc renderer missing."""
  from ml_switcheroo.utils.doc_renderer import OpPageRenderer

  r: OpPageRenderer = OpPageRenderer()
  res: str = r.render_rst({"name": "foo", "description": "foo", "args": [], "variants": []})
  assert "No implementations mapped" in res


def test_readme_editor_missing() -> None:
  """Verify `ReadmeEditor` handles readme editor missing as expected."""
  from pathlib import Path
  from unittest import mock

  from ml_switcheroo.utils.readme_editor import ReadmeEditor

  ed = ReadmeEditor(MagicMock(), Path("does_not_exist.md"))
  assert ed.update_matrix({"foo": True}) is False
  import tempfile

  with tempfile.TemporaryDirectory() as td:
    p: Path = Path(td) / "README.md"
    p.write_text("hello")
    ed2 = ReadmeEditor(MagicMock(), p)
    with mock.patch.object(Path, "read_text", side_effect=OSError("fail")):
      assert ed2.update_matrix({"foo": True}) is False


def test_readme_editor_write_error(tmp_path: pathlib.Path) -> None:
  """Verify `ReadmeEditor` handles readme editor write error as expected.

  Args:
      tmp_path: Pytest temporary directory path fixture.
  """
  from unittest import mock

  from ml_switcheroo.utils.readme_editor import ReadmeEditor

  p: pathlib.Path = tmp_path / "README.md"
  p.write_text("## Translation Matrix")
  ed = ReadmeEditor(MagicMock(), p)
  with (
    mock.patch("pathlib.Path.write_text", side_effect=OSError("fail")),
    mock.patch.object(ed, "_generate_markdown_table", return_value=""),
  ):
    assert ed.update_matrix({}) is False


def test_readme_editor_guess_category() -> None:
  """Verify `_guess_category` handles readme editor guess category as expected."""
  from ml_switcheroo.utils.readme_editor import _guess_category

  assert _guess_category("torch.add", {"requires_plugin": "foo"}) == "Special"
