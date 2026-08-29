"""Docstring."""

import pathlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo.utils.code_extractor import CodeExtractor
from ml_switcheroo.utils.readme_editor import ReadmeEditor, _guess_category


def test_code_extractor_oserror_1() -> None:
  """Docstring."""

  class DummyClass:
    pass

  with patch("ml_switcheroo.utils.code_extractor.inspect.getsource", side_effect=OSError("Test error")):
    with pytest.raises(OSError, match="Could not get source"):
      CodeExtractor.extract_class(DummyClass)


def test_readme_editor_no_header_1(tmp_path: Path) -> None:
  """Docstring."""
  readme: Path = tmp_path / "README.md"
  readme.write_text("Just some text", encoding="utf-8")
  semantics: MagicMock = MagicMock()
  editor: ReadmeEditor = ReadmeEditor(semantics, readme)
  result: bool = editor.update_matrix({})
  assert result is False


def test_readme_editor_no_mapping_1(tmp_path: Path) -> None:
  """Docstring."""
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
  """Docstring."""
  readme: Path = tmp_path / "README.md"
  readme.write_text("# ✅ Compatibility Matrix\n\nSome text\n\n## Next", encoding="utf-8")
  semantics: MagicMock = MagicMock()
  semantics.get_known_apis.return_value = {}
  editor: ReadmeEditor = ReadmeEditor(semantics, readme)
  with patch.object(Path, "write_text", side_effect=OSError("Mocked error")):
    result: bool = editor.update_matrix({})
    assert result is False


def test_readme_editor_generate_table_none_variant_1() -> None:
  """Docstring."""
  semantics: MagicMock = MagicMock()
  semantics.get_known_apis.return_value = {"op_1": {"variants": {"torch": {"api": "torch.foo"}, "jax": None}}}
  editor: ReadmeEditor = ReadmeEditor(semantics, Path("dummy"))
  table: str = editor._generate_markdown_table({"op_1": False})
  assert "⚠️ Untested/Fail" in table


def test_readme_editor_plugin_fail_1() -> None:
  """Docstring."""
  semantics: MagicMock = MagicMock()
  semantics.get_known_apis.return_value = {
    "op_1": {"variants": {"torch": {"api": "torch.foo"}, "jax": {"api": "jax.foo", "requires_plugin": True}}}
  }
  editor: ReadmeEditor = ReadmeEditor(semantics, Path("dummy"))
  table: str = editor._generate_markdown_table({"op_1": False})
  assert "🧩 Plugin (Complex)" in table


def test_guess_category_special_1() -> None:
  """Docstring."""
  assert _guess_category("torch.foo", {"requires_plugin": True}) == "Special"


# --- Merged from test_utils_missing.py ---


def test_console_missing_export() -> None:
  """Verifies the behavior of console missing export."""
  import unittest.mock

  from ml_switcheroo.utils.console import _ConsoleProxy

  p: _ConsoleProxy = _ConsoleProxy()
  mock_console: Any = unittest.mock.MagicMock()
  p._backend = mock_console
  p.export_text()
  p.export_html()
  p.export_svg()
  mock_console.export_text.assert_called_once()
  mock_console.export_html.assert_called_once()
  mock_console.export_svg.assert_called_once()
  _: bool = p.is_terminal


def test_console_missing_more() -> None:
  """Verifies the behavior of console missing more."""
  import logging

  from rich.console import Console

  from ml_switcheroo.utils.console import (
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
  if hasattr(log, "success"):
    getattr(log, "success")("It works")
  set_console(Console())
  reset_console()
  log_info("i")
  log_warning("w")
  log_error("e")
  log_success("s")
  get_console()


def test_console_missing_export_again() -> None:
  """Verifies the behavior of console missing export again."""
  import unittest.mock

  from ml_switcheroo.utils.console import _ConsoleProxy

  p: _ConsoleProxy = _ConsoleProxy()
  p._backend = unittest.mock.MagicMock()
  p.get_style("bold")
  p.print("hello")


def test_doc_renderer_missing() -> None:
  """Verifies the behavior of documentation renderer missing."""
  from ml_switcheroo.utils.doc_renderer import OpPageRenderer

  r: OpPageRenderer = OpPageRenderer()
  res: str = r.render_rst({"name": "foo", "description": "foo", "args": [], "variants": []})
  assert "No implementations mapped" in res


def test_readme_editor_missing() -> None:
  """Verifies the behavior of readme editor missing."""
  from pathlib import Path

  from ml_switcheroo.utils.readme_editor import ReadmeEditor

  ed: ReadmeEditor = ReadmeEditor(None, Path("does_not_exist.md"))
  assert ed.update_matrix({"foo": True}) is False
  import tempfile

  with tempfile.TemporaryDirectory() as td:
    p: Path = Path(td) / "README.md"
    p.write_text("hello")
    ed2: ReadmeEditor = ReadmeEditor(None, p)
    with __import__("unittest.mock").mock.patch.object(Path, "read_text", side_effect=OSError("fail")):
      assert ed2.update_matrix({"foo": True}) is False


def test_readme_editor_write_error(tmp_path: pathlib.Path) -> None:
  """Verifies the behavior of readme editor write correctly handling an error."""
  from ml_switcheroo.utils.readme_editor import ReadmeEditor

  p: pathlib.Path = tmp_path / "README.md"
  p.write_text("## Translation Matrix")
  ed: ReadmeEditor = ReadmeEditor(None, p)
  with __import__("unittest.mock").mock.patch("pathlib.Path.write_text", side_effect=OSError("fail")):
    with __import__("unittest.mock").mock.patch.object(ed, "_generate_markdown_table", return_value=""):
      assert ed.update_matrix({}) is False


def test_readme_editor_guess_category() -> None:
  """Verifies the behavior of readme editor guess category."""
  from ml_switcheroo.utils.readme_editor import _guess_category

  assert _guess_category("torch.add", {"requires_plugin": "foo"}) == "Special"
