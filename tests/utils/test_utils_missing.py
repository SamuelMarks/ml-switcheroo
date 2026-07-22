"""Test suite for the Utils Missing module."""


def test_console_missing_export():
  """Verifies the behavior of console missing export."""
  from ml_switcheroo.utils.console import _ConsoleProxy
  import unittest.mock

  p = _ConsoleProxy()
  mock_console = unittest.mock.MagicMock()
  p._backend = mock_console
  p.export_text()
  p.export_html()
  p.export_svg()
  mock_console.export_text.assert_called_once()
  mock_console.export_html.assert_called_once()
  mock_console.export_svg.assert_called_once()
  _ = p.is_terminal


def test_console_missing_more():
  """Verifies the behavior of console missing more."""
  import logging
  from ml_switcheroo.utils.console import (
    get_console,
    set_console,
    reset_console,
    log_info,
    log_warning,
    log_error,
    log_success,
  )
  from rich.console import Console

  log = logging.getLogger("test_success")
  log.setLevel(logging.INFO)
  if hasattr(log, "success"):
    log.success("It works")
  set_console(Console())
  reset_console()
  log_info("i")
  log_warning("w")
  log_error("e")
  log_success("s")
  get_console()


def test_console_missing_export_again():
  """Verifies the behavior of console missing export again."""
  from ml_switcheroo.utils.console import _ConsoleProxy
  import unittest.mock

  p = _ConsoleProxy()
  p._backend = unittest.mock.MagicMock()
  p.get_style("bold")
  p.print("hello")


def test_doc_renderer_missing():
  """Verifies the behavior of documentation renderer missing."""
  from ml_switcheroo.utils.doc_renderer import OpPageRenderer

  r = OpPageRenderer()
  res = r.render_rst({"name": "foo", "description": "foo", "args": [], "variants": []})
  assert "No implementations mapped" in res


def test_readme_editor_missing():
  """Verifies the behavior of readme editor missing."""
  from ml_switcheroo.utils.readme_editor import ReadmeEditor
  from pathlib import Path

  ed = ReadmeEditor(None, Path("does_not_exist.md"))
  assert ed.update_matrix({"foo": True}) is False
  import tempfile

  with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "README.md"
    p.write_text("hello")
    ed2 = ReadmeEditor(None, p)
    with __import__("unittest.mock").mock.patch.object(Path, "read_text", side_effect=OSError("fail")):
      assert ed2.update_matrix({"foo": True}) is False


def test_readme_editor_write_error(tmp_path):
  """Verifies the behavior of readme editor write correctly handling an error."""
  from ml_switcheroo.utils.readme_editor import ReadmeEditor

  p = tmp_path / "README.md"
  p.write_text("## Translation Matrix")
  ed = ReadmeEditor(None, p)
  with __import__("unittest.mock").mock.patch("pathlib.Path.write_text", side_effect=OSError("fail")):
    with __import__("unittest.mock").mock.patch.object(ed, "_generate_markdown_table", return_value=""):
      assert ed.update_matrix({}) is False


def test_readme_editor_guess_category():
  """Verifies the behavior of readme editor guess category."""
  from ml_switcheroo.utils.readme_editor import _guess_category

  assert _guess_category("torch.add", {"requires_plugin": "foo"}) == "Special"
