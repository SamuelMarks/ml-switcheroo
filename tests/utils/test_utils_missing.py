def test_console_missing_export():
  from ml_switcheroo.utils.console import _ConsoleProxy
  from rich.console import Console
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

  # also cover property
  _ = p.is_terminal


def test_console_missing_more():
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

  # 34-35: logger.success via logging
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
  from ml_switcheroo.utils.console import _ConsoleProxy
  import unittest.mock

  p = _ConsoleProxy()
  p._backend = unittest.mock.MagicMock()
  p.get_style("bold")
  p.print("hello")


def test_doc_renderer_missing():
  from ml_switcheroo.utils.doc_renderer import OpPageRenderer

  r = OpPageRenderer()
  # 60: empty variants
  res = r.render_rst({"name": "foo", "description": "foo", "args": [], "variants": []})
  assert "No implementations mapped" in res


def test_readme_editor_missing():
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
  from ml_switcheroo.utils.readme_editor import ReadmeEditor

  p = tmp_path / "README.md"
  p.write_text("## Translation Matrix")
  ed = ReadmeEditor(None, p)

  with __import__("unittest.mock").mock.patch("pathlib.Path.write_text", side_effect=OSError("fail")):
    with __import__("unittest.mock").mock.patch.object(ed, "_generate_markdown_table", return_value=""):
      assert ed.update_matrix({}) is False


def test_readme_editor_guess_category():
  from ml_switcheroo.utils.readme_editor import _guess_category

  assert _guess_category("torch.add", {"requires_plugin": "foo"}) == "Special"
