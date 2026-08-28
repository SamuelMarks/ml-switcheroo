"""Docstring."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from ml_switcheroo.utils.code_extractor import CodeExtractor
from ml_switcheroo.utils.readme_editor import ReadmeEditor, _guess_category


def test_code_extractor_oserror_1() -> None:
  """Docstring."""

  class DummyClass:
    """Docstring."""

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
