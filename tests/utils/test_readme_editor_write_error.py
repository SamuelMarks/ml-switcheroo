"""Test suite for the Readme Editor Write Error module."""

from typing import Any


def test_readme_editor_duck() -> None:
  """Verifies the behavior of readme editor duck."""
  from ml_switcheroo.utils.readme_editor import ReadmeEditor

  class DuckPath:
    """Docstring."""

    def exists(self) -> bool:
      """Helper to exists."""
      return True

    def read_text(self, *args: Any, **kwargs: Any) -> str:
      """Helper to read text."""
      return "## ✅ Compatibility Matrix\n"

    def write_text(self, *args: Any, **kwargs: Any) -> None:
      """Helper to write text."""
      raise OSError("fail")

  ed: ReadmeEditor = ReadmeEditor(type("Dummy", (), {"get_known_apis": lambda *args: {"op": {}}})(), DuckPath())
  assert ed.update_matrix({}) is False


def test_readme_editor_read_error() -> None:
  """Verifies the behavior of readme editor duck."""
  from ml_switcheroo.utils.readme_editor import ReadmeEditor

  class DuckPath:
    """Docstring."""

    def exists(self) -> bool:
      """Check if exists."""
      return True

    def read_text(self, *args: Any, **kwargs: Any) -> str:
      """Read text."""
      raise OSError("fail")

  ed: ReadmeEditor = ReadmeEditor(type("Dummy", (), {"get_known_apis": lambda *args: {"op": {}}})(), DuckPath())
  assert ed.update_matrix({}) is False


def test_readme_editor_no_map() -> None:
  """Verifies the behavior of readme editor duck."""
  import markdown_it

  from ml_switcheroo.utils.readme_editor import ReadmeEditor

  class DuckPath:
    """Docstring."""

    def exists(self) -> bool:
      """Check if exists."""
      return True

    def read_text(self, *args: Any, **kwargs: Any) -> str:
      """Read text."""
      return "## ✅ Compatibility Matrix\n"

    def write_text(self, *args: Any, **kwargs: Any) -> bool:
      """Write text."""
      return True

  # Patch the markdown parser to return tokens with no map
  original_parse: Any = markdown_it.MarkdownIt.parse

  def mock_parse(self: Any, content: str) -> Any:
    """Mock parse returning no map."""
    tokens: Any = original_parse(self, content)
    for t in tokens:
      t.map = None
    return tokens

  markdown_it.MarkdownIt.parse = mock_parse
  ed: ReadmeEditor = ReadmeEditor(type("Dummy", (), {"get_known_apis": lambda *args: {"op": {}}})(), DuckPath())
  assert ed.update_matrix({}) is False
  markdown_it.MarkdownIt.parse = original_parse


def test_guess_category_missing_plugin() -> None:
  """Verifies guess_category."""
  from ml_switcheroo.utils.readme_editor import _guess_category

  assert _guess_category("something_else", {}) == "Math"
  assert _guess_category("something_else", {"requires_plugin": True}) == "Special"
