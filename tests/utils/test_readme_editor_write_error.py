"""Test suite for the Readme Editor Write Error module."""

import typing

from typing import Any


def test_readme_editor_duck() -> None:
  """Verifies the behavior of readme editor duck."""
  from ml_switcheroo.utils.readme_editor import ReadmeEditor

  class DuckPath:
    """Test helper class DuckPath."""

    def exists(self) -> bool:
      """Helper to exists.

      Returns:
          Computed test result or fixture value.
      """
      return True

    def read_text(self, *args: Any, **kwargs: Any) -> str:
      """Helper to read text.

      Args:
          args: Test argument or fixture.
          kwargs: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return "## ✅ Compatibility Matrix\n"

    def write_text(self, *args: Any, **kwargs: Any) -> None:
      """Helper to write text.

      Args:
          args: Test argument or fixture.
          kwargs: Test argument or fixture.
      """
      raise OSError("fail")

  ed: typing.Any = ReadmeEditor(type("Dummy", (), {"get_known_apis": lambda *args: {"op": {}}})(), DuckPath())
  assert ed.update_matrix({}) is False


def test_readme_editor_read_error() -> None:
  """Verifies the behavior of readme editor duck."""
  from ml_switcheroo.utils.readme_editor import ReadmeEditor

  class DuckPath:
    """Test helper class DuckPath."""

    def exists(self) -> bool:
      """Check if exists.

      Returns:
          Computed test result or fixture value.
      """
      return True

    def read_text(self, *args: Any, **kwargs: Any) -> str:
      """Read text.

      Args:
          args: Test argument or fixture.
          kwargs: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      raise OSError("fail")

  ed: typing.Any = ReadmeEditor(type("Dummy", (), {"get_known_apis": lambda *args: {"op": {}}})(), DuckPath())
  assert ed.update_matrix({}) is False


def test_readme_editor_no_map() -> None:
  """Verifies the behavior of readme editor duck."""
  import markdown_it

  from ml_switcheroo.utils.readme_editor import ReadmeEditor

  class DuckPath:
    """Test helper class DuckPath."""

    def exists(self) -> bool:
      """Check if exists.

      Returns:
          Computed test result or fixture value.
      """
      return True

    def read_text(self, *args: Any, **kwargs: Any) -> str:
      """Read text.

      Args:
          args: Test argument or fixture.
          kwargs: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return "## ✅ Compatibility Matrix\n"

    def write_text(self, *args: Any, **kwargs: Any) -> bool:
      """Write text.

      Args:
          args: Test argument or fixture.
          kwargs: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return True

  # Patch the markdown parser to return tokens with no map
  original_parse: Any = markdown_it.MarkdownIt.parse

  def mock_parse(self: Any, content: str) -> Any:
    """Mock parse returning no map.

    Args:
        self: MarkdownIt instance.
        content: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    tokens: Any = original_parse(self, content)
    for t in tokens:
      t.map = None
    return tokens

  setattr(markdown_it.MarkdownIt, "parse", mock_parse)
  ed: typing.Any = ReadmeEditor(type("Dummy", (), {"get_known_apis": lambda *args: {"op": {}}})(), DuckPath())
  assert ed.update_matrix({}) is False
  setattr(markdown_it.MarkdownIt, "parse", original_parse)


def test_guess_category_missing_plugin() -> None:
  """Verifies guess_category."""
  from ml_switcheroo.utils.readme_editor import _guess_category

  assert _guess_category("something_else", {}) == "Math"
  assert _guess_category("something_else", {"requires_plugin": True}) == "Special"
