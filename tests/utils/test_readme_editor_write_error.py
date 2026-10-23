"""Test suite for the Readme Editor Write Error module."""

from pathlib import Path
from markdown_it import MarkdownIt
from markdown_it.token import Token


def test_readme_editor_duck() -> None:
  """Verifies the behavior of readme editor duck."""
  from ml_switcheroo.semantics.manager import SemanticsManager
  from ml_switcheroo.utils.readme_editor import ReadmeEditor

  class DuckPath(Path):
    """Test helper class DuckPath."""

    _flavour = getattr(type(Path()), "_flavour", None)

    def exists(self, *, follow_symlinks: bool = True) -> bool:
      """Helper to exists.

      Args:
          follow_symlinks: Follow symlinks flag.

      Returns:
          Computed test result or fixture value.
      """
      return True

    def read_text(
      self,
      encoding: str | None = None,
      errors: str | None = None,
      newline: str | None = None,
    ) -> str:
      """Helper to read text.

      Args:
          encoding: Encoding string.
          errors: Error handling string.
          newline: Newline string.

      Returns:
          Computed test result or fixture value.
      """
      return "## ✅ Compatibility Matrix\n"

    def write_text(
      self,
      data: str,
      encoding: str | None = None,
      errors: str | None = None,
      newline: str | None = None,
    ) -> int:
      """Helper to write text.

      Args:
          data: Text data.
          encoding: Encoding string.
          errors: Error handling string.
          newline: Newline string.

      Returns:
          Bytes written.
      """
      raise OSError("fail")

  class DummySM(SemanticsManager):
    """Test helper class DummySM."""

    def __init__(self) -> None:
      """Initialize DummySM."""
      self.data = {"op": {}}

  ed = ReadmeEditor(DummySM(), DuckPath("dummy.md"))
  assert ed.update_matrix({}) is False


def test_readme_editor_read_error() -> None:
  """Verifies the behavior of readme editor duck."""
  from ml_switcheroo.semantics.manager import SemanticsManager
  from ml_switcheroo.utils.readme_editor import ReadmeEditor

  class DuckPath(Path):
    """Test helper class DuckPath."""

    _flavour = getattr(type(Path()), "_flavour", None)

    def exists(self, *, follow_symlinks: bool = True) -> bool:
      """Check if exists.

      Args:
          follow_symlinks: Follow symlinks flag.

      Returns:
          Computed test result or fixture value.
      """
      return True

    def read_text(
      self,
      encoding: str | None = None,
      errors: str | None = None,
      newline: str | None = None,
    ) -> str:
      """Read text.

      Args:
          encoding: Encoding string.
          errors: Error handling string.
          newline: Newline string.

      Returns:
          Computed test result or fixture value.
      """
      raise OSError("fail")

  class DummySM(SemanticsManager):
    """Test helper class DummySM."""

    def __init__(self) -> None:
      """Initialize DummySM."""
      self.data = {"op": {}}

  ed = ReadmeEditor(DummySM(), DuckPath("dummy.md"))
  assert ed.update_matrix({}) is False


def test_readme_editor_no_map() -> None:
  """Verifies the behavior of readme editor duck."""
  import markdown_it

  from ml_switcheroo.semantics.manager import SemanticsManager
  from ml_switcheroo.utils.readme_editor import ReadmeEditor

  class DuckPath(Path):
    """Test helper class DuckPath."""

    _flavour = getattr(type(Path()), "_flavour", None)

    def exists(self, *, follow_symlinks: bool = True) -> bool:
      """Check if exists.

      Args:
          follow_symlinks: Follow symlinks flag.

      Returns:
          Computed test result or fixture value.
      """
      return True

    def read_text(
      self,
      encoding: str | None = None,
      errors: str | None = None,
      newline: str | None = None,
    ) -> str:
      """Read text.

      Args:
          encoding: Encoding string.
          errors: Error handling string.
          newline: Newline string.

      Returns:
          Computed test result or fixture value.
      """
      return "## ✅ Compatibility Matrix\n"

    def write_text(
      self,
      data: str,
      encoding: str | None = None,
      errors: str | None = None,
      newline: str | None = None,
    ) -> int:
      """Write text.

      Args:
          data: Text data.
          encoding: Encoding string.
          errors: Error handling string.
          newline: Newline string.

      Returns:
          Computed test result or fixture value.
      """
      return 1

  # Patch the markdown parser to return tokens with no map
  original_parse = markdown_it.MarkdownIt.parse

  def mock_parse(self: MarkdownIt, content: str) -> list[Token]:
    """Mock parse returning no map.

    Args:
        self: MarkdownIt instance.
        content: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    tokens = original_parse(self, content)
    for t in tokens:
      t.map = None
    return tokens

  class DummySM(SemanticsManager):
    """Test helper class DummySM."""

    def __init__(self) -> None:
      """Initialize DummySM."""
      self.data = {"op": {}}

  setattr(markdown_it.MarkdownIt, "parse", mock_parse)
  ed = ReadmeEditor(DummySM(), DuckPath("dummy.md"))
  assert ed.update_matrix({}) is False
  setattr(markdown_it.MarkdownIt, "parse", original_parse)


def test_guess_category_missing_plugin() -> None:
  """Verifies guess_category."""
  from ml_switcheroo.utils.readme_editor import _guess_category

  assert _guess_category("something_else", {}) == "Math"
  assert _guess_category("something_else", {"requires_plugin": True}) == "Special"
