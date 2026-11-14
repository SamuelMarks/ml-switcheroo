"""Test suite for the Readme Editor Write Error module."""

from pathlib import Path

import pytest
from markdown_it import MarkdownIt
from markdown_it.token import Token


def test_readme_editor_duck() -> None:
  """Verify `ReadmeEditor` handles readme editor duck as expected."""
  from ml_switcheroo.semantics.manager import SemanticsManager
  from ml_switcheroo.utils.readme_editor import ReadmeEditor

  class DuckPath(Path):
    """Stub `Path` subclass implementing `exists`, `read_text` for readme editor write error tests."""

    def exists(self, *, follow_symlinks: bool = True) -> bool:
      """Execute `exists` on `DuckPath` for readme editor write error tests.

      Args:
          follow_symlinks: Follow symlinks flag.

      Returns:
          Resulting `bool` from exists.
      """
      return True

    def read_text(
      self,
      encoding: str | None = None,
      errors: str | None = None,
      newline: str | None = None,
    ) -> str:
      """Execute `read_text` on `DuckPath` for readme editor write error tests.

      Args:
          encoding: Encoding string.
          errors: Error handling string.
          newline: Newline string.

      Returns:
          Resulting `str` from read text.
      """
      return "## ✅ Compatibility Matrix\n"

    def write_text(
      self,
      data: str,
      encoding: str | None = None,
      errors: str | None = None,
      newline: str | None = None,
    ) -> int:
      """Execute `OSError` to support write text in readme editor write error tests.

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
    """Minimal `SemanticsManager` test double used in readme editor write error tests."""

    def __init__(self) -> None:
      """Initialize DummySM."""
      super().__init__(load_defaults=False)
      self.data = {"op": {}}

  ed = ReadmeEditor(DummySM(), DuckPath("dummy.md"))
  assert ed.update_matrix({}) is False


def test_readme_editor_read_error() -> None:
  """Verify `ReadmeEditor` handles readme editor read error as expected."""
  from ml_switcheroo.semantics.manager import SemanticsManager
  from ml_switcheroo.utils.readme_editor import ReadmeEditor

  class DuckPath(Path):
    """Stub `Path` subclass implementing `exists`, `read_text` for readme editor write error tests."""

    def exists(self, *, follow_symlinks: bool = True) -> bool:
      """Check if exists.

      Args:
          follow_symlinks: Follow symlinks flag.

      Returns:
          Resulting `bool` from exists.
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
          Resulting `str` from read text.
      """
      raise OSError("fail")

  class DummySM(SemanticsManager):
    """Minimal `SemanticsManager` test double used in readme editor write error tests."""

    def __init__(self) -> None:
      """Initialize DummySM."""
      super().__init__(load_defaults=False)
      self.data = {"op": {}}

  ed = ReadmeEditor(DummySM(), DuckPath("dummy.md"))
  assert ed.update_matrix({}) is False


def test_readme_editor_no_map(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `ReadmeEditor` handles readme editor no map as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  import markdown_it

  from ml_switcheroo.semantics.manager import SemanticsManager
  from ml_switcheroo.utils.readme_editor import ReadmeEditor

  class DuckPath(Path):
    """Stub `Path` subclass implementing `exists`, `read_text` for readme editor write error tests."""

    def exists(self, *, follow_symlinks: bool = True) -> bool:
      """Check if exists.

      Args:
          follow_symlinks: Follow symlinks flag.

      Returns:
          Resulting `bool` from exists.
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
          Resulting `str` from read text.
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
          Resulting `int` from write text.
      """
      return 1

  # Patch the markdown parser to return tokens with no map
  original_parse = markdown_it.MarkdownIt.parse

  def mock_parse(self: MarkdownIt, content: str) -> list[Token]:
    """Mock parse returning no map.

    Args:
        self: MarkdownIt instance.
        content: Input `content` parameter value (`str`).

    Returns:
        Resulting `list[Token]` from mock parse.
    """
    tokens = original_parse(self, content)
    for t in tokens:
      t.map = None
    return tokens

  class DummySM(SemanticsManager):
    """Minimal `SemanticsManager` test double used in readme editor write error tests."""

    def __init__(self) -> None:
      """Initialize DummySM."""
      super().__init__(load_defaults=False)
      self.data = {"op": {}}

  monkeypatch.setattr(markdown_it.MarkdownIt, "parse", mock_parse, raising=False)
  ed = ReadmeEditor(DummySM(), DuckPath("dummy.md"))
  assert ed.update_matrix({}) is False
  monkeypatch.setattr(markdown_it.MarkdownIt, "parse", original_parse, raising=False)


def test_guess_category_missing_plugin() -> None:
  """Verifies guess_category."""
  from ml_switcheroo.utils.readme_editor import _guess_category

  assert _guess_category("something_else", {}) == "Math"
  assert _guess_category("something_else", {"requires_plugin": True}) == "Special"
