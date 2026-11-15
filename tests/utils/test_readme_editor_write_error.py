"""Test suite for the Readme Editor Write Error module."""

import typing
from pathlib import Path

import pytest
from markdown_it import MarkdownIt
from markdown_it.token import Token


def test_readme_editor_duck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `ReadmeEditor.update_matrix` returns `False` when `Path.write_text` raises `OSError`.

  Args:
      tmp_path: Temporary directory path fixture.
      monkeypatch: Pytest monkeypatch fixture.
  """
  from ml_switcheroo.semantics.manager import SemanticsManager
  from ml_switcheroo.utils.readme_editor import ReadmeEditor

  readme = tmp_path / "dummy.md"
  readme.write_text("## ✅ Compatibility Matrix\n", encoding="utf-8")

  def mock_write_text(
    self: Path,
    data: str,
    encoding: str | None = None,
    errors: str | None = None,
    newline: str | None = None,
  ) -> typing.NoReturn:
    """Raise `OSError` when writing the updated README content.

    Args:
        self: Target `Path` instance.
        data: Text data to write.
        encoding: Optional text encoding.
        errors: Optional encoding error handler.
        newline: Optional newline mode.

    Returns:
        Never returns; raises `OSError`.

    Raises:
        OSError: Always raised to simulate write failure.
    """
    _ = (self, data, encoding, errors, newline)
    raise OSError("fail")

  class DummySM(SemanticsManager):
    """Minimal `SemanticsManager` test double used in readme editor write error tests."""

    def __init__(self) -> None:
      """Initialize DummySM."""
      super().__init__(load_defaults=False)
      self.data = {"op": {}}

  monkeypatch.setattr(Path, "write_text", mock_write_text)
  ed = ReadmeEditor(DummySM(), readme)
  assert ed.update_matrix({}) is False


def test_readme_editor_read_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `ReadmeEditor.update_matrix` returns `False` when `Path.read_text` raises `OSError`.

  Args:
      tmp_path: Temporary directory path fixture.
      monkeypatch: Pytest monkeypatch fixture.
  """
  from ml_switcheroo.semantics.manager import SemanticsManager
  from ml_switcheroo.utils.readme_editor import ReadmeEditor

  readme = tmp_path / "dummy.md"
  readme.write_text("## ✅ Compatibility Matrix\n", encoding="utf-8")

  def mock_read_text(
    self: Path,
    encoding: str | None = None,
    errors: str | None = None,
    newline: str | None = None,
  ) -> typing.NoReturn:
    """Raise `OSError` when reading the README content.

    Args:
        self: Target `Path` instance.
        encoding: Optional text encoding.
        errors: Optional encoding error handler.
        newline: Optional newline mode.

    Returns:
        Never returns; raises `OSError`.

    Raises:
        OSError: Always raised to simulate read failure.
    """
    _ = (self, encoding, errors, newline)
    raise OSError("fail")

  class DummySM(SemanticsManager):
    """Minimal `SemanticsManager` test double used in readme editor write error tests."""

    def __init__(self) -> None:
      """Initialize DummySM."""
      super().__init__(load_defaults=False)
      self.data = {"op": {}}

  monkeypatch.setattr(Path, "read_text", mock_read_text)
  ed = ReadmeEditor(DummySM(), readme)
  assert ed.update_matrix({}) is False


def test_readme_editor_no_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `ReadmeEditor.update_matrix` returns `False` when parsed heading tokens have `map=None`.

  Args:
      tmp_path: Temporary directory path fixture.
      monkeypatch: Pytest monkeypatch fixture.
  """
  import markdown_it

  from ml_switcheroo.semantics.manager import SemanticsManager
  from ml_switcheroo.utils.readme_editor import ReadmeEditor

  readme = tmp_path / "dummy.md"
  readme.write_text("## ✅ Compatibility Matrix\n", encoding="utf-8")

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
  ed = ReadmeEditor(DummySM(), readme)
  assert ed.update_matrix({}) is False
  monkeypatch.setattr(markdown_it.MarkdownIt, "parse", original_parse, raising=False)


def test_guess_category_missing_plugin() -> None:
  """Verifies guess_category."""
  from ml_switcheroo.utils.readme_editor import _guess_category

  assert _guess_category("something_else", {}) == "Math"
  assert _guess_category("something_else", {"requires_plugin": True}) == "Special"
