"""Unit tests for ReadmeEditor using real tmp_path filesystem fixtures."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import JsonValue

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.utils.readme_editor import ReadmeEditor, _guess_category

NL = chr(10)


class MockReadmeSemantics(SemanticsManager):
  """Minimal SemanticsManager double returning representative operation variants."""

  def __init__(self, apis: dict[str, dict[str, JsonValue]] | None = None) -> None:
    """Initialize MockReadmeSemantics with custom or default known APIs.

    Args:
        apis: Optional custom mapping returned by ``get_known_apis``.
    """
    super().__init__(load_defaults=False)
    self._apis: dict[str, dict[str, JsonValue]] = (
      apis
      if apis is not None
      else {
        "abs": {"variants": {"torch": {"api": "t.abs"}, "jax": {"api": "j.abs"}}},
        "complex_layer": {"variants": {"torch": {"api": "t.nn.C"}, "jax": {"requires_plugin": "p"}}},
        "unsupported": {"variants": {"torch": {"api": "t.bad"}, "jax": None}},
      }
    )

  def get_known_apis(self) -> dict[str, dict[str, JsonValue]]:
    """Return configured operation mapping dictionary.

    Returns:
        Dictionary of known abstract operations and variants.
    """
    return self._apis


@pytest.fixture
def editor(tmp_path: Path) -> ReadmeEditor:
  """Create a ReadmeEditor bound to a temporary README.md file.

  Args:
      tmp_path: Pytest temporary directory path fixture.

  Returns:
      Configured ``ReadmeEditor`` instance.
  """
  readme = tmp_path / "README.md"
  readme.write_text(NL.join(["# Test Project", "Start."]), encoding="utf-8")
  return ReadmeEditor(MockReadmeSemantics(), readme)


def test_readme_editor_init_and_missing_file(tmp_path: Path) -> None:
  """Verify ReadmeEditor stores readme_path and returns False when file does not exist.

  Args:
      tmp_path: Pytest temporary directory path fixture.
  """
  missing_path = tmp_path / "NONEXISTENT.md"
  ed = ReadmeEditor(MockReadmeSemantics(), missing_path)
  assert ed.readme_path.name == "NONEXISTENT.md"
  assert ed.update_matrix({}) is False


def test_missing_section_header_returns_false(editor: ReadmeEditor) -> None:
  """Verify update_matrix returns False without modifying files lacking the matrix heading.

  Args:
      editor: ReadmeEditor fixture instance.
  """
  no_matrix = NL.join(["# Title", "", "No matrix here."])
  editor.readme_path.write_text(no_matrix, encoding="utf-8")
  assert editor.update_matrix({}) is False
  assert editor.readme_path.read_text(encoding="utf-8") == no_matrix


def test_injection_between_headers(editor: ReadmeEditor) -> None:
  """Verify update_matrix replaces only the compatibility matrix section between headers.

  Args:
      editor: ReadmeEditor fixture instance.
  """
  original_content = NL.join(
    [
      "# Title",
      "Intro text.",
      "",
      "## ✅ Compatibility Matrix",
      "",
      "| Old | Table |",
      "| --- | --- |",
      "| row | 1 |",
      "",
      "## Contributing",
      "Please help.",
      "",
    ]
  )
  editor.readme_path.write_text(original_content, encoding="utf-8")
  assert editor.update_matrix({"abs": True, "complex_layer": False}) is True
  new_text = editor.readme_path.read_text(encoding="utf-8")
  assert NL.join(["# Title", "Intro text."]) in new_text
  assert "## ✅ Compatibility Matrix" in new_text
  assert NL.join(["## Contributing", "Please help."]) in new_text
  assert "| `t.abs` | `j.abs` | ✅ Passing |" in new_text
  assert "🧩 Plugin (Complex)" in new_text
  assert "| Old | Table |" not in new_text


def test_injection_at_end_of_file(editor: ReadmeEditor) -> None:
  """Verify update_matrix replaces content to EOF when no subsequent heading exists.

  Args:
      editor: ReadmeEditor fixture instance.
  """
  editor.readme_path.write_text(NL.join(["# Title", "## ✅ Compatibility Matrix", "Old Data", ""]), encoding="utf-8")
  assert editor.update_matrix({}) is True
  new_text = editor.readme_path.read_text(encoding="utf-8")
  assert "## ✅ Compatibility Matrix" in new_text
  assert "| Category | PyTorch" in new_text
  assert "Old Data" not in new_text


def test_category_heuristics_and_null_variants(editor: ReadmeEditor) -> None:
  """Verify Neural/Special/Math heuristics and null variant rendering on disk.

  Args:
      editor: ReadmeEditor fixture instance.
  """
  assert _guess_category("torch.nn.Linear", None) == "Neural"
  assert _guess_category("torch.Conv2d", None) == "Neural"
  assert _guess_category("torch.add", {"requires_plugin": "foo"}) == "Special"
  assert _guess_category("torch.add", None) == "Math"

  editor.readme_path.write_text("## ✅ Compatibility Matrix" + NL, encoding="utf-8")
  assert editor.update_matrix({"unsupported": False}) is True
  content = editor.readme_path.read_text(encoding="utf-8")
  assert "| **Neural** | `t.nn.C`" in content
  assert "| `t.bad` | — |" in content
  assert "Untested/Fail" in content


def test_update_matrix_io_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify update_matrix returns False cleanly on directory read error or write failure.

  Args:
      tmp_path: Pytest temporary directory path fixture.
      monkeypatch: Pytest monkeypatch fixture.
  """
  dir_editor = ReadmeEditor(MockReadmeSemantics(), tmp_path)
  assert dir_editor.update_matrix({}) is False

  readme = tmp_path / "RO_README.md"
  readme.write_text(NL.join(["# Title", "## ✅ Compatibility Matrix", "Old", ""]), encoding="utf-8")
  ro_editor = ReadmeEditor(MockReadmeSemantics(), readme)

  def _raise_os_error(*_args: Path | str, **_kwargs: str) -> int:
    """Raise simulated disk-full OSError.

    Args:
        *_args: Positional write_text arguments.
        **_kwargs: Keyword write_text arguments.

    Returns:
        Never returns.

    Raises:
        OSError: Always raised to simulate disk write failure.
    """
    raise OSError("disk full")

  with monkeypatch.context() as mp:
    mp.setattr(Path, "write_text", _raise_os_error)
    assert ro_editor.update_matrix({"abs": True}) is False


def test_update_matrix_markdown_token_edge_branches(tmp_path: Path) -> None:
  """Verify MarkdownIt token edge cases (map=None and non-inline heading children).

  Args:
      tmp_path: Pytest temporary directory path fixture.
  """
  readme = tmp_path / "TOKENS.md"
  readme.write_text(NL.join(["# Line 0", "# Line 1", "# Line 2"]), encoding="utf-8")

  token_target_none_map = MagicMock(type="heading_open", map=None)
  token_target_inline = MagicMock(type="inline", content="## ✅ Compatibility Matrix")
  ed_none_map = ReadmeEditor(MockReadmeSemantics({}), readme)
  with patch("markdown_it.MarkdownIt.parse", return_value=[token_target_none_map, token_target_inline]):
    assert ed_none_map.update_matrix({}) is False

  mock_tokens = [
    MagicMock(type="heading_open", map=[0, 1]),
    MagicMock(type="heading_close"),
    MagicMock(type="heading_open", map=[1, 2]),
    token_target_inline,
    MagicMock(type="heading_close"),
    MagicMock(type="heading_open", map=None),
  ]
  ed_edges = ReadmeEditor(
    MockReadmeSemantics(
      {
        "NoTorchApi": {"variants": {"torch": {}, "jax": {"api": "jax.foo"}}},
        "PluginOp": {"variants": {"torch": {"api": "torch.foo"}, "jax": {"requires_plugin": "plug"}}},
        "NoJaxApi": {"variants": {"torch": {"api": "torch.bar"}, "jax": {}}},
      }
    ),
    readme,
  )
  with patch("markdown_it.MarkdownIt.parse", return_value=mock_tokens):
    assert ed_edges.update_matrix({"NoTorchApi": False, "PluginOp": False, "NoJaxApi": False}) is True
  written = readme.read_text(encoding="utf-8")
  assert "`jax.foo`" in written
  assert "`torch.bar`" in written
