"""Test suite for the Readme Editor module."""

from pathlib import Path
from typing import Dict

import pytest

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.utils.readme_editor import ReadmeEditor


class MockSemantics(SemanticsManager):
  """Docstring."""

  def get_known_apis(self) -> dict:
    """Mock implementation of get known apis."""
    return {
      "abs": {"variants": {"torch": {"api": "t.abs"}, "jax": {"api": "j.abs"}}},
      "complex_layer": {"variants": {"torch": {"api": "t.nn.C"}, "jax": {"requires_plugin": "p"}}},
      "unsupported": {"variants": {"torch": {"api": "t.bad"}, "jax": None}},
    }


@pytest.fixture
def editor(tmp_path: Path) -> ReadmeEditor:
  """Docstring."""
  mgr: MockSemantics = MockSemantics()
  readme: Path = tmp_path / "README.md"
  readme.write_text("# Test Project\nStart.", encoding="utf-8")
  return ReadmeEditor(mgr, readme)


def test_missing_file_returns_false(tmp_path: Path) -> None:
  """Verifies the behavior of missing file returns false."""
  mgr: MockSemantics = MockSemantics()
  missing_path: Path = tmp_path / "NONEXISTENT.md"
  editor: ReadmeEditor = ReadmeEditor(mgr, missing_path)
  success: bool = editor.update_matrix({})
  assert success is False


def test_missing_section_header_returns_false(editor: ReadmeEditor) -> None:
  """Verifies the behavior of missing section header returns false."""
  editor.readme_path.write_text("# Title\n\nNo matrix here.", encoding="utf-8")
  success: bool = editor.update_matrix({})
  assert success is False
  assert editor.readme_path.read_text(encoding="utf-8") == "# Title\n\nNo matrix here."


def test_injection_between_headers(editor: ReadmeEditor) -> None:
  """Verifies the behavior of injection between headers."""
  original_content: str = "# Title\nIntro text.\n\n## ✅ Compatibility Matrix\n\n| Old | Table |\n| --- | --- |\n| row | 1 |\n\n## Contributing\nPlease help.\n"
  editor.readme_path.write_text(original_content, encoding="utf-8")
  results: Dict[str, bool] = {"abs": True, "complex_layer": False}
  success: bool = editor.update_matrix(results)
  assert success is True
  new_text: str = editor.readme_path.read_text(encoding="utf-8")
  assert "# Title\nIntro text." in new_text
  assert "## ✅ Compatibility Matrix" in new_text
  assert "## Contributing\nPlease help." in new_text
  assert "| `t.abs` | `j.abs` | ✅ Passing |" in new_text
  assert "🧩 Plugin (Complex)" in new_text
  assert "| Old | Table |" not in new_text


def test_injection_at_end_of_file(editor: ReadmeEditor) -> None:
  """Verifies the behavior of injection at end of file."""
  original_content: str = "# Title\n## ✅ Compatibility Matrix\nOld Data\n"
  editor.readme_path.write_text(original_content, encoding="utf-8")
  success: bool = editor.update_matrix({})
  assert success is True
  new_text: str = editor.readme_path.read_text(encoding="utf-8")
  assert "## ✅ Compatibility Matrix" in new_text
  assert "| Category | PyTorch" in new_text
  assert "Old Data" not in new_text


def test_category_heuristics(editor: ReadmeEditor) -> None:
  """Verifies the behavior of category heuristics."""
  editor.readme_path.write_text("## ✅ Compatibility Matrix\n", encoding="utf-8")
  results: Dict[str, bool] = {}
  success: bool = editor.update_matrix(results)
  assert success is True
  content: str = editor.readme_path.read_text(encoding="utf-8")
  assert "| **Neural** | `t.nn.C`" in content


def test_null_variant_handling(editor: ReadmeEditor) -> None:
  """Verifies the behavior of null variant handling."""
  editor.readme_path.write_text("## ✅ Compatibility Matrix\n", encoding="utf-8")
  results: Dict[str, bool] = {"unsupported": False}
  success: bool = editor.update_matrix(results)
  assert success is True
  content: str = editor.readme_path.read_text(encoding="utf-8")
  assert "| `t.bad` | — |" in content
  assert "Untested/Fail" in content
