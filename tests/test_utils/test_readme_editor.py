"""Test suite for the Readme Editor module."""

import pytest
from ml_switcheroo.utils.readme_editor import ReadmeEditor
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  """Mock Semantics class for testing purposes."""

  def get_known_apis(self) -> dict:
    """Mock implementation of get known apis."""
    return {
      "abs": {"variants": {"torch": {"api": "t.abs"}, "jax": {"api": "j.abs"}}},
      "complex_layer": {"variants": {"torch": {"api": "t.nn.C"}, "jax": {"requires_plugin": "p"}}},
      "unsupported": {"variants": {"torch": {"api": "t.bad"}, "jax": None}},
    }


@pytest.fixture
def editor(tmp_path):
  """Provides a mock editor for testing."""
  mgr = MockSemantics()
  readme = tmp_path / "README.md"
  readme.write_text("# Test Project\nStart.", encoding="utf-8")
  return ReadmeEditor(mgr, readme)


def test_missing_file_returns_false(tmp_path):
  """Verifies the behavior of missing file returns false."""
  mgr = MockSemantics()
  missing_path = tmp_path / "NONEXISTENT.md"
  editor = ReadmeEditor(mgr, missing_path)
  success = editor.update_matrix({})
  assert success is False


def test_missing_section_header_returns_false(editor):
  """Verifies the behavior of missing section header returns false."""
  editor.readme_path.write_text("# Title\n\nNo matrix here.", encoding="utf-8")
  success = editor.update_matrix({})
  assert success is False
  assert editor.readme_path.read_text(encoding="utf-8") == "# Title\n\nNo matrix here."


def test_injection_between_headers(editor):
  """Verifies the behavior of injection between headers."""
  original_content = "# Title\nIntro text.\n\n## ✅ Compatibility Matrix\n\n| Old | Table |\n| --- | --- |\n| row | 1 |\n\n## Contributing\nPlease help.\n"
  editor.readme_path.write_text(original_content, encoding="utf-8")
  results = {"abs": True, "complex_layer": False}
  success = editor.update_matrix(results)
  assert success is True
  new_text = editor.readme_path.read_text(encoding="utf-8")
  assert "# Title\nIntro text." in new_text
  assert "## ✅ Compatibility Matrix" in new_text
  assert "## Contributing\nPlease help." in new_text
  assert "| `t.abs` | `j.abs` | ✅ Passing |" in new_text
  assert "🧩 Plugin (Complex)" in new_text
  assert "| Old | Table |" not in new_text


def test_injection_at_end_of_file(editor):
  """Verifies the behavior of injection at end of file."""
  original_content = "# Title\n## ✅ Compatibility Matrix\nOld Data\n"
  editor.readme_path.write_text(original_content, encoding="utf-8")
  success = editor.update_matrix({})
  assert success is True
  new_text = editor.readme_path.read_text(encoding="utf-8")
  assert "## ✅ Compatibility Matrix" in new_text
  assert "| Category | PyTorch" in new_text
  assert "Old Data" not in new_text


def test_category_heuristics(editor):
  """Verifies the behavior of category heuristics."""
  editor.readme_path.write_text("## ✅ Compatibility Matrix\n", encoding="utf-8")
  results = {}
  success = editor.update_matrix(results)
  assert success is True
  content = editor.readme_path.read_text(encoding="utf-8")
  assert "| **Neural** | `t.nn.C`" in content


def test_null_variant_handling(editor):
  """Verifies the behavior of null variant handling."""
  editor.readme_path.write_text("## ✅ Compatibility Matrix\n", encoding="utf-8")
  results = {"unsupported": False}
  success = editor.update_matrix(results)
  assert success is True
  content = editor.readme_path.read_text(encoding="utf-8")
  assert "| `t.bad` | — |" in content
  assert "Untested/Fail" in content
