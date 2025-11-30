"""
Tests for automatic README documentation updating.

Verifies that:
1.  The `ReadmeEditor` correctly identifies the injection point.
2.  Tables are generated with correct formatting and icons.
3.  Logic gracefully handles missing files or missing headers.
4.  Subsequent headers are preserved (not overwritten).
"""

import pytest

from ml_switcheroo.utils.readme_editor import ReadmeEditor
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  """Mock manager returning deterministic API data."""

  def get_known_apis(self) -> dict:
    return {
      "abs": {"variants": {"torch": {"api": "t.abs"}, "jax": {"api": "j.abs"}}},
      "complex_layer": {"variants": {"torch": {"api": "t.nn.C"}, "jax": {"requires_plugin": "p"}}},
      "unsupported": {"variants": {"torch": {"api": "t.bad"}, "jax": None}},
    }


@pytest.fixture
def editor(tmp_path):
  """Creates an editor pointing to a temp file."""
  mgr = MockSemantics()
  # Create initial dummy file
  readme = tmp_path / "README.md"
  readme.write_text("# Test Project\nStart.", encoding="utf-8")
  return ReadmeEditor(mgr, readme)


def test_missing_file_returns_false(tmp_path):
  """
  Scenario: Path does not exist.
  Expectation: Returns False, logs error.
  """
  mgr = MockSemantics()
  missing_path = tmp_path / "NONEXISTENT.md"
  editor = ReadmeEditor(mgr, missing_path)

  success = editor.update_matrix({})
  assert success is False


def test_missing_section_header_returns_false(editor):
  """
  Scenario: README exists but lacks '## ✅ Compatibility Matrix'.
  Expectation: Returns False, no changes made.
  """
  editor.readme_path.write_text("# Title\n\nNo matrix here.", encoding="utf-8")

  success = editor.update_matrix({})
  assert success is False

  # Ensure file wasn't mangled
  assert editor.readme_path.read_text(encoding="utf-8") == "# Title\n\nNo matrix here."


def test_injection_between_headers(editor):
  """
  Scenario: Table section exists between two headers.
  Expectation: Old table replaced, surrounding content preserved.
  """
  original_content = """# Title
Intro text.

## ✅ Compatibility Matrix

| Old | Table |
| --- | --- |
| row | 1 |

## Contributing
Please help.
"""
  editor.readme_path.write_text(original_content, encoding="utf-8")

  # Results for verification column
  results = {"abs": True, "complex_layer": False}

  success = editor.update_matrix(results)
  assert success is True

  new_text = editor.readme_path.read_text(encoding="utf-8")

  # Check Persisted Structure
  assert "# Title\nIntro text." in new_text
  assert "## ✅ Compatibility Matrix" in new_text
  assert "## Contributing\nPlease help." in new_text

  # Check Table Content
  assert "| `t.abs` | `j.abs` | ✅ Passing |" in new_text
  assert "🧩 Plugin (Complex)" in new_text

  # Check Old Table Removed
  assert "| Old | Table |" not in new_text


def test_injection_at_end_of_file(editor):
  """
  Scenario: Table section is the last section in the file.
  Expectation: content appends correctly without crashing search regex.
  """
  original_content = """# Title
## ✅ Compatibility Matrix
Old Data
"""
  editor.readme_path.write_text(original_content, encoding="utf-8")

  success = editor.update_matrix({})
  assert success is True

  new_text = editor.readme_path.read_text(encoding="utf-8")
  assert "## ✅ Compatibility Matrix" in new_text
  # Should contain generated header
  assert "| Category | PyTorch" in new_text
  # Old data gone
  assert "Old Data" not in new_text


def test_category_heuristics(editor):
  """
  Verify _guess_category logic inside the table generation.
  """
  # Pre-seed the header is critical!
  editor.readme_path.write_text("## ✅ Compatibility Matrix\n", encoding="utf-8")

  # Mock semantics provides 'complex_layer' with api 't.nn.C'.
  # Heuristic checks 'nn' -> assigns 'Neural'.

  results = {}
  success = editor.update_matrix(results)
  assert success is True

  content = editor.readme_path.read_text(encoding="utf-8")

  # t.nn.C should correspond to Neural category
  # Row regex check is simpler: look for substring
  assert "| **Neural** | `t.nn.C`" in content


def test_null_variant_handling(editor):
  """
  Verify operations with no target (None) are rendered as dashes.
  """
  # Pre-seed header
  editor.readme_path.write_text("## ✅ Compatibility Matrix\n", encoding="utf-8")

  results = {"unsupported": False}
  success = editor.update_matrix(results)
  assert success is True

  content = editor.readme_path.read_text(encoding="utf-8")

  # Check unsupported row
  # Torch has `t.bad`, JAX is None -> `—`
  assert "| `t.bad` | — |" in content
  assert "Untested/Fail" in content
