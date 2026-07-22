"""Test suite for the Readme Editor Write Error module."""


def test_readme_editor_duck():
  """Verifies the behavior of readme editor duck."""
  from ml_switcheroo.utils.readme_editor import ReadmeEditor

  class DuckPath:
    """Test suite for the Duck Path component."""

    def exists(self):
      """Helper to exists."""
      return True

    def read_text(self, *args, **kwargs):
      """Helper to read text."""
      return "## ✅ Compatibility Matrix\n"

    def write_text(self, *args, **kwargs):
      """Helper to write text."""
      raise OSError("fail")

  ed = ReadmeEditor(type("Dummy", (), {"get_known_apis": lambda *args: {"op": {}}})(), DuckPath())
  assert ed.update_matrix({}) is False
