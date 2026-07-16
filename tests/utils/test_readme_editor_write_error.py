"""Auto-generated doc."""


def test_readme_editor_duck():
  """Auto-generated doc."""
  from ml_switcheroo.utils.readme_editor import ReadmeEditor

  class DuckPath:
    """Auto-generated doc."""

    def exists(self):
      """Auto-generated doc."""
      return True

    def read_text(self, *args, **kwargs):
      """Auto-generated doc."""
      return "## ✅ Compatibility Matrix\n"

    def write_text(self, *args, **kwargs):
      """Auto-generated doc."""
      raise OSError("fail")

  ed = ReadmeEditor(type("Dummy", (), {"get_known_apis": lambda *args: {"op": {}}})(), DuckPath())
  assert ed.update_matrix({}) is False
