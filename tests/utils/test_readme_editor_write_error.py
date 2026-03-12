def test_readme_editor_duck():
  from ml_switcheroo.utils.readme_editor import ReadmeEditor

  class DuckPath:
    def exists(self):
      return True

    def read_text(self, *args, **kwargs):
      return "## ✅ Compatibility Matrix\n"

    def write_text(self, *args, **kwargs):
      raise OSError("fail")

  ed = ReadmeEditor(type("Dummy", (), {"get_known_apis": lambda *args: {"op": {}}})(), DuckPath())
  assert ed.update_matrix({}) is False
