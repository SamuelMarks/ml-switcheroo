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

  ed = ReadmeEditor(type("Dummy", (), {"get_known_apis": lambda *args: {"op": {}}})(), DuckPath())  # type: ignore
  assert ed.update_matrix({}) is False


def test_readme_editor_read_error():
  """Verifies the behavior of readme editor duck."""
  from ml_switcheroo.utils.readme_editor import ReadmeEditor

  class DuckPath:
    """Mock Path for testing read error."""

    def exists(self):
      """Check if exists."""
      return True

    def read_text(self, *args, **kwargs):
      """Read text."""
      raise OSError("fail")

  ed = ReadmeEditor(type("Dummy", (), {"get_known_apis": lambda *args: {"op": {}}})(), DuckPath())  # type: ignore
  assert ed.update_matrix({}) is False


def test_readme_editor_no_map():
  """Verifies the behavior of readme editor duck."""
  from ml_switcheroo.utils.readme_editor import ReadmeEditor
  import markdown_it

  class DuckPath:
    """Mock Path for testing no map."""

    def exists(self):
      """Check if exists."""
      return True

    def read_text(self, *args, **kwargs):
      """Read text."""
      return "## ✅ Compatibility Matrix\n"

    def write_text(self, *args, **kwargs):
      """Write text."""
      return True

  # Patch the markdown parser to return tokens with no map
  original_parse = markdown_it.MarkdownIt.parse

  def mock_parse(self, content):
    """Mock parse returning no map."""
    tokens = original_parse(self, content)
    for t in tokens:
      t.map = None
    return tokens

  markdown_it.MarkdownIt.parse = mock_parse
  ed = ReadmeEditor(type("Dummy", (), {"get_known_apis": lambda *args: {"op": {}}})(), DuckPath())  # type: ignore
  assert ed.update_matrix({}) is False
  markdown_it.MarkdownIt.parse = original_parse


def test_guess_category_missing_plugin():
  """Verifies guess_category."""
  from ml_switcheroo.utils.readme_editor import _guess_category

  assert _guess_category("something_else", {}) == "Math"
  assert _guess_category("something_else", {"requires_plugin": True}) == "Special"
