"""Docstring."""

from ml_switcheroo.utils.readme_editor import ReadmeEditor
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_readme_editor_init():
  """Docstring."""
  editor = ReadmeEditor(MagicMock(), Path("fake.md"))
  assert editor.readme_path.name == "fake.md"


def test_update_matrix_no_readme():
  """Docstring."""
  editor = ReadmeEditor(MagicMock(), Path("does_not_exist.md"))
  with patch("pathlib.Path.exists", return_value=False):
    assert editor.update_matrix({}) is False


def test_update_matrix_no_heading():
  """Docstring."""
  editor = ReadmeEditor(MagicMock(), Path("fake.md"))
  content = "# Title\nSome content."
  with patch("pathlib.Path.exists", return_value=True):
    with patch("pathlib.Path.read_text", return_value=content):
      assert editor.update_matrix({}) is False


def test_update_matrix_success():
  """Docstring."""
  semantics = MagicMock()
  # Mocking standard map format
  sm = MagicMock()
  sm.name = "Abs"
  sm.api = "torch.abs"
  sm.framework = "torch"
  sm.kind = "func"
  sm.get_variant.return_value = {"api": "jnp.abs"}
  sm.to_dict.return_value = {"abstract": "Abs", "kind": "func"}

  semantics.get_known_apis.return_value = {"Abs": sm}

  editor = ReadmeEditor(semantics, Path("fake.md"))

  content = "# Title\n## ✅ Compatibility Matrix\nOld table\n## Next section"
  with patch("pathlib.Path.exists", return_value=True):
    with patch("pathlib.Path.read_text", return_value=content):
      with patch("pathlib.Path.write_text"):
        assert editor.update_matrix({"Abs": True}) is True


def test_update_matrix_success_no_next_heading():
  """Docstring."""
  semantics = MagicMock()
  # Mocking standard map format
  sm = MagicMock()
  sm.name = "Abs"
  sm.api = "torch.abs"
  sm.framework = "torch"
  sm.kind = "func"
  sm.get_variant.return_value = {"api": "jnp.abs"}
  sm.to_dict.return_value = {"abstract": "Abs", "kind": "func"}

  semantics.get_known_apis.return_value = {"Abs": sm}

  editor = ReadmeEditor(semantics, Path("fake.md"))

  content = "# Title\n## ✅ Compatibility Matrix\nOld table"
  with patch("pathlib.Path.exists", return_value=True):
    with patch("pathlib.Path.read_text", return_value=content):
      with patch("pathlib.Path.write_text"):
        assert editor.update_matrix({"Abs": True}) is True


def test_update_matrix_read_error():
  """Docstring."""
  editor = ReadmeEditor(MagicMock(), Path("fake.md"))
  with patch("pathlib.Path.exists", return_value=True):
    with patch("pathlib.Path.read_text", side_effect=OSError("Read err")):
      assert editor.update_matrix({}) is False


def test_update_matrix_write_error():
  """Docstring."""
  semantics = MagicMock()
  sm = MagicMock()
  sm.name = "Abs"
  sm.api = "torch.abs"
  sm.framework = "torch"
  sm.kind = "func"
  sm.get_variant.return_value = {"api": "jnp.abs"}
  sm.to_dict.return_value = {"abstract": "Abs", "kind": "func"}
  semantics.get_known_apis.return_value = {"Abs": sm}

  editor = ReadmeEditor(semantics, Path("fake.md"))
  content = "# Title\n## ✅ Compatibility Matrix\nOld table\n## Next section"

  with patch("pathlib.Path.exists", return_value=True):
    with patch("pathlib.Path.read_text", return_value=content):
      with patch("pathlib.Path.write_text", side_effect=OSError("Write err")):
        assert editor.update_matrix({"Abs": True}) is False


def test_generate_markdown_table_none_variant():
  """Docstring."""
  semantics = MagicMock()
  sm = MagicMock()
  sm.name = "Abs"
  sm.api = "torch.abs"
  sm.framework = "torch"
  sm.kind = "func"
  sm.get_variant.return_value = None  # This will hit the `jax_variant is None` check
  sm.to_dict.return_value = {"abstract": "Abs", "kind": "func"}
  semantics.get_known_apis.return_value = {"Abs": sm}

  editor = ReadmeEditor(semantics, Path("fake.md"))
  content = "# Title\n## ✅ Compatibility Matrix\nOld table\n## Next section"

  with patch("pathlib.Path.exists", return_value=True):
    with patch("pathlib.Path.read_text", return_value=content):
      with patch("pathlib.Path.write_text"):
        assert editor.update_matrix({"Abs": True}) is True


def test_guess_category():
  """Docstring."""
  from ml_switcheroo.utils.readme_editor import _guess_category

  assert _guess_category("torch.nn.Linear", None) == "Neural"
  assert _guess_category("torch.add", {"requires_plugin": "foo"}) == "Special"
  assert _guess_category("torch.add", None) == "Math"
