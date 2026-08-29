"""Docstring."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from ml_switcheroo.utils.readme_editor import ReadmeEditor


def test_readme_editor_init() -> None:
  """Docstring."""
  editor: ReadmeEditor = ReadmeEditor(MagicMock(), Path("fake.md"))
  assert editor.readme_path.name == "fake.md"


def test_update_matrix_no_readme() -> None:
  """Docstring."""
  editor: ReadmeEditor = ReadmeEditor(MagicMock(), Path("does_not_exist.md"))
  with patch("pathlib.Path.exists", return_value=False):
    assert editor.update_matrix({}) is False


def test_update_matrix_no_heading() -> None:
  """Docstring."""
  editor: ReadmeEditor = ReadmeEditor(MagicMock(), Path("fake.md"))
  content: str = "# Title\nSome content."
  with patch("pathlib.Path.exists", return_value=True):
    with patch("pathlib.Path.read_text", return_value=content):
      assert editor.update_matrix({}) is False


def test_update_matrix_success() -> None:
  """Docstring."""
  semantics: MagicMock = MagicMock()
  # Mocking standard map format
  sm: MagicMock = MagicMock()
  sm.name = "Abs"
  sm.api = "torch.abs"
  sm.framework = "torch"
  sm.kind = "func"
  sm.get_variant.return_value = {"api": "jnp.abs"}
  sm.to_dict.return_value = {"abstract": "Abs", "kind": "func"}

  semantics.get_known_apis.return_value = {"Abs": sm}

  editor: ReadmeEditor = ReadmeEditor(semantics, Path("fake.md"))

  content: str = "# Title\n## ✅ Compatibility Matrix\nOld table\n## Next section"
  with patch("pathlib.Path.exists", return_value=True):
    with patch("pathlib.Path.read_text", return_value=content):
      with patch("pathlib.Path.write_text"):
        assert editor.update_matrix({"Abs": True}) is True


def test_update_matrix_success_no_next_heading() -> None:
  """Docstring."""
  semantics: MagicMock = MagicMock()
  # Mocking standard map format
  sm: MagicMock = MagicMock()
  sm.name = "Abs"
  sm.api = "torch.abs"
  sm.framework = "torch"
  sm.kind = "func"
  sm.get_variant.return_value = {"api": "jnp.abs"}
  sm.to_dict.return_value = {"abstract": "Abs", "kind": "func"}

  semantics.get_known_apis.return_value = {"Abs": sm}

  editor: ReadmeEditor = ReadmeEditor(semantics, Path("fake.md"))

  content: str = "# Title\n## ✅ Compatibility Matrix\nOld table"
  with patch("pathlib.Path.exists", return_value=True):
    with patch("pathlib.Path.read_text", return_value=content):
      with patch("pathlib.Path.write_text"):
        assert editor.update_matrix({"Abs": True}) is True


def test_update_matrix_read_error() -> None:
  """Docstring."""
  editor: ReadmeEditor = ReadmeEditor(MagicMock(), Path("fake.md"))
  with patch("pathlib.Path.exists", return_value=True):
    with patch("pathlib.Path.read_text", side_effect=OSError("Read err")):
      assert editor.update_matrix({}) is False


def test_update_matrix_write_error() -> None:
  """Docstring."""
  semantics: MagicMock = MagicMock()
  sm: MagicMock = MagicMock()
  sm.name = "Abs"
  sm.api = "torch.abs"
  sm.framework = "torch"
  sm.kind = "func"
  sm.get_variant.return_value = {"api": "jnp.abs"}
  sm.to_dict.return_value = {"abstract": "Abs", "kind": "func"}
  semantics.get_known_apis.return_value = {"Abs": sm}

  editor: ReadmeEditor = ReadmeEditor(semantics, Path("fake.md"))
  content: str = "# Title\n## ✅ Compatibility Matrix\nOld table\n## Next section"

  with patch("pathlib.Path.exists", return_value=True):
    with patch("pathlib.Path.read_text", return_value=content):
      with patch("pathlib.Path.write_text", side_effect=OSError("Write err")):
        assert editor.update_matrix({"Abs": True}) is False


def test_generate_markdown_table_none_variant() -> None:
  """Docstring."""
  semantics: MagicMock = MagicMock()
  sm: MagicMock = MagicMock()
  sm.name = "Abs"
  sm.api = "torch.abs"
  sm.framework = "torch"
  sm.kind = "func"
  sm.get_variant.return_value = None  # This will hit the `jax_variant is None` check
  sm.to_dict.return_value = {"abstract": "Abs", "kind": "func"}
  semantics.get_known_apis.return_value = {"Abs": sm}

  editor: ReadmeEditor = ReadmeEditor(semantics, Path("fake.md"))
  content: str = "# Title\n## ✅ Compatibility Matrix\nOld table\n## Next section"

  with patch("pathlib.Path.exists", return_value=True):
    with patch("pathlib.Path.read_text", return_value=content):
      with patch("pathlib.Path.write_text"):
        assert editor.update_matrix({"Abs": True}) is True


def test_guess_category() -> None:
  """Docstring."""
  from ml_switcheroo.utils.readme_editor import _guess_category

  assert _guess_category("torch.nn.Linear", None) == "Neural"
  assert _guess_category("torch.add", {"requires_plugin": "foo"}) == "Special"
  assert _guess_category("torch.add", None) == "Math"
