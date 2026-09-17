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
  semantics.get_known_apis.return_value = {
    "Abs": {
      "variants": {
        "torch": {"api": "torch.abs"},
        "jax": None,
      }
    }
  }

  editor: ReadmeEditor = ReadmeEditor(semantics, Path("fake.md"))
  content: str = "# Title\n## ✅ Compatibility Matrix\nOld table\n## Next section"

  with patch("pathlib.Path.exists", return_value=True):
    with patch("pathlib.Path.read_text", return_value=content):
      with patch("pathlib.Path.write_text"):
        assert editor.update_matrix({"Abs": True}) is True


def test_update_matrix_target_map_none() -> None:
  """Verifies error handling when header line mapping is None."""
  token_target_heading: MagicMock = MagicMock()
  token_target_heading.type = "heading_open"
  token_target_heading.map = None

  token_target_inline: MagicMock = MagicMock()
  token_target_inline.type = "inline"
  token_target_inline.content = "## ✅ Compatibility Matrix"

  mock_tokens: list[MagicMock] = [token_target_heading, token_target_inline]

  semantics: MagicMock = MagicMock()
  semantics.get_known_apis.return_value = {}
  editor: ReadmeEditor = ReadmeEditor(semantics, Path("fake.md"))
  content: str = "# Title\n## ✅ Compatibility Matrix\nOld table"

  with patch("pathlib.Path.exists", return_value=True):
    with patch("pathlib.Path.read_text", return_value=content):
      with patch("markdown_it.MarkdownIt.parse", return_value=mock_tokens):
        assert editor.update_matrix({}) is False


def test_guess_category() -> None:
  """Docstring."""
  from ml_switcheroo.utils.readme_editor import _guess_category

  assert _guess_category("torch.nn.Linear", None) == "Neural"
  assert _guess_category("torch.Conv2d", None) == "Neural"
  assert _guess_category("torch.add", {"requires_plugin": "foo"}) == "Special"
  assert _guess_category("torch.add", None) == "Math"


def test_update_matrix_heading_branches_and_table_formatting() -> None:
  """Verifies branch 74->72 (heading not followed by inline) and 102->105 (next heading with None map)."""
  token_empty_heading: MagicMock = MagicMock()
  token_empty_heading.type = "heading_open"

  token_non_inline: MagicMock = MagicMock()
  token_non_inline.type = "heading_close"

  token_target_heading: MagicMock = MagicMock()
  token_target_heading.type = "heading_open"
  token_target_heading.map = [1, 2]

  token_target_inline: MagicMock = MagicMock()
  token_target_inline.type = "inline"
  token_target_inline.content = "## ✅ Compatibility Matrix"

  token_dummy_close: MagicMock = MagicMock()
  token_dummy_close.type = "heading_close"

  token_next_heading: MagicMock = MagicMock()
  token_next_heading.type = "heading_open"
  token_next_heading.map = None

  mock_tokens: list[MagicMock] = [
    token_empty_heading,
    token_non_inline,
    token_target_heading,
    token_target_inline,
    token_dummy_close,
    token_next_heading,
  ]

  semantics: MagicMock = MagicMock()
  semantics.get_known_apis.return_value = {
    "NoTorchApi": {"variants": {"torch": {}, "jax": {"api": "jax.foo"}}},
    "PluginOp": {"variants": {"torch": {"api": "torch.foo"}, "jax": {"requires_plugin": "plug"}}},
    "NoJaxApi": {"variants": {"torch": {"api": "torch.bar"}, "jax": {}}},
  }

  editor: ReadmeEditor = ReadmeEditor(semantics, Path("fake.md"))
  content: str = "# Line 0\n# Line 1\n# Line 2"

  with patch("pathlib.Path.exists", return_value=True):
    with patch("pathlib.Path.read_text", return_value=content):
      with patch("markdown_it.MarkdownIt.parse", return_value=mock_tokens):
        with patch("pathlib.Path.write_text"):
          assert editor.update_matrix({"NoTorchApi": False, "PluginOp": False, "NoJaxApi": False}) is True
