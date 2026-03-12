"""
Tests for Suggest Command (Prompt Generator).

Verifies:
1. Live introspection of existing modules.
2. Correct formatting of prompt template containing:
   - Signature
   - Docstring
   - JSON Schema
3. Error handling for missing modules.
4. Wildcard expansion handling.
5. File output batching logic.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from ml_switcheroo.cli.handlers.suggest import handle_suggest, _inspect_live_object, _extract_metadata


@pytest.fixture
def mock_module():
  """Injects a fake module into sys.modules for inspection."""
  mod_name = "test_pkg"

  def sample_func(x: int, y: int = 1) -> int:
    """A sample function docstring."""
    return x + y

  class SampleClass:
    """A sample class."""

    pass

  mock_mod = MagicMock()
  mock_mod.sample_func = sample_func
  mock_mod.SampleClass = SampleClass
  # Simulate getmembers iteration behavior
  mock_mod.__name__ = mod_name

  with patch.dict(sys.modules, {mod_name: mock_mod}):
    # Order of getmembers is usually name sorted, but we return explicit list for predictable chunking test.
    # We add 2 functions. If batch_size=1, should create 2 files.
    members = [
      ("SampleClass", SampleClass),
      ("sample_func", sample_func),
      ("_private", MagicMock()),
    ]
    with patch("inspect.getmembers", return_value=members):
      yield mod_name


def test_inspect_valid_function(mock_module):
  """
  Scenario: Object exists and has signature/docstring.
  """
  res = _inspect_live_object("test_pkg.sample_func")

  assert "sample_func(x: int, y: int = 1)" in res["signature"] or "(x: int, y: int = 1)" in res["signature"]
  assert res["docstring"] == "A sample function docstring."
  assert res["kind"] == "function"


def test_extract_metadata_class():
  """Function docstring."""

  class C:
    """Class Doc."""

    pass

  res = _extract_metadata(C)
  assert res["kind"] == "class"
  assert res["docstring"] == "Class Doc."


def test_handle_suggest_output(mock_module, capsys):
  """
  Scenario: User runs `ml_switcheroo suggest test_pkg.sample_func`.
  Expectation: Stdout contains structured prompt with schema.
  """
  ret = handle_suggest("test_pkg.sample_func")
  assert ret == 0

  captured = capsys.readouterr().out

  # Check Template Sections
  assert "--- TARGET OPERATION ---" in captured
  assert "Name: test_pkg.sample_func" in captured
  assert "Signature:" in captured
  assert "Docstring:" in captured

  # Check Schema Presence
  assert '"$defs":' in captured or '"definitions":' in captured or '"properties":' in captured
  assert "OperationDef" in captured

  # Check Examples
  assert "--- ONE-SHOT EXAMPLE ---" in captured
  assert 'operation: "Abs"' in captured


def test_handle_suggest_missing_module(capsys):
  """
  Scenario: User requests non-existent module.
  Expectation: Log error and return non-zero exit code.
  """
  ret = handle_suggest("ghost.module.func")
  assert ret == 1


def test_handle_suggest_wildcard_stdout(mock_module, capsys):
  """
  Scenario: User runs `suggest test_pkg.*` without out_dir.
  Expectation:
    - Prints Header once.
    - Prints 2 ops (SampleClass, sample_func).
    - Prints Footer once.
  """
  ret = handle_suggest("test_pkg.*")
  assert ret == 0

  captured = capsys.readouterr().out
  # Must find both ops
  assert "Name: test_pkg.sample_func" in captured
  assert "Name: test_pkg.SampleClass" in captured

  # Header should appear once
  assert captured.count("You are an expert AI assistant") == 1

  # Footer should appear once
  assert captured.count("--- INSTRUCTIONS ---") == 1

  # Should have multiple TARGET OPERATION blocks
  assert captured.count("--- TARGET OPERATION ---") == 2

  # Should not analyze private
  assert "_private" not in captured


def test_handle_suggest_wildcard_file_batching(mock_module, tmp_path):
  """
  Scenario: `suggest test_pkg.* --out-dir /tmp --batch-size 1`.
  Expectation:
    - 2 public items in mock module.
    - Batch size 1 -> 2 output files.
  """
  out_dir = tmp_path / "prompts"
  ret = handle_suggest("test_pkg.*", out_dir=out_dir, batch_size=1)

  assert ret == 0

  # Check files
  files = sorted(list(out_dir.glob("*.md")))
  assert len(files) == 2

  # Check file 1 content
  c1 = files[0].read_text()
  assert "You are an expert AI assistant" in c1  # Header
  assert "INSTRUCTIONS" in c1  # Footer
  # Only one op per file
  assert c1.count("--- TARGET OPERATION ---") == 1

  # Check file 2 content
  c2 = files[1].read_text()
  assert "You are an expert AI assistant" in c2  # Header
  assert c2.count("--- TARGET OPERATION ---") == 1

  # Ensure filenames are ordered
  assert files[0].name == "suggest_test_pkg_001.md"
  assert files[1].name == "suggest_test_pkg_002.md"


def test_suggest_import_error():
  from ml_switcheroo.cli.handlers.suggest import handle_suggest

  # Line 67-69
  assert handle_suggest("invalid_module.*") == 1


def test_suggest_skip_module_no_targets():
  from ml_switcheroo.cli.handlers.suggest import handle_suggest
  import types

  with patch("ml_switcheroo.cli.handlers.suggest.importlib.import_module") as mock_imp:
    mock_mod = type("MockMod", (), {"submod": types.ModuleType("submod")})()
    mock_imp.return_value = mock_mod
    # Line 62 (skip module), Line 80-81 (no targets)
    assert handle_suggest("foo.*") == 1


def test_suggest_inspect_exception():
  from ml_switcheroo.cli.handlers.suggest import handle_suggest

  with patch("ml_switcheroo.cli.handlers.suggest.importlib.import_module") as mock_imp:
    mock_mod = type("MockMod", (), {"sub": type("MockObj", (), {})()})()
    mock_imp.return_value = mock_mod
  with (
    patch("ml_switcheroo.cli.handlers.suggest._extract_metadata", side_effect=Exception("error")),
    patch("ml_switcheroo.cli.handlers.suggest.importlib.import_module", return_value=mock_mod),
  ):
    assert handle_suggest("foo.*") == 1


def test_suggest_fallback_signature():
  from ml_switcheroo.cli.handlers.suggest import _extract_metadata

  # Line 145-147 (C-extension fallback)
  assert _extract_metadata(str)["signature"] == "Unknown Signature"


def test_inspect_live_object_invalid_path():
  from ml_switcheroo.cli.handlers.suggest import _inspect_live_object
  import pytest

  with pytest.raises(ImportError):
    _inspect_live_object("sys")
