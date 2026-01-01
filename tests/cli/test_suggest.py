"""
Tests for Suggest Command (Prompt Generator).

Verifies:
1. Live introspection of existing modules.
2. Correct formatting of prompt template containing:
   - Signature
   - Docstring
   - JSON Schema
3. Error handling for missing modules.
"""

import pytest
import sys
from unittest.mock import MagicMock, patch
from ml_switcheroo.cli.handlers.suggest import handle_suggest, _inspect_live_object


@pytest.fixture
def mock_module():
  """Injects a fake module into sys.modules for inspection."""
  mod_name = "test_pkg"

  def sample_func(x: int, y: int = 1) -> int:
    """A sample function docstring."""
    return x + y

  mock_mod = MagicMock()
  mock_mod.sample_func = sample_func

  with patch.dict(sys.modules, {mod_name: mock_mod}):
    yield mod_name


def test_inspect_valid_function(mock_module):
  """
  Scenario: Object exists and has signature/docstring.
  """
  res = _inspect_live_object("test_pkg.sample_func")

  assert "sample_func(x: int, y: int = 1)" in res["signature"] or "(x: int, y: int = 1)" in res["signature"]
  assert res["docstring"] == "A sample function docstring."
  assert res["kind"] == "function"


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

  # Log uses rich/logging which usually goes to stderr or custom backend,
  # capture log via capsys might miss if purely using `log_error`.
  # For integration test we check return value is enough.
