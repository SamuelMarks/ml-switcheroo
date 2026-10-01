"""Test suite for shell completion handlers."""

import io
from unittest.mock import patch

from ml_switcheroo.cli.handlers.completion import (
  generate_bash_completion,
  generate_fish_completion,
  generate_zsh_completion,
  handle_completion,
)


def test_generate_bash_completion() -> None:
  """Verifies bash completion script generation."""
  script = generate_bash_completion()
  assert "_ml_switcheroo_completion" in script
  assert "convert" in script
  assert "torch" in script
  assert "ml_switcheroo_ir" in script
  assert "wasm" not in script
  assert "cpp" not in script
  assert "completion" in script


def test_generate_zsh_completion() -> None:
  """Verifies zsh completion script generation."""
  script = generate_zsh_completion()
  assert "#compdef ml_switcheroo" in script
  assert "convert" in script
  assert "torch" in script
  assert "ml_switcheroo_ir" in script
  assert "wasm" not in script
  assert "cpp" not in script


def test_generate_fish_completion() -> None:
  """Verifies fish completion script generation."""
  script = generate_fish_completion()
  assert "complete -c ml_switcheroo" in script
  assert "convert" in script
  assert "torch" in script
  assert "ml_switcheroo_ir" in script
  assert "wasm" not in script
  assert "cpp" not in script


def test_handle_completion_success() -> None:
  """Verifies handle_completion for bash, zsh, and fish."""
  for shell in ["bash", "zsh", "fish"]:
    with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
      res = handle_completion(shell)
      assert res == 0
      assert len(mock_out.getvalue()) > 50


def test_handle_completion_invalid() -> None:
  """Verifies handle_completion for unsupported shell."""
  with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
    res = handle_completion("powershell")
    assert res == 1
    assert "Unsupported shell" in mock_err.getvalue()
