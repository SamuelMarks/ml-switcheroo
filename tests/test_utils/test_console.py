"""
Tests for Centralized Logging Utility and Injection Mechanics.

Verifies:
1. Singleton Proxy correctness.
2. Injection capabilities (`set_console`).
3. Standard logging wrappers.
"""

import pytest
from rich.console import Console
from ml_switcheroo.utils.console import (
  console,
  set_console,
  reset_console,
  log_info,
  log_error,
  get_console,
)


@pytest.fixture(autouse=True)
def cleanup_console():
  """Ensures console is reset to stdout after every test."""
  reset_console()
  yield
  reset_console()


def test_console_singleton_proxy():
  """
  Verify `console` acts as a proxy to a real Rich console.
  """
  # Should define print method
  assert hasattr(console, "print")
  assert callable(console.print)

  # Should define export_text (forwarded)
  assert hasattr(console, "export_text")

  # Should have a backend
  assert isinstance(get_console(), Console)


def test_custom_console_injection():
  """
  Verify we can inject a capturing console and retrieve logs.
  This simulates how a Web API would capture logs for a response.
  """
  # 1. Create a capturing console
  capture_console = Console(record=True, file=None)

  # 2. Inject it
  set_console(capture_console)

  # 3. Use the global logger
  log_info("Captured Web Log")

  # 4. Export from our capture instance
  output = capture_console.export_text()

  assert "Captured Web Log" in output
  assert "ℹ️" in output  # Unicode preserved


def test_reset_functionality():
  """
  Verify `reset_console` restores default behavior.
  """
  original_backend = get_console()

  # Inject temporary
  temp = Console()
  set_console(temp)
  assert get_console() is temp

  # Reset
  reset_console()
  current = get_console()

  assert current is not temp
  assert current is not original_backend  # It creates a FRESH instance
  assert isinstance(current, Console)


def test_logging_wrappers_format(capsys):
  """
  Verify semantic wrappers utilize the theme colors and prefixes.
  Note: We rely on capsys capturing stdout from the default console.
  """
  # Ensure default
  reset_console()

  log_info("InfoText")
  log_error("ErrorText")

  captured = capsys.readouterr()

  # Check content
  assert "InfoText" in captured.out
  assert "ErrorText" in captured.out

  # Check prefixes
  assert "ℹ️" in captured.out
  assert "❌" in captured.out


def test_proxy_getattr_delegation():
  """
  Verify that accessing attributes not explicitly defined on the proxy
  falls through to the backend via __getattr__.
  """
  # 'width' is a property of Rich Console, not defined on _ConsoleProxy
  width = console.width
  assert isinstance(width, int)
  assert width > 0
