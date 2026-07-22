"""Test suite for the Console module."""

import pytest
from rich.console import Console
from ml_switcheroo.utils.console import console, set_console, reset_console, log_info, log_error, get_console


@pytest.fixture(autouse=True)
def cleanup_console():
  """Helper to cleanup console."""
  reset_console()
  yield
  reset_console()


def test_console_singleton_proxy():
  """Verifies the behavior of console singleton proxy."""
  assert hasattr(console, "print")
  assert callable(console.print)
  assert hasattr(console, "export_text")
  assert isinstance(get_console(), Console)


def test_custom_console_injection():
  """Verifies the behavior of custom console injection."""
  capture_console = Console(record=True, file=None)
  set_console(capture_console)
  log_info("Captured Web Log")
  output = capture_console.export_text()
  assert "Captured Web Log" in output
  assert "ℹ️" in output


def test_reset_functionality():
  """Verifies the behavior of reset functionality."""
  original_backend = get_console()
  temp = Console()
  set_console(temp)
  assert get_console() is temp
  reset_console()
  current = get_console()
  assert current is not temp
  assert current is not original_backend
  assert isinstance(current, Console)


def test_logging_wrappers_format(capsys):
  """Verifies the behavior of logging wrappers format."""
  reset_console()
  log_info("InfoText")
  log_error("ErrorText")
  captured = capsys.readouterr()
  assert "InfoText" in captured.out
  assert "ErrorText" in captured.out
  assert "ℹ️" in captured.out
  assert "❌" in captured.out


def test_proxy_getattr_delegation():
  """Verifies the behavior of proxy getattr delegation."""
  width = console.width
  assert isinstance(width, int)
  assert width > 0
