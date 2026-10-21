"""Test suite for the Console module."""

import typing

import pytest
from rich.console import Console
from rich.style import Style

from ml_switcheroo.utils.console import console, get_console, log_error, log_info, reset_console, set_console


@pytest.fixture(autouse=True)
def cleanup_console() -> typing.Generator[None, None, None]:
  """Helper to cleanup console.

  Returns:
      Computed test result or fixture value.
  """
  reset_console()
  yield
  reset_console()


def test_console_singleton_proxy() -> None:
  """Verifies the behavior of console singleton proxy."""
  assert hasattr(console, "print")
  assert callable(console.print)
  assert hasattr(console, "export_text")
  assert isinstance(get_console(), Console)


def test_console_proxy_methods() -> None:
  """Verifies the behavior of explicitly proxied methods in SingletonConsoleProxy."""
  capture_console: Console = Console(record=True, file=None)
  set_console(capture_console)
  console.print("test_print")
  out: str = console.export_text(clear=False)
  assert "test_print" in out

  html: str = console.export_html(clear=False)
  assert "test_print" in html

  # Test get_style
  style: Style = console.get_style("bold")
  assert style is not None


def test_logging_wrappers_format(capsys: pytest.CaptureFixture[str]) -> None:
  """Verifies the behavior of logging wrappers format.

  Args:
      capsys: Test argument or fixture.
  """
  reset_console()
  log_info("InfoText")
  from ml_switcheroo.utils.console import log_success, log_warning

  log_warning("WarningText")
  log_success("SuccessText")
  log_error("ErrorText")
  captured = capsys.readouterr()
  assert "InfoText" in captured.out
  assert "WarningText" in captured.out
  assert "SuccessText" in captured.out
  assert "ErrorText" in captured.out
  assert "ℹ️" in captured.out
  assert "⚠️" in captured.out
  assert "✅" in captured.out
  assert "❌" in captured.out


def test_success_logger() -> None:
  """Verifies the custom success log level."""
  import logging

  logger: logging.Logger = logging.getLogger("test_logger")
  # Test the patched method exists
  assert hasattr(logger, "success")

  # Ensure it can log
  logger.setLevel(logging.DEBUG)
  with pytest.MonkeyPatch.context() as m:
    logs: list[tuple[int, str]] = []
    m.setattr(logger, "_log", lambda level, msg, args, **kwargs: logs.append((level, msg)))
    getattr(logger, "success")("my_success_message")
    assert len(logs) == 1
    assert logs[0][1] == "my_success_message"
    assert logs[0][0] == 25

    # Ensure it skips logging when level is too high
    logger.setLevel(logging.ERROR)
    logs.clear()
    getattr(logger, "success")("should_not_log")
    assert len(logs) == 0


def test_proxy_getattr_delegation() -> None:
  """Verifies the behavior of proxy getattr delegation."""
  width = console.width
  assert isinstance(width, int)
  assert width > 0
