"""Test suite for the Console module."""

import typing

import pytest
from rich.console import Console
from rich.style import Style

from ml_switcheroo.utils.console import (
  _SuccessLogger,
  console,
  get_console,
  log_error,
  log_info,
  reset_console,
  set_console,
)


@pytest.fixture(autouse=True)
def cleanup_console() -> typing.Generator[None, None, None]:
  """Create a configured `typing.Generator[None, None, None]` fixture for console tests.

  Yields:
      Configured `cleanup_console` fixture value (typing.Generator[None, None, None]).
  """
  reset_console()
  yield
  reset_console()


def test_console_singleton_proxy() -> None:
  """Verify `callable` handles console singleton proxy as expected."""
  assert callable(console.print)
  assert callable(console.export_text)
  assert isinstance(get_console(), Console)
  assert get_console() is console.backend
  assert console.width == get_console().width

  buf_console = Console(record=True, file=None)
  set_console(buf_console)
  assert get_console() is buf_console
  console.print("singleton_proxy_check")
  assert "singleton_proxy_check" in console.export_text(clear=False)


def test_console_proxy_methods() -> None:
  """Verify `Console` emits `test_print` for proxy methods."""
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
  """Verify `reset_console` emits `InfoText` for logging wrappers format.

  Args:
      capsys: Pytest stdout and stderr capture fixture.
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
  assert "\u2139\ufe0f" in captured.out
  assert "⚠️" in captured.out
  assert "✅" in captured.out
  assert "❌" in captured.out


def test_success_logger() -> None:
  """Verifies the custom success log level."""
  import logging

  logger: logging.Logger = logging.getLogger("test_logger")
  assert isinstance(logger, _SuccessLogger)

  # Ensure it can log
  logger.setLevel(logging.DEBUG)
  with pytest.MonkeyPatch.context() as m:
    logs: list[tuple[int, str]] = []
    m.setattr(logger, "_log", lambda level, msg, args, **kwargs: logs.append((level, msg)))
    logger.success("my_success_message")
    assert len(logs) == 1
    assert logs[0][1] == "my_success_message"
    assert logs[0][0] == 25

    # Ensure it skips logging when level is too high
    logger.setLevel(logging.ERROR)
    logs.clear()
    logger.success("should_not_log")
    assert len(logs) == 0


def test_proxy_getattr_delegation() -> None:
  """Verify proxy getattr delegation produces the expected state."""
  width = console.width
  assert isinstance(width, int)
  assert width > 0
