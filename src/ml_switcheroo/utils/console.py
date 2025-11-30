"""
Central Logging and Console Utilities.

This module provides a singleton interface to `rich.console.Console` configured
for the ml-switcheroo theme. It replaces standard print statements across
the application to ensure consistent, colored output.

It implements a Proxy pattern to allow the underlying Console instance to be
swapped (injected) at runtime. This "Injectable Console" architecture is crucial
for Web and WASM integration, allowing the application to redirect logs to
HTTP strings or browser capture buffers without modifying business logic.
"""

from typing import Any
from rich.console import Console
from rich.theme import Theme
from rich.style import Style

# Define standard semantic colors for the CLI
_THEME = Theme(
  {
    "info": "dim cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "green",
    "path": "bold blue",
    "code": "bold magenta",
  }
)


class _ConsoleProxy:
  """
  A Proxy wrapper around `rich.console.Console`.

  This class maintains a reference to a 'backend' Console instance.
  All printing operations are forwarded to this backend. This allows the
  backend to be swapped at runtime (e.g., swapping `stdout` for `io.StringIO`)
  while maintaining the module-level `console` object reference imported
  by other modules.

  Attributes:
      _backend (Console): The active Rich Console instance.
  """

  def __init__(self) -> None:
    """Initializes the proxy with a default Standard Output console."""
    self._backend: Console = Console(theme=_THEME)

  def set_backend(self, new_console: Console) -> None:
    """
    Injects a new Console backend.

    Args:
        new_console (Console): The new Rich Console instance to use.
    """
    self._backend = new_console

  def reset(self) -> None:
    """
    Resets the proxy to use a fresh standard output console.
    """
    self._backend = Console(theme=_THEME)

  @property
  def backend(self) -> Console:
    """
    Access the raw backend console.

    Returns:
        Console: The currently active implementation.
    """
    return self._backend

  def print(self, *args: Any, **kwargs: Any) -> None:
    """
    Forwards `print` calls to the active backend.

    Args:
        *args: Positional arguments for Rich print.
        **kwargs: Keyword arguments for Rich print.
    """
    self._backend.print(*args, **kwargs)

  def get_style(self, name: str) -> Style:
    """
    Forwards `get_style` calls to the active backend.

    Args:
        name (str): The name of the style to look up.

    Returns:
        Style: The resolved style object.
    """
    return self._backend.get_style(name)

  def export_text(self, **kwargs: Any) -> str:
    """
    Forwards `export_text` (useful for log capturing).

    Args:
        **kwargs: Options passed to console.export_text.

    Returns:
        str: The captured text output.
    """
    return self._backend.export_text(**kwargs)

  def export_html(self, **kwargs: Any) -> str:
    """
    Forwards `export_html` (useful for web rendering).

    Args:
        **kwargs: Options passed to console.export_html.

    Returns:
        str: The captured HTML output.
    """
    return self._backend.export_html(**kwargs)

  def __getattr__(self, name: str) -> Any:
    """
    Fallback to forward any other attributes/methods to the backend.

    Args:
        name (str): Attribute name.

    Returns:
        Any: The attribute from the backend console.
    """
    return getattr(self._backend, name)


# Singleton instance exposed to the application.
# Modules import this 'console' object. The underlying implementation
# can be changed via 'set_console'.
console = _ConsoleProxy()


def set_console(new_console: Console) -> None:
  """
  Global helper to inject a specific console instance.

  Use this in Web/WASM contexts to redirect output to a capturing console.

  Args:
      new_console (Console): The configured Rich console to use globally.
  """
  console.set_backend(new_console)


def reset_console() -> None:
  """
  Global helper to reset logging to standard output.
  """
  console.reset()


def get_console() -> Console:
  """
  Retrieves the currently active console backend.

  Returns:
      Console: The active Rich Console.
  """
  return console.backend


def log_info(msg: str) -> None:
  """
  Prints an informational message (prefixed with ℹ️).

  Args:
      msg (str): The text content will be formatted with [info] style.
  """
  console.print(f"ℹ️  [info]{msg}[/info]")


def log_success(msg: str) -> None:
  """
  Prints a success message (prefixed with ✅).

  Args:
      msg (str): The text content will be formatted with [success] style.
  """
  console.print(f"✅ [success]{msg}[/success]")


def log_warning(msg: str) -> None:
  """
  Prints a warning message (prefixed with ⚠️).

  Args:
      msg (str): The text content will be formatted with [warning] style.
  """
  console.print(f"⚠️  [warning]{msg}[/warning]")


def log_error(msg: str) -> None:
  """
  Prints an error message (prefixed with ❌).

  Args:
      msg (str): The text content will be formatted with [error] style.
  """
  console.print(f"❌ [error]{msg}[/error]")
