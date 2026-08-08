"""Central Logging and Console Utilities.

This module unifies the application's output mechanism using the Python standard
`logging` library, backed by `rich` for formatting.

It serves two primary purposes:

1.  **Standard Logging Integration**: Provides a configured `logging.Logger` and
    adapter functions (`log_success`, `log_warning`) that route to standard logging channels.
2.  **Environment Injection**: Implements a Proxy pattern for the Rich Console.
    This allows the output destination (stdout, file, or in-memory buffer) to be
    swapped at runtime via `set_console`. This is critical for WebAssembly (WASM)
    integration, where logs must be captured and returned to the browser context.
"""

from typing import Any

import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.style import Style
from rich.theme import Theme

# --- Constants & Configuration ---

# Define custom logging level for Success (higher than INFO, lower than WARNING)
SUCCESS_LEVEL_NUM = 25
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")


def _success(self, message: Any, *args: Any, **kwargs: Any) -> Any:  # type: ignore
  """Method injected into Logger to support logger.success().

  Args:
      self: The logger instance.
      message (Any): The message content to be logged.
      *args (Any): Variable length argument list passed to the underlying logger.
      **kwargs (Any): Arbitrary keyword arguments passed to the underlying logger.

  Returns:
      Any: The return value of the underlying log call.
  """
  if self.isEnabledFor(SUCCESS_LEVEL_NUM):
    self._log(SUCCESS_LEVEL_NUM, message, args, **kwargs)


# Patch basic Logger class
logging.Logger.success = _success  # type: ignore

# Define standard semantic colors for the CLI
_THEME = Theme(
  {
    "logging.level.success": "green",
    "info": "dim cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "green",
    "path": "bold blue",
    "code": "bold magenta",
  }
)


class _ConsoleProxy:
  """A Proxy wrapper around ``rich.console.Console``.

  This class maintains a reference to a 'backend' Console instance.
  All printing operations are forwarded to this backend. This allows the
  backend to be swapped at runtime (e.g., swapping ``stdout`` for ``io.StringIO``)
  while maintaining the module-level ``console`` object reference imported
  by other modules.

  When the backend changes, this proxy also reconfigures the Python ``logging``
  handlers to ensure that ``logging.info(...)`` writes to the new destination.
  """

  def __init__(self) -> None:
    """Initializes the proxy with a default Standard Output console.

    Returns:
        None
    """
    self._backend: Console = Console(theme=_THEME)
    self._configure_logging()

  def set_backend(self, new_console: Console) -> None:
    """Injects a new Console backend and updates logging handlers.

    Args:
        new_console (Console): The new Rich Console instance to use.

    Returns:
        None
    """
    self._backend = new_console
    self._configure_logging()

  def reset(self) -> None:
    """Resets the proxy to use a fresh standard output console.

    Returns:
        None
    """
    self._backend = Console(theme=_THEME)
    self._configure_logging()

  @property
  def backend(self) -> Console:
    """Access the raw backend console.

    Returns:
        Console: The currently active implementation.
    """
    return self._backend

  def _configure_logging(self) -> None:
    """Configures or reconfigures the standard Python logging library.

    This updates the logging root handler to route messages to the currently active
    backend console.

    Returns:
        None
    """
    # Remove existing RichHandlers to prevent duplicate logs/wrong destinations
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
      if isinstance(handler, RichHandler):
        root_logger.removeHandler(handler)

    # Create new handler coupled to the active console backend
    rich_handler = RichHandler(
      console=self._backend,
      show_time=False,
      omit_repeated_times=False,
      show_path=False,
      markup=True,
      rich_tracebacks=True,
    )

    # Ensure we capture everything INFO and above by default
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(rich_handler)

  def print(self, *args: Any, **kwargs: Any) -> None:
    """Forwards ``print`` calls to the active backend.

    Args:
        args (Any): Positional arguments for Rich print.
        kwargs (Any): Keyword arguments for Rich print.

    Returns:
        None
    """
    self._backend.print(*args, **kwargs)

  def get_style(self, name: str) -> Style:
    """Forwards ``get_style`` calls to the active backend.

    Args:
        name (str): The name of the style to look up.

    Returns:
        Style: The resolved style object.
    """
    return self._backend.get_style(name)

  def export_text(self, **kwargs: Any) -> str:
    """Forwards ``export_text`` (useful for log capturing).

    Args:
        kwargs (Any): Options passed to console.export_text.

    Returns:
        str: The captured text output.
    """
    return self._backend.export_text(**kwargs)

  def export_html(self, **kwargs: Any) -> str:
    """Forwards ``export_html`` (useful for web rendering).

    Args:
        kwargs (Any): Options passed to console.export_html.

    Returns:
        str: The captured HTML output.
    """
    return self._backend.export_html(**kwargs)

  def __getattr__(self, name: str) -> Any:
    """Fallback to forward any other attributes/methods to the backend.

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
  """Global helper to inject a specific console instance.

  This updates both the ``console`` proxy object and the standard ``logging``
  handlers to write to the new instance. Use this in Web/WASM contexts to
  redirect output to a capture buffer.

  Args:
      new_console (Console): The configured Rich console to use globally.

  Returns:
      None
  """
  console.set_backend(new_console)


def reset_console() -> None:
  """Global helper to reset logging and console to standard output.

  Returns:
      None
  """
  console.reset()


def get_console() -> Console:
  """Retrieves the currently active console backend.

  Returns:
      Console: The active Rich Console.
  """
  return console.backend


def log_info(msg: str) -> None:
  """Logs an informational message via standard logging.

  Args:
      msg (str): The message content. Can include rich markup like [bold].

  Returns:
      None
  """
  logging.info(f"ℹ️  {msg}", extra={"markup": True})


def log_success(msg: str) -> None:
  """Logs a success message via standard logging.

  Args:
      msg (str): The message content.

  Returns:
      None
  """
  logging.log(SUCCESS_LEVEL_NUM, f"✅ {msg}", extra={"markup": True})


def log_warning(msg: str) -> None:
  """Logs a warning message via standard logging.

  Args:
      msg (str): The message content.

  Returns:
      None
  """
  logging.warning(f"⚠️  {msg}", extra={"markup": True})


def log_error(msg: str) -> None:
  """Logs an error message via standard logging.

  Args:
      msg (str): The message content.

  Returns:
      None
  """
  logging.error(f"❌ {msg}", extra={"markup": True})
