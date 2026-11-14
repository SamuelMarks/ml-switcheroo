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

import logging
import typing

from pydantic import JsonValue
from rich.console import Console, RenderableType
from rich.logging import RichHandler
from rich.style import Style
from rich.theme import Theme

ConsolePrintArg: typing.TypeAlias = RenderableType | JsonValue

# --- Constants & Configuration ---

# Define custom logging level for Success (higher than INFO, lower than WARNING)
SUCCESS_LEVEL_NUM = 25
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")


def _success(
  self: "logging.Logger",
  message: str | JsonValue,
  *args: str | int | float | bool | None,
  **kwargs: JsonValue,
) -> None:
  """Output injected into Logger to support logger.success().

  Args:
      self: The logger instance.
      message: The message content to be logged.
      *args: Variable length argument list passed to the underlying logger.
      **kwargs: Arbitrary keyword arguments passed to the underlying logger.

  Returns:
      None
  """
  if self.isEnabledFor(SUCCESS_LEVEL_NUM):
    self.log(SUCCESS_LEVEL_NUM, str(message), *args)


class _SuccessLogger(logging.Logger):
  """Custom Logger class providing a ``.success()`` method."""

  success = _success


logging.setLoggerClass(_SuccessLogger)

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
    """Initialize the proxy with a default Standard Output console.

    Returns:
        None
    """
    self._backend: Console = Console(theme=_THEME)
    self._configure_logging()

  def set_backend(self, new_console: Console) -> None:
    """Inject a new Console backend and updates logging handlers.

    Args:
        new_console (Console): The new Rich Console instance to use.

    Returns:
        None
    """
    self._backend = new_console
    self._configure_logging()

  def reset(self) -> None:
    """Reset the proxy to use a fresh standard output console.

    Returns:
        None
    """
    self._backend = Console(theme=_THEME)
    self._configure_logging()

  @property
  def backend(self) -> Console:
    """Acces the raw backend console.

    Returns:
        Console: The currently active implementation.
    """
    return self._backend

  def _configure_logging(self) -> None:
    """Configure or reconfigures the standard Python logging library.

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

  def print(self, *args: ConsolePrintArg, **kwargs: JsonValue) -> None:
    """Forward ``print`` calls to the active backend.

    Args:
        args: Positional arguments for Rich print.
        kwargs: Keyword arguments for Rich print.

    Returns:
        None
    """
    print_fn = self._backend.print
    print_fn(*args)

  def get_style(self, name: str) -> Style:
    """Forward ``get_style`` calls to the active backend.

    Args:
        name (str): The name of the style to look up.

    Returns:
        Style: The resolved style object.
    """
    return self._backend.get_style(name)

  def export_text(self, *, clear: bool = True, styles: bool = False) -> str:
    """Forward ``export_text`` (useful for log capturing).

    Args:
        clear: Whether to clear the record buffer after exporting.
        styles: Whether to include ANSI style codes.

    Returns:
        str: The captured text output.
    """
    return self._backend.export_text(clear=clear, styles=styles)

  def export_html(self, *, inline_styles: bool = False, clear: bool = True) -> str:
    """Forward ``export_html`` (useful for web rendering).

    Args:
        inline_styles: Whether to render inline CSS styles.
        clear: Whether to clear the record buffer after exporting.

    Returns:
        str: The captured HTML output.
    """
    return self._backend.export_html(inline_styles=inline_styles, clear=clear)

  def export_svg(self, *, clear: bool = True) -> str:
    """Forward ``export_svg`` to the active backend.

    Args:
        clear: Whether to clear the record buffer after exporting.

    Returns:
        str: The captured SVG output.
    """
    return self._backend.export_svg(clear=clear)

  @property
  def width(self) -> int:
    """Return the width of the active backend console.

    Returns:
        int: Console width in characters.
    """
    return self._backend.width

  @property
  def is_terminal(self) -> bool:
    """Return whether the active backend console is a terminal.

    Returns:
        bool: True if terminal, False otherwise.
    """
    return self._backend.is_terminal


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
  """Retrieve the currently active console backend.

  Returns:
      Console: The active Rich Console.
  """
  return console.backend


def log_info(msg: str) -> None:
  """Log an informational message via standard logging.

  Args:
      msg (str): The message content. Can include rich markup like [bold].

  Returns:
      None
  """
  logging.info(f"\u2139\ufe0f  {msg}")


def log_success(msg: str) -> None:
  """Log a success message via standard logging.

  Args:
      msg (str): The message content.

  Returns:
      None
  """
  logging.log(SUCCESS_LEVEL_NUM, f"✅ {msg}")


def log_warning(msg: str) -> None:
  """Log a warning message via standard logging.

  Args:
      msg (str): The message content.

  Returns:
      None
  """
  logging.warning(f"⚠️  {msg}")


def log_error(msg: str) -> None:
  """Log an error message via standard logging.

  Args:
      msg (str): The message content.

  Returns:
      None
  """
  logging.error(f"❌ {msg}")
