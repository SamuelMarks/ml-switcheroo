"""Structured Formatting Utilities.

This module provides classes and functions to programmatically build structured
output text, such as HTML or LaTeX, with appropriate indentation and character
escaping helper utilities.
"""

import html
import typing

from pydantic import JsonValue

from ml_switcheroo.core.markup.cst import MarkupNode
from ml_switcheroo.enums import _CustomInstanceProtocol
from ml_switcheroo.exceptions import CSTValidationError


@typing.runtime_checkable
class _MarkupChildrenProtocol(typing.Protocol):
  """Protocol for markup nodes exposing `children`."""

  @property
  def children(self) -> typing.Sequence["MarkupVisitable"]:
    """Return nested child nodes.

    Returns:
        Sequence of child markup items.
    """


MarkupVisitable: typing.TypeAlias = MarkupNode | _MarkupChildrenProtocol | _CustomInstanceProtocol | JsonValue


class StructuredFormatter:
  """A utility class to build structured output (like HTML or LaTeX) programmatically.

  Attributes:
    lines: A list of formatted lines of text.
    indent_size: The number of spaces per indentation level.
  """

  def __init__(self, indent_size: int = 2):
    """Initialize the structured formatter with a custom indentation size.

    Args:
        indent_size: The number of spaces representing a single level of
        indentation. Defaults to 2.
    """
    self.lines: list[str] = []
    self.indent_size = indent_size

  def add_line(self, text: str, indent_level: int = 0) -> None:
    """Add a line of text with the specified indentation level.

    Args:
        text: The text content of the line to add.
        indent_level: The level of indentation to apply to the line. Defaults to 0.
    """
    if not text.strip() and not text:
      self.lines.append("")
    else:
      indent = " " * (indent_level * self.indent_size)
      self.lines.append(f"{indent}{text}")

  def add_block(self, text: str, indent_level: int = 0) -> None:
    """Add a block of text, properly indenting each line within the block.

    Args:
        text: The multi-line block of text to add.
        indent_level: The level of indentation to apply to all lines in the block.
        Defaults to 0.
    """
    for line in text.splitlines():
      self.add_line(line, indent_level)

  def build(self) -> str:
    """Return the constructed structured string.

    Returns:
        The complete constructed output with lines joined by newline characters.
    """
    return "\n".join(self.lines)


def escape_html(text: str | int | float | bool | None) -> str:
  """Escape HTML special characters in the given text safely.

  Args:
      text: The raw text containing potential HTML special characters.

  Returns:
      The HTML-escaped string.
  """
  return html.escape(str(text))


def escape_latex(text: str | int | float | bool | None) -> str:
  r"""Escape LaTeX content safely. Simple implementation.

  In actual LaTeX escaping, there are many characters (e.g. % $ # _ { } ~ ^ \\).
  We will just do a basic one or pass through if it's considered safe in our DSL.
  Let's escape only a few core ones if needed. For now, replace % with \\%.

  Args:
      text: The raw text containing potential LaTeX special characters.

  Returns:
      The LaTeX-escaped string.
  """
  # In actual LaTeX escaping, there are many characters (e.g. % $ # _ { } ~ ^ \).
  # We will just do a basic one or pass through if it's considered safe in our DSL.
  # Let's escape only a few core ones if needed. For now, replace % with \%.
  text_str = str(text)
  return text_str.replace("%", "\\%")


MarkupNodeT = typing.TypeVar("MarkupNodeT")


class MarkupVisitor:
  """Base class for traversing Markup nodes via explicit type-keyed handler registration."""

  def __init__(self) -> None:
    """Initialize the visitor handler registry."""
    self._visitors: dict[type, typing.Callable[[MarkupVisitable], None]] = {}

  def register_visitor(
    self,
    node_type: type[MarkupNodeT],
    handler: typing.Callable[[MarkupNodeT], None],
  ) -> None:
    """Register a typed visitor callback for ``node_type``.

    Args:
        node_type: Concrete markup node class to match.
        handler: Callback invoked when visiting an instance of ``node_type``.
    """

    def _wrapped(node: MarkupVisitable) -> None:
      """Narrow ``node`` to ``node_type`` and invoke ``handler``.

      Args:
          node: Visited markup node matching ``node_type``.

      Raises:
          CSTValidationError: If ``node`` is not an instance of ``node_type``.
      """
      if not isinstance(node, node_type):
        raise CSTValidationError(f"Expected {node_type.__name__}, got {type(node).__name__}.")
      handler(node)

    self._visitors[node_type] = _wrapped

  def visit(self, node: MarkupVisitable) -> None:
    """Dispatch traversal of a markup AST node to its registered visitor method or ``generic_visit``.

    Args:
        node: Markup AST node to visit.
    """
    handler = self._visitors.get(type(node))
    if handler is None:
      self.generic_visit(node)
      return
    handler(node)

  def generic_visit(self, node: MarkupVisitable) -> None:
    """Recursively visit all child nodes when no specialized visitor method exists.

    Args:
        node: Markup AST node whose ``children`` sequence will be traversed.
    """
    if isinstance(node, _MarkupChildrenProtocol):
      for child in node.children:
        self.visit(child)
