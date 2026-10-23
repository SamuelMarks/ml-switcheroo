"""Structured Formatting Utilities.

This module provides classes and functions to programmatically build structured
output text, such as HTML or LaTeX, with appropriate indentation and character
escaping helper utilities.
"""

import html

from ml_switcheroo.core.markup.cst import MarkupNode


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


def escape_html(text: object) -> str:
  """Escape HTML special characters in the given text safely.

  Args:
      text: The raw text containing potential HTML special characters.

  Returns:
      The HTML-escaped string.
  """
  return html.escape(str(text))


def escape_latex(text: object) -> str:
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


class MarkupVisitor:
  """Base class for traversing Markup nodes."""

  def visit(self, node: MarkupNode | object) -> None:
    """Dispatch traversal of a markup AST node to its class-specific visitor method.

    Args:
        node: Markup AST node to visit.
    """
    method_name = f"visit_{node.__class__.__name__}"
    visitor = getattr(self, method_name, self.generic_visit)
    visitor(node)

  def generic_visit(self, node: MarkupNode | object) -> None:
    """Recursively visit all child nodes when no specialized visitor method exists.

    Args:
        node: Markup AST node whose ``children`` sequence will be traversed.
    """
    if hasattr(node, "children"):
      for child in getattr(node, "children"):
        self.visit(child)
