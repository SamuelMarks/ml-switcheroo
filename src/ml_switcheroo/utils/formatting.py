"""Structured Formatting Utilities.

This module provides classes and functions to programmatically build structured
output text, such as HTML or LaTeX, with appropriate indentation and character
escaping helper utilities.
"""

import html
from typing import List


class StructuredFormatter:
  """A utility class to build structured output (like HTML or LaTeX) programmatically.

  Attributes:
    lines: A list of formatted lines of text.
    indent_size: The number of spaces per indentation level.
  """

  def __init__(self, indent_size: int = 2):
    """Initializes the structured formatter with a custom indentation size.

    Args:
        indent_size: The number of spaces representing a single level of
        indentation. Defaults to 2.
    """
    self.lines: List[str] = []
    self.indent_size = indent_size

  def add_line(self, text: str, indent_level: int = 0) -> None:
    """Adds a line of text with the specified indentation level.

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
    """Adds a block of text, properly indenting each line within the block.

    Args:
        text: The multi-line block of text to add.
        indent_level: The level of indentation to apply to all lines in the block.
        Defaults to 0.
    """
    for line in text.splitlines():
      self.add_line(line, indent_level)

  def build(self) -> str:
    """Returns the constructed structured string.

    Returns:
        The complete constructed output with lines joined by newline characters.
    """
    return "\n".join(self.lines)


def escape_html(text: str) -> str:
  """Escapes HTML special characters in the given text safely.

  Args:
      text: The raw text containing potential HTML special characters.

  Returns:
      The HTML-escaped string.
  """
  return html.escape(str(text))


def escape_latex(text: str) -> str:
  r"""Escapes LaTeX content safely. Simple implementation.

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
  text = str(text)
  return text.replace("%", "\\%")
