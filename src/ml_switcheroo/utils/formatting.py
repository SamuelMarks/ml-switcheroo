"""Structured Formatting Utilities."""

import html
from typing import List


class StructuredFormatter:
  """A utility class to build structured output (like HTML or LaTeX) programmatically."""

  def __init__(self, indent_size: int = 2):
    """Initializes the formatter."""
    self.lines: List[str] = []
    self.indent_size = indent_size

  def add_line(self, text: str, indent_level: int = 0) -> None:
    """Adds a line with the given indentation."""
    if not text.strip() and not text:
      self.lines.append("")
    else:
      indent = " " * (indent_level * self.indent_size)
      self.lines.append(f"{indent}{text}")

  def add_block(self, text: str, indent_level: int = 0) -> None:
    """Adds a block of text, properly indenting each line."""
    for line in text.splitlines():
      self.add_line(line, indent_level)

  def build(self) -> str:
    """Returns the constructed string."""
    return "\n".join(self.lines)


def escape_html(text: str) -> str:
  """Escapes HTML content safely."""
  return html.escape(str(text))


def escape_latex(text: str) -> str:
  """Escapes LaTeX content safely. Simple implementation."""
  # In actual LaTeX escaping, there are many characters (e.g. % $ # _ { } ~ ^ \).
  # We will just do a basic one or pass through if it's considered safe in our DSL.
  # Let's escape only a few core ones if needed. For now, replace % with \%.
  text = str(text)
  return text.replace("%", "\\%")
