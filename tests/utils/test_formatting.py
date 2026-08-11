"""Tests for formatting utilities."""

from ml_switcheroo.utils.formatting import StructuredFormatter, escape_html, escape_latex


def test_structured_formatter_basic():
  """Test standard adding lines."""
  fmt = StructuredFormatter(indent_size=4)
  fmt.add_line("def foo():", 0)
  fmt.add_line("return True", 1)
  fmt.add_line("", 0)

  result = fmt.build()
  assert result == "def foo():\n    return True\n"


def test_structured_formatter_add_block():
  """Test adding multiline blocks."""
  fmt = StructuredFormatter(indent_size=2)
  block = "line1\nline2\nline3"
  fmt.add_block(block, indent_level=2)

  result = fmt.build()
  assert result == "    line1\n    line2\n    line3"


def test_escape_html():
  """Test HTML escaping."""
  raw = "<div>&'\"</div>"
  escaped = escape_html(raw)
  assert escaped == "&lt;div&gt;&amp;&#x27;&quot;&lt;/div&gt;"

  raw2 = 123
  assert escape_html(raw2) == "123"


def test_escape_latex():
  """Test LaTeX escaping."""
  raw = "50% accuracy"
  assert escape_latex(raw) == "50\\% accuracy"

  raw2 = 123
  assert escape_latex(raw2) == "123"
