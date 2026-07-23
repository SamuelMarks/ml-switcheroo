"""Test suite for the formatting utility module."""

from ml_switcheroo.utils.formatting import StructuredFormatter, escape_html, escape_latex


def test_structured_formatter_empty():
  """Test that StructuredFormatter handles empty lines."""
  fmt = StructuredFormatter()
  fmt.add_line("")
  fmt.add_line("   ")
  assert fmt.build() == "\n   "


def test_structured_formatter_indent():
  """Test that StructuredFormatter handles indentation correctly."""
  fmt = StructuredFormatter(indent_size=4)
  fmt.add_line("hello", 1)
  fmt.add_block("world\n!", 2)
  assert fmt.build() == "    hello\n        world\n        !"


def test_escape_html():
  """Test HTML escaping functionality."""
  assert escape_html("<script>") == "&lt;script&gt;"


def test_escape_latex():
  """Test LaTeX escaping functionality."""
  assert escape_latex("100% pure") == r"100\% pure"
