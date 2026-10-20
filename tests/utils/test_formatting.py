"""Tests for formatting utilities."""

from ml_switcheroo.utils.formatting import MarkupVisitor, StructuredFormatter, escape_html, escape_latex


def test_structured_formatter_basic() -> None:
  """Verify structured formatter basic behavior."""
  fmt: StructuredFormatter = StructuredFormatter(indent_size=4)
  fmt.add_line("def foo():", 0)
  fmt.add_line("return True", 1)
  fmt.add_line("", 0)

  result: str = fmt.build()
  assert result == "def foo():\n    return True\n"


def test_structured_formatter_add_block() -> None:
  """Verify structured formatter add block behavior."""
  fmt: StructuredFormatter = StructuredFormatter(indent_size=2)
  block: str = "line1\nline2\nline3"
  fmt.add_block(block, indent_level=2)

  result: str = fmt.build()
  assert result == "    line1\n    line2\n    line3"


def test_escape_html() -> None:
  """Verify escape html behavior."""
  raw: str = "<div>&'\"</div>"
  escaped: str = escape_html(raw)
  assert escaped == "&lt;div&gt;&amp;&#x27;&quot;&lt;/div&gt;"

  raw2: int = 123
  assert escape_html(raw2) == "123"


def test_escape_latex() -> None:
  """Verify escape latex behavior."""
  raw: str = "50% accuracy"
  assert escape_latex(raw) == "50\\% accuracy"

  raw2: int = 123
  assert escape_latex(raw2) == "123"


def test_markup_visitor() -> None:
  """Verify MarkupVisitor dispatches to generic_visit with and without children."""

  class Leaf:
    """Leaf node without children."""

  class Parent:
    """Parent node with children."""

    def __init__(self) -> None:
      """Initialize parent node."""
      self.children = [Leaf()]

  visited: list[Leaf] = []

  class CustomVisitor(MarkupVisitor):
    """Visitor recording visited Leaf nodes."""

    def visit_Leaf(self, node: Leaf) -> None:
      """Record visited Leaf node.

      Args:
          node: Leaf instance.
      """
      visited.append(node)

  visitor = CustomVisitor()
  parent = Parent()
  visitor.visit(parent)
  assert len(visited) == 1
  visitor.generic_visit(Leaf())
  assert len(visited) == 1
