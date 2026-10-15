"""Tests for Latex nodes coverage."""

from ml_switcheroo.core.latex.nodes import DocumentNode, LatexNode, TextNode


def test_latex_node_abstract() -> None:
  """Verify latex node abstract behavior."""

  class DummyNode(LatexNode):
    """Test helper class DummyNode."""

    def to_latex(self) -> str:
      """To latex."""
      try:
        super().to_latex()
      except NotImplementedError:
        pass
      return "dummy"

  assert DummyNode().to_latex() == "dummy"


def test_text_node_emit() -> None:
  """Verify text node emit behavior."""
  node = TextNode(content="some text")
  assert node.emit(0) == "some text"
  assert node.emit(1) == "  some text"


def test_document_node_emit() -> None:
  """Verify document node emit behavior."""
  node = DocumentNode(children=[TextNode(content="inner")])
  # DocumentNode inherits LatexNode but its emit is custom to join children
  assert node.emit(1) == "  inner"
