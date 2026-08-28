"""Tests for Latex nodes coverage."""

from ml_switcheroo.core.latex.nodes import LatexNode, TextNode, DocumentNode


def test_latex_node_abstract() -> None:
  """Test latex node abstract base class."""

  class DummyNode(LatexNode):
    """Dummy."""

    def to_latex(self) -> str:
      """To latex."""
      try:
        super().to_latex()
      except NotImplementedError:
        pass
      return "dummy"

  assert DummyNode().to_latex() == "dummy"


def test_text_node_emit() -> None:
  """Test TextNode emit."""
  node = TextNode(content="some text")
  assert node.emit(0) == "some text"
  assert node.emit(1) == "  some text"


def test_document_node_emit() -> None:
  """Test DocumentNode emit."""
  node = DocumentNode(children=[TextNode(content="inner")])
  # DocumentNode inherits LatexNode but its emit is custom to join children
  assert node.emit(1) == "  inner"
