"""Tests for TikZ nodes coverage."""

from ml_switcheroo.core.tikz.nodes import TikzBaseNode, TikzGraph, TikzNode, TriviaNode


def test_tikz_base_node_abstract():
  """Test tikz base node abstract."""

  class Dummy(TikzBaseNode):
    """Dummy."""

    def to_text(self):
      """To text."""
      super().to_text()
      return ""

  Dummy().to_text()


def test_nodenode_with_trivia():
  """Test nodenode with trivia."""
  triv = TriviaNode("% comment")
  node = TikzNode(node_id="A", x=0.0, y=0.0, leading_trivia=[triv], options=[], content="")
  res = node.to_text()
  assert "% comment" in res


def test_tikzpicturenode_no_options():
  """Test tikzpicturenode no options."""
  pic = TikzGraph(options=[], children=[])
  res = pic.to_text()
  assert "\\begin{tikzpicture}" in res
