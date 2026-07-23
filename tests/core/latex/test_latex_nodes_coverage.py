"""Tests for Latex nodes coverage."""

from ml_switcheroo.core.latex.nodes import LatexNode


def test_latex_node_abstract():
  """Test latex node abstract base class."""

  class DummyNode(LatexNode):
    """Dummy."""

    def to_latex(self):
      """To latex."""
      super().to_latex()
      return "dummy"

  assert DummyNode().to_latex() == "dummy"
