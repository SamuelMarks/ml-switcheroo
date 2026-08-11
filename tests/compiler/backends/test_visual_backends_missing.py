"""Tests."""

from ml_switcheroo.core.compiler.backends.visual_backends import TikzBackend, LatexBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge


def test_calculate_layout_cycle():
  """Test function."""
  b = TikzBackend()
  g = LogicalGraph("T", nodes=[LogicalNode("n1", "Op"), LogicalNode("n2", "Op")])
  g.edges.extend([LogicalEdge("n1", "n2"), LogicalEdge("n2", "n1")])
  ranks = b._calculate_layout(g)
  assert ranks


def test_calculate_layout_disconnected_explicit():
  """Test function."""
  b = TikzBackend()
  g = LogicalGraph("T")
  g.nodes.append(LogicalNode("n1", "Input"))
  g.nodes.append(LogicalNode("n2", "Op"))
  g.edges.append(LogicalEdge("n1", "n2"))

  g.nodes.append(LogicalNode("c1", "Op"))
  g.nodes.append(LogicalNode("c2", "Op"))
  g.edges.append(LogicalEdge("c1", "c2"))
  g.edges.append(LogicalEdge("c2", "c1"))

  ranks = b._calculate_layout(g)
  assert "c1" in ranks


def test_latex_backend_formatting():
  """Test function."""
  b = LatexBackend()
  g = LogicalGraph("T")
  g.nodes.append(LogicalNode("n1", "Input"))
  g.nodes.append(LogicalNode("n2", "a.b.Add", metadata={"other": "v"}))
  g.nodes.append(LogicalNode("n3", "Output"))
  g.edges.append(LogicalEdge("n1", "n2"))
  g.edges.append(LogicalEdge("n2", "n3"))
  b.compile(g)


def test_latex_backend_duplicate_edge_and_noarg():
  """Test function."""
  b = LatexBackend()
  g = LogicalGraph("T")
  g.nodes.append(LogicalNode("n1", "Input"))
  g.nodes.append(LogicalNode("n2", "func_something", metadata={"notarg": "val"}))
  g.nodes.append(LogicalNode("n3", "math.add"))
  g.nodes.append(LogicalNode("n4", "Output"))
  g.edges.append(LogicalEdge("n1", "n3"))
  g.edges.append(LogicalEdge("n3", "n2"))
  g.edges.append(LogicalEdge("n2", "n4"))
  g.edges.append(LogicalEdge("n3", "n4"))
  g.edges.append(LogicalEdge("n3", "n4"))
  b.compile(g)


def test_force_transcode_lines():
  """Test function."""
  b = LatexBackend()
  g3 = LogicalGraph("T")
  g3.nodes.append(LogicalNode("n1", "Input"))
  g3.nodes.append(LogicalNode("func_n2", "foo.bar", metadata={"notkey": "val"}))
  g3.nodes.append(LogicalNode("n3", "Output"))
  g3.edges.append(LogicalEdge("n1", "func_n2"))
  g3.edges.append(LogicalEdge("func_n2", "n3"))

  g3.nodes.append(LogicalNode("func_n4", "Op"))
  g3.edges.append(LogicalEdge("func_n4", "func_n2"))
  g3.edges.append(LogicalEdge("func_n2", "func_n4"))
  b._transcode_graph(g3, "T")


def test_visual_backends_rank_existing_higher():
  # Hit 147->146
  """Test visual backends rank existing higher."""
  from ml_switcheroo.core.compiler.backends.visual_backends import TikzBackend
  from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge

  g = LogicalGraph("Test")
  g.nodes.append(LogicalNode("A", "Input"))
  g.nodes.append(LogicalNode("B", "Input"))
  g.nodes.append(LogicalNode("C", "Linear"))
  # A -> C, B -> C
  g.edges.append(LogicalEdge("A", "C"))
  g.edges.append(LogicalEdge("B", "C"))

  # If A is processed first, ranks[C] = 1
  # When B is processed, curr_rank = 0, ranks[C] = 1, so 1 < 0+1 is False!
  # This hits 147->146.
  class DummyVisual(TikzBackend):
    """Dummy visual."""

    def _get_shape(self, n):
      return "box"

  backend = DummyVisual()
  # just run _layout_graph
  backend._calculate_layout(g)
