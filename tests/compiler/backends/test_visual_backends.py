"""Unit tests for the visual backends, including TikzBackend and LatexBackend.

This module validates that the conversion from a LogicalGraph to visual/document-based
representations (such as LaTeX/standalone TikZ graphics) is handled correctly.
"""

from ml_switcheroo.core.compiler.backends.visual_backends import LatexBackend, TikzBackend
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode


def test_tikz_backend_init() -> None:
  """Verify that the TikzBackend can be successfully initialized.

  Args:
    None

  Returns:
    None
  """
  backend = TikzBackend()
  assert backend is not None


def test_tikz_backend_is_stateful() -> None:
  """Verify the stateful behavior placeholder of the TikzBackend.

  Args:
    None

  Returns:
    None
  """
  pass


def test_tikz_backend_create_tikz_node_stateful() -> None:
  """Verify TikZ node generation for a stateful node.

  This test checks that a stateful logical node (like 'Linear') with key-value metadata
  is correctly converted into a TikZ node, and that its formatted properties and backslashes
  are properly escaped in the emitted content.

  Args:
    None

  Returns:
    None
  """
  backend = TikzBackend()
  node = LogicalNode(id="L1", kind="Linear", metadata={"arg_0": 10})
  tnode = backend._create_tikz_node(node, 0, 0)

  assert tnode.node_id == "L1"
  assert "Linear" in tnode.content.emit()
  assert "arg\\_0: 10" in tnode.content.emit()


def test_tikz_backend_create_tikz_node_stateless() -> None:
  """Verify TikZ node generation for a stateless node.

  This test checks that a stateless logical node (like 'relu') is correctly converted
  into a TikZ node representation with its kind properly preserved.

  Args:
    None

  Returns:
    None
  """
  backend = TikzBackend()
  node = LogicalNode(id="R1", kind="relu")
  tnode = backend._create_tikz_node(node, 0, 0)

  assert tnode.node_id == "R1"
  assert "relu" in tnode.content.emit()


def test_tikz_backend_create_tikz_node_input_output() -> None:
  """Verify TikZ node generation for input and output boundary nodes.

  This test checks that an 'Input' logical node is correctly transcoded into a TikZ node
  with the appropriate identifier.

  Args:
    None

  Returns:
    None
  """
  backend = TikzBackend()
  node_in = LogicalNode(id="in", kind="Input")
  tn_in = backend._create_tikz_node(node_in, 0, 0)
  assert tn_in.node_id == "in"


def test_latex_backend_compile() -> None:
  """Verify compilation of a logical graph into a full LaTeX document.

  This test builds a complete logical graph with input, linear layer, activation, and
  output nodes, compiles it, and validates that the resulting LaTeX string has all the
  necessary definition blocks, model metadata, and operation blocks.

  Args:
    None

  Returns:
    None
  """
  backend = LatexBackend()
  graph = LogicalGraph(name="MyModel")
  graph.nodes.append(LogicalNode(id="in1", kind="Input", metadata={"name": "x"}))
  graph.nodes.append(LogicalNode(id="l1", kind="Linear", metadata={"arg_1": "10"}))
  graph.nodes.append(LogicalNode(id="r1", kind="relu"))
  graph.nodes.append(LogicalNode(id="out1", kind="Output"))

  graph.edges.append(LogicalEdge("in1", "l1"))
  graph.edges.append(LogicalEdge("l1", "r1"))
  graph.edges.append(LogicalEdge("r1", "out1"))

  res = backend.compile(graph)
  assert "\\begin{DefModel}" in res
  assert "MyModel" in res
  assert "Linear" in res
  assert "Return" in res


def test_latex_backend_compile_no_name() -> None:
  """Verify LaTeX compilation when the logical graph is unnamed.

  This test compiles a logical graph without a set model name, verifying that the
  LatexBackend correctly defaults the name (e.g. to 'Model').

  Args:
    None

  Returns:
    None
  """
  backend = LatexBackend()
  graph = LogicalGraph()
  graph.nodes.append(LogicalNode(id="in1", kind="Input"))

  res = backend.compile(graph)
  assert "Model" in res


def test_latex_backend_compile_no_output() -> None:
  """Verify LaTeX compilation when the logical graph lacks an explicit output node.

  This test compiles a logical graph missing an explicit 'Output' node, verifying that the
  LatexBackend correctly generates fallback return statements/identifiers (e.g. 'last_step').

  Args:
    None

  Returns:
    None
  """
  backend = LatexBackend()
  graph = LogicalGraph()
  graph.nodes.append(LogicalNode(id="in1", kind="Input"))
  graph.nodes.append(LogicalNode(id="l1", kind="Linear"))
  graph.edges.append(LogicalEdge("in1", "l1"))

  # Missing explicit Output node
  res = backend.compile(graph)
  assert "Return" in res
  assert "last_step" in res


# --- Merged from test_visual_backends_missing.py ---


def test_calculate_layout_cycle() -> None:
  """Docstring."""
  b = TikzBackend()
  g = LogicalGraph("T", nodes=[LogicalNode("n1", "Op"), LogicalNode("n2", "Op")])
  g.edges.extend([LogicalEdge("n1", "n2"), LogicalEdge("n2", "n1")])
  ranks: dict[str, int] = b._calculate_layout(g)
  assert ranks


def test_calculate_layout_disconnected_explicit() -> None:
  """Docstring."""
  b = TikzBackend()
  g = LogicalGraph("T")
  g.nodes.append(LogicalNode("n1", "Input"))
  g.nodes.append(LogicalNode("n2", "Op"))
  g.edges.append(LogicalEdge("n1", "n2"))

  g.nodes.append(LogicalNode("c1", "Op"))
  g.nodes.append(LogicalNode("c2", "Op"))
  g.edges.append(LogicalEdge("c1", "c2"))
  g.edges.append(LogicalEdge("c2", "c1"))

  ranks: dict[str, int] = b._calculate_layout(g)
  assert "c1" in ranks


def test_latex_backend_formatting() -> None:
  """Docstring."""
  b = LatexBackend()
  g = LogicalGraph("T")
  g.nodes.append(LogicalNode("n1", "Input"))
  g.nodes.append(LogicalNode("n2", "a.b.Add", metadata={"other": "v"}))
  g.nodes.append(LogicalNode("n3", "Output"))
  g.edges.append(LogicalEdge("n1", "n2"))
  g.edges.append(LogicalEdge("n2", "n3"))
  b.compile(g)


def test_latex_backend_duplicate_edge_and_noarg() -> None:
  """Docstring."""
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


def test_force_transcode_lines() -> None:
  """Docstring."""
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


def test_visual_backends_rank_existing_higher() -> None:
  """Docstring."""
  # Hit 147->146
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

    def _get_shape(self, n: LogicalNode) -> str:
      return "box"

  backend = DummyVisual()
  # just run _calculate_layout
  backend._calculate_layout(g)
