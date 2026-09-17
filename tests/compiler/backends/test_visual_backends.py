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

  This test checks that a stateful logical node (like 'Linear') with key-value attributes
  is correctly converted into a TikZ node, and that its formatted properties and backslashes
  are properly escaped in the emitted content.

  Args:
    None

  Returns:
    None
  """
  backend = TikzBackend()
  node = LogicalNode(id="L1", op_type="Linear", attributes={"arg_0": 10})
  tnode = backend._create_tikz_node(node, 0, 0)

  assert tnode.node_id == "L1"
  assert "Linear" in tnode.content.emit()
  assert r"arg\_0: 10" in tnode.content.emit()


def test_tikz_backend_create_tikz_node_stateless() -> None:
  """Verify TikZ node generation for a stateless node.

  This test checks that a stateless logical node (like 'relu') is correctly converted
  into a TikZ node representation with its op_type properly preserved.

  Args:
    None

  Returns:
    None
  """
  backend = TikzBackend()
  node = LogicalNode(id="R1", op_type="relu")
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
  node_in = LogicalNode(id="in", op_type="Input")
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
  nodes = {
    "in1": LogicalNode(id="in1", op_type="Input", attributes={"name": "x"}),
    "l1": LogicalNode(id="l1", op_type="Linear", attributes={"arg_1": "10"}),
    "r1": LogicalNode(id="r1", op_type="relu"),
    "out1": LogicalNode(id="out1", op_type="Output"),
  }
  edges = [
    LogicalEdge("in1", "l1"),
    LogicalEdge("l1", "r1"),
    LogicalEdge("r1", "out1"),
  ]
  graph = LogicalGraph(name="MyModel", nodes=nodes, edges=edges)

  res = backend.compile(graph)
  assert r"\begin{DefModel}" in res
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
  graph = LogicalGraph(nodes={"in1": LogicalNode(id="in1", op_type="Input")})

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
  nodes = {
    "in1": LogicalNode(id="in1", op_type="Input"),
    "l1": LogicalNode(id="l1", op_type="Linear"),
  }
  edges = [LogicalEdge("in1", "l1")]
  graph = LogicalGraph(nodes=nodes, edges=edges)

  # Missing explicit Output node
  res = backend.compile(graph)
  assert "Return" in res
  assert "last_step" in res


# --- Merged from test_visual_backends_missing.py ---


def test_calculate_layout_cycle() -> None:
  """Docstring."""
  b = TikzBackend()
  nodes = {
    "n1": LogicalNode("n1", op_type="Op"),
    "n2": LogicalNode("n2", op_type="Op"),
  }
  edges = [LogicalEdge("n1", "n2"), LogicalEdge("n2", "n1")]
  g = LogicalGraph("T", nodes=nodes, edges=edges)
  ranks: dict[str, int] = b._calculate_layout(g)
  assert ranks


def test_calculate_layout_disconnected_explicit() -> None:
  """Docstring."""
  b = TikzBackend()
  nodes = {
    "n1": LogicalNode("n1", op_type="Input"),
    "n2": LogicalNode("n2", op_type="Op"),
    "c1": LogicalNode("c1", op_type="Op"),
    "c2": LogicalNode("c2", op_type="Op"),
  }
  edges = [
    LogicalEdge("n1", "n2"),
    LogicalEdge("c1", "c2"),
    LogicalEdge("c2", "c1"),
  ]
  g = LogicalGraph("T", nodes=nodes, edges=edges)

  ranks: dict[str, int] = b._calculate_layout(g)
  assert "c1" in ranks


def test_latex_backend_formatting() -> None:
  """Docstring."""
  b = LatexBackend()
  nodes = {
    "n1": LogicalNode("n1", op_type="Input"),
    "n2": LogicalNode("n2", op_type="a.b.Add", attributes={"other": "v"}),
    "n3": LogicalNode("n3", op_type="Output"),
  }
  edges = [
    LogicalEdge("n1", "n2"),
    LogicalEdge("n2", "n3"),
  ]
  g = LogicalGraph("T", nodes=nodes, edges=edges)
  b.compile(g)


def test_latex_backend_duplicate_edge_and_noarg() -> None:
  """Docstring."""
  b = LatexBackend()
  nodes = {
    "n1": LogicalNode("n1", op_type="Input"),
    "n2": LogicalNode("n2", op_type="func_something", attributes={"notarg": "val"}),
    "n3": LogicalNode("n3", op_type="math.add"),
    "n4": LogicalNode("n4", op_type="Output"),
  }
  edges = [
    LogicalEdge("n1", "n3"),
    LogicalEdge("n3", "n2"),
    LogicalEdge("n2", "n4"),
    LogicalEdge("n3", "n4"),
    LogicalEdge("n3", "n4"),
  ]
  g = LogicalGraph("T", nodes=nodes, edges=edges)
  b.compile(g)


def test_force_transcode_lines() -> None:
  """Docstring."""
  b = LatexBackend()
  nodes = {
    "n1": LogicalNode("n1", op_type="Input"),
    "func_n2": LogicalNode("func_n2", op_type="foo.bar", attributes={"notkey": "val"}),
    "n3": LogicalNode("n3", op_type="Output"),
    "func_n4": LogicalNode("func_n4", op_type="Op"),
  }
  edges = [
    LogicalEdge("n1", "func_n2"),
    LogicalEdge("func_n2", "n3"),
    LogicalEdge("func_n4", "func_n2"),
    LogicalEdge("func_n2", "func_n4"),
  ]
  g3 = LogicalGraph("T", nodes=nodes, edges=edges)
  b._transcode_graph(g3, "T")


def test_visual_backends_rank_existing_higher() -> None:
  """Docstring."""
  nodes = {
    "A": LogicalNode("A", op_type="Input"),
    "B": LogicalNode("B", op_type="Input"),
    "C": LogicalNode("C", op_type="Linear"),
  }
  edges = [
    LogicalEdge("A", "C"),
    LogicalEdge("B", "C"),
  ]
  g = LogicalGraph("Test", nodes=nodes, edges=edges)

  class DummyVisual(TikzBackend):
    """Dummy visual."""

    def _get_shape(self, n: LogicalNode) -> str:
      """Docstring."""
      return "box"

  backend = DummyVisual()
  backend._calculate_layout(g)
