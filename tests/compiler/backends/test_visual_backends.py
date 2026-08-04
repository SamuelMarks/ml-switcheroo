"""Unit tests for the visual backends, including TikzBackend and LatexBackend.

This module validates that the conversion from a LogicalGraph to visual/document-based
representations (such as LaTeX/standalone TikZ graphics) is handled correctly.
"""

from ml_switcheroo.core.compiler.backends.visual_backends import TikzBackend, LatexBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge


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
