"""Tests."""

from ml_switcheroo.core.compiler.backends.sass.synthesizer import SassSynthesizer
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.compiler.frontends.sass.cst import SassComment


def test_sass_synthesizer_macro_exact_match():
  """Test function."""
  semantics = SemanticsManager()
  synth = SassSynthesizer(semantics)

  synth.macro_registry["my_macro"] = lambda alloc, nid, meta: [SassComment(text="mock_my_macro")]

  graph = LogicalGraph("test")
  n1 = LogicalNode("n1", "tensor")
  n2 = LogicalNode("n2", "my_macro")
  graph.nodes.extend([n1, n2])
  graph.edges.append(LogicalEdge("n1", "n2"))

  original = semantics.get_definition

  def mock_get_def(kind):
    """Test function."""
    if kind == "my_macro":
      return ("my_macro", {})
    return original(kind)

  semantics.get_definition = mock_get_def

  nodes = synth.from_graph(graph)

  found = False
  for n in nodes:
    if isinstance(n, SassComment) and n.text == "mock_my_macro":
      found = True
  assert found
