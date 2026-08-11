"""Tests."""

from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer, RdnaBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment


def test_rdna_synthesizer_macro_exact_match():
  """Test function."""
  semantics = SemanticsManager()
  synth = RdnaSynthesizer(semantics)
  # mock a macro
  synth.macro_registry["my_abstract_id"] = lambda alloc, nid, meta: [RdnaComment(text="mock")]

  graph = LogicalGraph("test")
  n = LogicalNode("n1", "my_abstract_id")
  graph.nodes.append(n)

  # We also mock get_definition
  original = semantics.get_definition

  def mock_get_def(kind):
    """Test function."""
    if kind == "my_abstract_id":
      return ("my_abstract_id", {})
    return original(kind)

  semantics.get_definition = mock_get_def

  nodes = synth.from_graph(graph)
  assert len(nodes) == 1
  assert isinstance(nodes[0], RdnaComment)
  assert nodes[0].text == "mock"


def test_rdna_backend_compile():
  """Test function."""
  backend = RdnaBackend()
  graph = LogicalGraph("test")
  n = LogicalNode("n1", "Input")
  graph.nodes.append(n)
  code = backend.compile(graph)
  assert "; RDNA Code Generation Initialized" in code
