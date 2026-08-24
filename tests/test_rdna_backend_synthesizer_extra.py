"""Test module."""

from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge
from unittest.mock import patch


class DummySemantics:
  """Test element."""

  def resolve_variant(self, abstract_id, flavor):
    """Test element."""
    return None

  def get_definition(self, kind: str):
    """Test element."""
    return None


def test_rdna_synthesizer_branches():
  """Test element."""
  # 1. Test when macros.json does not exist
  with patch("os.path.exists", return_value=False):
    synth = RdnaSynthesizer(semantics=DummySemantics())
    assert not synth.macro_registry

  # 2. Test when macros.json exists but is empty
  with patch("os.path.exists", return_value=True):
    with patch("builtins.open", __import__("unittest").mock.mock_open(read_data="{}")):
      synth = RdnaSynthesizer(semantics=DummySemantics())
      assert not synth.macro_registry

  # 3. Test multiple edges to same target (174->176)
  nodes = [
    LogicalNode(id="in1", kind="Input"),
    LogicalNode(id="in2", kind="Input"),
    LogicalNode(id="add", kind="Add"),
    LogicalNode(id="out", kind="Output"),  # Empty output sources?
  ]
  edges = [
    LogicalEdge(source="in1", target="add"),
    LogicalEdge(source="in2", target="add"),  # multiple edges
  ]
  graph = LogicalGraph(nodes=nodes, edges=edges)
  synth = RdnaSynthesizer(semantics=DummySemantics())
  synth.from_graph(graph)

  # 4. Test when output has no sources (188->178)
  nodes = [LogicalNode(id="out2", kind="Output")]
  edges = []
  graph = LogicalGraph(nodes=nodes, edges=edges)
  synth = RdnaSynthesizer(semantics=DummySemantics())
  synth.from_graph(graph)


def test_unmapped_op_and_comment():
  """Test element."""
  # Test Unmapped Op and RdnaComment branches
  nodes = [LogicalNode(id="unmapped1", kind="TotallyUnknownOp")]
  graph = LogicalGraph(nodes=nodes, edges=[])

  synth = RdnaSynthesizer(semantics=DummySemantics())
  cst_mod = synth.from_graph(graph)
  assert cst_mod is not None
