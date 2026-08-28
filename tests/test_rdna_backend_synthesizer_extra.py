"""Test module."""

from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge
from unittest.mock import patch
from typing import Optional


class DummySemantics:
  """Test element."""

  def resolve_variant(self, abstract_id: str, flavor: str) -> Optional[dict]:
    """Test element."""
    return None

  def get_definition(self, kind: str) -> Optional[dict]:
    """Test element."""
    return None


def test_rdna_synthesizer_branches() -> None:
  """Test element."""
  # 1. Test when macros.json does not exist
  with patch("os.path.exists", return_value=False):
    synth: RdnaSynthesizer = RdnaSynthesizer(semantics=DummySemantics())
    assert not getattr(synth, "macro_registry")

  # 2. Test when macros.json exists but is empty
  with patch("os.path.exists", return_value=True):
    with patch("builtins.open", __import__("unittest").mock.mock_open(read_data="{}")):
      synth2: RdnaSynthesizer = RdnaSynthesizer(semantics=DummySemantics())
      assert not getattr(synth2, "macro_registry")

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
  graph: LogicalGraph = LogicalGraph(nodes=nodes, edges=edges)
  synth3: RdnaSynthesizer = RdnaSynthesizer(semantics=DummySemantics())
  synth3.from_graph(graph)

  # 4. Test when output has no sources (188->178)
  nodes2 = [LogicalNode(id="out2", kind="Output")]
  edges2 = []
  graph2: LogicalGraph = LogicalGraph(nodes=nodes2, edges=edges2)
  synth4: RdnaSynthesizer = RdnaSynthesizer(semantics=DummySemantics())
  synth4.from_graph(graph2)


def test_unmapped_op_and_comment() -> None:
  """Test element."""
  # Test Unmapped Op and RdnaComment branches
  nodes = [LogicalNode(id="unmapped1", kind="TotallyUnknownOp")]
  graph: LogicalGraph = LogicalGraph(nodes=nodes, edges=[])

  synth: RdnaSynthesizer = RdnaSynthesizer(semantics=DummySemantics())
  cst_mod: list = synth.from_graph(graph)
  assert cst_mod is not None
