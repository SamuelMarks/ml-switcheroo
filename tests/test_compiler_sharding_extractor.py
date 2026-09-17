"""Test suite for the compiler sharding extractor pass."""

from typing import Any, Dict, List, Optional, Set, Tuple

from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode, PartitionSpec
from ml_switcheroo.core.compiler.sharding_extractor import ShardingExtractionPass


def test_sharding_extraction_pass_no_nodes() -> None:
  """Docstring."""
  graph: LogicalGraph = LogicalGraph(nodes={"n1": LogicalNode(id="n1", op_type="Linear")}, edges=[])
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert len(new_graph.nodes) == 1
  assert "n1" in new_graph.nodes
  assert new_graph.nodes["n1"].id == "n1"
  assert new_graph.nodes["n1"].sharding is None


def test_sharding_extraction_pass_missing_source() -> None:
  """Docstring."""
  nodes: List[LogicalNode] = [
    LogicalNode(id="n1", op_type="Linear"),
    LogicalNode(id="s1", op_type="with_sharding_constraint", attributes={"arg_1": "PartitionSpec('data', None)"}),
  ]
  graph: LogicalGraph = LogicalGraph(nodes={n.id: n for n in nodes}, edges=[])
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert len(new_graph.nodes) == 2  # The node should not be removed


def test_sharding_extraction_pass_missing_source_node() -> None:
  """Docstring."""
  nodes: List[LogicalNode] = [
    LogicalNode(id="s1", op_type="with_sharding_constraint", attributes={"arg_1": "PartitionSpec('data', None)"})
  ]
  edges: List[LogicalEdge] = [
    LogicalEdge(source="n1", target="s1")  # n1 not in nodes
  ]
  graph: LogicalGraph = LogicalGraph(nodes={n.id: n for n in nodes}, edges=edges)
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert len(new_graph.nodes) == 1


def test_sharding_extraction_pass_invalid_parse() -> None:
  """Docstring."""
  nodes: List[LogicalNode] = [
    LogicalNode(id="n1", op_type="Linear"),
    LogicalNode(id="s1", op_type="with_sharding_constraint", attributes={"arg_1": "PartitionSpec(!"}),
  ]
  edges: List[LogicalEdge] = [LogicalEdge(source="n1", target="s1")]
  graph: LogicalGraph = LogicalGraph(nodes={n.id: n for n in nodes}, edges=edges)
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert len(new_graph.nodes) == 2


def test_sharding_extraction_pass_success() -> None:
  """Docstring."""
  nodes: List[LogicalNode] = [
    LogicalNode(id="n1", op_type="Linear"),
    LogicalNode(id="s1", op_type="with_sharding_constraint", attributes={"arg_1": "PartitionSpec('data', None)"}),
    LogicalNode(id="n2", op_type="Output"),
  ]
  edges: List[LogicalEdge] = [
    LogicalEdge(source="n1", target="s1"),
    LogicalEdge(source="s1", target="n2"),
    # test that duplicate edge is not added
    LogicalEdge(source="n1", target="n2"),  # already exists, tests the `if new_edge not in new_edges` block
  ]
  graph: LogicalGraph = LogicalGraph(nodes={n.id: n for n in nodes}, edges=edges)
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)

  assert len(new_graph.nodes) == 2
  node_ids: Set[str] = set(new_graph.nodes.keys())
  assert "s1" not in node_ids

  n1_node: LogicalNode = new_graph.nodes["n1"]
  assert n1_node.sharding is not None
  assert n1_node.sharding.axes == ("data", None)

  edge_sources: Set[Tuple[str, str]] = {(e.source, e.target) for e in new_graph.edges}
  assert ("n1", "n2") in edge_sources


def test_parse_partition_spec_various() -> None:
  """Docstring."""
  pass_: ShardingExtractionPass = ShardingExtractionPass()

  assert pass_._parse_partition_spec("foo()") is None
  res1: Optional[PartitionSpec] = pass_._parse_partition_spec("PartitionSpec('tensor')")
  assert res1 is not None and res1.axes == ("tensor",)
  res2: Optional[PartitionSpec] = pass_._parse_partition_spec("NamedSharding('tensor')")
  assert res2 is not None and res2.axes == ("tensor",)

  # Test tuple support
  res3: Optional[PartitionSpec] = pass_._parse_partition_spec("PartitionSpec(('data', 'tensor'), None)")
  assert res3 is not None and res3.axes == (("data", "tensor"), None)

  # Test unknown arg type
  res4: Optional[PartitionSpec] = pass_._parse_partition_spec("PartitionSpec(some_var)")
  assert res4 is not None and res4.axes == (None,)


def test_sharding_extraction_pass_preserve_other_edges() -> None:
  """Docstring."""
  nodes: List[LogicalNode] = [
    LogicalNode(id="n1", op_type="Linear"),
    LogicalNode(id="s1", op_type="with_sharding_constraint", attributes={"arg_1": "PartitionSpec('data', None)"}),
    LogicalNode(id="n2", op_type="Output"),
    LogicalNode(id="other1", op_type="Other"),
    LogicalNode(id="other2", op_type="Other"),
  ]
  edges: List[LogicalEdge] = [
    LogicalEdge(source="n1", target="s1"),
    LogicalEdge(source="s1", target="n2"),
    LogicalEdge(source="other1", target="other2"),  # preserved
  ]
  graph: LogicalGraph = LogicalGraph(nodes={n.id: n for n in nodes}, edges=edges)
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  edge_sources: Set[Tuple[str, str]] = {(e.source, e.target) for e in new_graph.edges}
  assert ("other1", "other2") in edge_sources


def test_sharding_extraction_pass_metadata_fallback() -> None:
  """Verifies sharding extraction falls back to metadata when attributes is missing."""

  class LegacyNode:
    """Mock node mimicking legacy node structures without an attributes field."""

    def __init__(self, node_id: str, op_type: str, metadata: Dict[str, str]) -> None:
      """Initializes a LegacyNode instance.

      Args:
          node_id: Unique identifier of the node.
          op_type: Type of operation.
          metadata: Dictionary of legacy metadata.
      """
      self.id = node_id
      self.op_type = op_type
      self.metadata = metadata
      self.inputs: List[str] = []
      self.outputs: List[str] = []
      self.sharding: Optional[PartitionSpec] = None

  n1 = LogicalNode(id="n1", op_type="Linear")
  s1 = LegacyNode(
    node_id="s1",
    op_type="with_sharding_constraint",
    metadata={"arg_1": "PartitionSpec('data', None)"},
  )
  nodes: Dict[str, Any] = {"n1": n1, "s1": s1}
  edges: List[LogicalEdge] = [LogicalEdge(source="n1", target="s1")]
  graph: LogicalGraph = LogicalGraph(nodes=nodes, edges=edges)
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert new_graph.nodes["n1"].sharding is not None
  assert new_graph.nodes["n1"].sharding.axes == ("data", None)
