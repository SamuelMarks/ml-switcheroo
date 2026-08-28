"""Test suite for the compiler sharding extractor pass."""

from typing import List, Set, Tuple, Optional
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge, PartitionSpec
from ml_switcheroo.core.compiler.sharding_extractor import ShardingExtractionPass


def test_sharding_extraction_pass_no_nodes() -> None:
  """Test with no sharding constraint nodes."""
  graph: LogicalGraph = LogicalGraph(nodes=[LogicalNode(id="n1", kind="Linear")], edges=[])
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert len(new_graph.nodes) == 1
  assert new_graph.nodes[0].id == "n1"
  assert new_graph.nodes[0].sharding is None


def test_sharding_extraction_pass_missing_source() -> None:
  """Test with sharding constraint node that has no incoming edge."""
  nodes: List[LogicalNode] = [
    LogicalNode(id="n1", kind="Linear"),
    LogicalNode(id="s1", kind="with_sharding_constraint", metadata={"arg_1": "PartitionSpec('data', None)"}),
  ]
  graph: LogicalGraph = LogicalGraph(nodes=nodes, edges=[])
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert len(new_graph.nodes) == 2  # The node should not be removed


def test_sharding_extraction_pass_missing_source_node() -> None:
  """Test with sharding constraint node but the source node does not exist in graph.nodes."""
  nodes: List[LogicalNode] = [
    LogicalNode(id="s1", kind="with_sharding_constraint", metadata={"arg_1": "PartitionSpec('data', None)"})
  ]
  edges: List[LogicalEdge] = [
    LogicalEdge(source="n1", target="s1")  # n1 not in nodes
  ]
  graph: LogicalGraph = LogicalGraph(nodes=nodes, edges=edges)
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert len(new_graph.nodes) == 1


def test_sharding_extraction_pass_invalid_parse() -> None:
  """Test with sharding constraint that fails to parse."""
  nodes: List[LogicalNode] = [
    LogicalNode(id="n1", kind="Linear"),
    LogicalNode(id="s1", kind="with_sharding_constraint", metadata={"arg_1": "PartitionSpec(!"}),
  ]
  edges: List[LogicalEdge] = [LogicalEdge(source="n1", target="s1")]
  graph: LogicalGraph = LogicalGraph(nodes=nodes, edges=edges)
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert len(new_graph.nodes) == 2


def test_sharding_extraction_pass_success() -> None:
  """Test successfully extracting sharding constraints and rewiring edges."""
  nodes: List[LogicalNode] = [
    LogicalNode(id="n1", kind="Linear"),
    LogicalNode(id="s1", kind="with_sharding_constraint", metadata={"arg_1": "PartitionSpec('data', None)"}),
    LogicalNode(id="n2", kind="Output"),
  ]
  edges: List[LogicalEdge] = [
    LogicalEdge(source="n1", target="s1"),
    LogicalEdge(source="s1", target="n2"),
    # test that duplicate edge is not added
    LogicalEdge(source="n1", target="n2"),  # already exists, tests the `if new_edge not in new_edges` block
  ]
  graph: LogicalGraph = LogicalGraph(nodes=nodes, edges=edges)
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)

  assert len(new_graph.nodes) == 2
  node_ids: Set[str] = {n.id for n in new_graph.nodes}
  assert "s1" not in node_ids

  n1_node: LogicalNode = next(n for n in new_graph.nodes if n.id == "n1")
  assert n1_node.sharding is not None
  assert n1_node.sharding.axes == ("data", None)

  edge_sources: Set[Tuple[str, str]] = {(e.source, e.target) for e in new_graph.edges}
  assert ("n1", "n2") in edge_sources


def test_parse_partition_spec_various() -> None:
  """Test _parse_partition_spec edge cases."""
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
  """Test that edges not related to removal are preserved."""
  nodes: List[LogicalNode] = [
    LogicalNode(id="n1", kind="Linear"),
    LogicalNode(id="s1", kind="with_sharding_constraint", metadata={"arg_1": "PartitionSpec('data', None)"}),
    LogicalNode(id="n2", kind="Output"),
    LogicalNode(id="other1", kind="Other"),
    LogicalNode(id="other2", kind="Other"),
  ]
  edges: List[LogicalEdge] = [
    LogicalEdge(source="n1", target="s1"),
    LogicalEdge(source="s1", target="n2"),
    LogicalEdge(source="other1", target="other2"),  # preserved
  ]
  graph: LogicalGraph = LogicalGraph(nodes=nodes, edges=edges)
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  edge_sources: Set[Tuple[str, str]] = {(e.source, e.target) for e in new_graph.edges}
  assert ("other1", "other2") in edge_sources
