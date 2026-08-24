"""Test suite for the compiler sharding extractor pass."""

from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.core.compiler.sharding_extractor import ShardingExtractionPass


def test_sharding_extraction_pass_no_nodes():
  """Test with no sharding constraint nodes."""
  graph = LogicalGraph(nodes=[LogicalNode(id="n1", kind="Linear")], edges=[])
  pass_ = ShardingExtractionPass()
  new_graph = pass_.apply(graph)
  assert len(new_graph.nodes) == 1
  assert new_graph.nodes[0].id == "n1"
  assert new_graph.nodes[0].sharding is None


def test_sharding_extraction_pass_missing_source():
  """Test with sharding constraint node that has no incoming edge."""
  nodes = [
    LogicalNode(id="n1", kind="Linear"),
    LogicalNode(id="s1", kind="with_sharding_constraint", metadata={"arg_1": "PartitionSpec('data', None)"}),
  ]
  graph = LogicalGraph(nodes=nodes, edges=[])
  pass_ = ShardingExtractionPass()
  new_graph = pass_.apply(graph)
  assert len(new_graph.nodes) == 2  # The node should not be removed


def test_sharding_extraction_pass_missing_source_node():
  """Test with sharding constraint node but the source node does not exist in graph.nodes."""
  nodes = [LogicalNode(id="s1", kind="with_sharding_constraint", metadata={"arg_1": "PartitionSpec('data', None)"})]
  edges = [
    LogicalEdge(source="n1", target="s1")  # n1 not in nodes
  ]
  graph = LogicalGraph(nodes=nodes, edges=edges)
  pass_ = ShardingExtractionPass()
  new_graph = pass_.apply(graph)
  assert len(new_graph.nodes) == 1


def test_sharding_extraction_pass_invalid_parse():
  """Test with sharding constraint that fails to parse."""
  nodes = [
    LogicalNode(id="n1", kind="Linear"),
    LogicalNode(id="s1", kind="with_sharding_constraint", metadata={"arg_1": "PartitionSpec(!"}),
  ]
  edges = [LogicalEdge(source="n1", target="s1")]
  graph = LogicalGraph(nodes=nodes, edges=edges)
  pass_ = ShardingExtractionPass()
  new_graph = pass_.apply(graph)
  assert len(new_graph.nodes) == 2


def test_sharding_extraction_pass_success():
  """Test successfully extracting sharding constraints and rewiring edges."""
  nodes = [
    LogicalNode(id="n1", kind="Linear"),
    LogicalNode(id="s1", kind="with_sharding_constraint", metadata={"arg_1": "PartitionSpec('data', None)"}),
    LogicalNode(id="n2", kind="Output"),
  ]
  edges = [
    LogicalEdge(source="n1", target="s1"),
    LogicalEdge(source="s1", target="n2"),
    # test that duplicate edge is not added
    LogicalEdge(source="n1", target="n2"),  # already exists, tests the `if new_edge not in new_edges` block
  ]
  graph = LogicalGraph(nodes=nodes, edges=edges)
  pass_ = ShardingExtractionPass()
  new_graph = pass_.apply(graph)

  assert len(new_graph.nodes) == 2
  node_ids = {n.id for n in new_graph.nodes}
  assert "s1" not in node_ids

  n1_node = next(n for n in new_graph.nodes if n.id == "n1")
  assert n1_node.sharding is not None
  assert n1_node.sharding.axes == ("data", None)

  edge_sources = {(e.source, e.target) for e in new_graph.edges}
  assert ("n1", "n2") in edge_sources


def test_parse_partition_spec_various():
  """Test _parse_partition_spec edge cases."""
  pass_ = ShardingExtractionPass()

  assert pass_._parse_partition_spec("foo()") is None
  assert pass_._parse_partition_spec("PartitionSpec('tensor')").axes == ("tensor",)
  assert pass_._parse_partition_spec("NamedSharding('tensor')").axes == ("tensor",)

  # Test tuple support
  res = pass_._parse_partition_spec("PartitionSpec(('data', 'tensor'), None)")
  assert res.axes == (("data", "tensor"), None)

  # Test unknown arg type
  res = pass_._parse_partition_spec("PartitionSpec(some_var)")
  assert res.axes == (None,)


def test_sharding_extraction_pass_preserve_other_edges():
  """Test that edges not related to removal are preserved."""
  nodes = [
    LogicalNode(id="n1", kind="Linear"),
    LogicalNode(id="s1", kind="with_sharding_constraint", metadata={"arg_1": "PartitionSpec('data', None)"}),
    LogicalNode(id="n2", kind="Output"),
    LogicalNode(id="other1", kind="Other"),
    LogicalNode(id="other2", kind="Other"),
  ]
  edges = [
    LogicalEdge(source="n1", target="s1"),
    LogicalEdge(source="s1", target="n2"),
    LogicalEdge(source="other1", target="other2"),  # preserved
  ]
  graph = LogicalGraph(nodes=nodes, edges=edges)
  pass_ = ShardingExtractionPass()
  new_graph = pass_.apply(graph)
  edge_sources = {(e.source, e.target) for e in new_graph.edges}
  assert ("other1", "other2") in edge_sources
