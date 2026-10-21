"""Test suite for the compiler sharding extractor pass."""

from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode, PartitionSpec
from ml_switcheroo.core.compiler.sharding_extractor import ShardingExtractionPass


def test_sharding_extraction_pass_no_nodes() -> None:
  """Verify sharding extraction pass no nodes behavior."""
  graph: LogicalGraph = LogicalGraph(nodes={"n1": LogicalNode(id="n1", op_type="Linear")}, edges=[])
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert len(new_graph.nodes) == 1
  assert "n1" in new_graph.nodes
  assert new_graph.nodes["n1"].id == "n1"
  assert new_graph.nodes["n1"].sharding is None


def test_sharding_extraction_pass_missing_source() -> None:
  """Verify sharding extraction pass missing source behavior."""
  nodes: list[LogicalNode] = [
    LogicalNode(id="n1", op_type="Linear"),
    LogicalNode(id="s1", op_type="with_sharding_constraint", attributes={"arg_1": "PartitionSpec('data', None)"}),
  ]
  graph: LogicalGraph = LogicalGraph(nodes={n.id: n for n in nodes}, edges=[])
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert len(new_graph.nodes) == 2  # The node should not be removed


def test_sharding_extraction_pass_missing_source_node() -> None:
  """Verify sharding extraction pass missing source node behavior."""
  nodes: list[LogicalNode] = [
    LogicalNode(id="s1", op_type="with_sharding_constraint", attributes={"arg_1": "PartitionSpec('data', None)"})
  ]
  edges: list[LogicalEdge] = [
    LogicalEdge(source="n1", target="s1")  # n1 not in nodes
  ]
  graph: LogicalGraph = LogicalGraph(nodes={n.id: n for n in nodes}, edges=edges)
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert len(new_graph.nodes) == 1


def test_sharding_extraction_pass_invalid_parse() -> None:
  """Verify sharding extraction pass invalid parse behavior."""
  nodes: list[LogicalNode] = [
    LogicalNode(id="n1", op_type="Linear"),
    LogicalNode(id="s1", op_type="with_sharding_constraint", attributes={"arg_1": "PartitionSpec(!"}),
  ]
  edges: list[LogicalEdge] = [LogicalEdge(source="n1", target="s1")]
  graph: LogicalGraph = LogicalGraph(nodes={n.id: n for n in nodes}, edges=edges)
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert len(new_graph.nodes) == 2


def test_sharding_extraction_pass_success() -> None:
  """Verify sharding extraction pass success behavior."""
  nodes: list[LogicalNode] = [
    LogicalNode(id="n1", op_type="Linear"),
    LogicalNode(id="s1", op_type="with_sharding_constraint", attributes={"arg_1": "PartitionSpec('data', None)"}),
    LogicalNode(id="n2", op_type="Output"),
  ]
  edges: list[LogicalEdge] = [
    LogicalEdge(source="n1", target="s1"),
    LogicalEdge(source="s1", target="n2"),
    # test that duplicate edge is not added
    LogicalEdge(source="n1", target="n2"),  # already exists, tests the `if new_edge not in new_edges` block
  ]
  graph: LogicalGraph = LogicalGraph(nodes={n.id: n for n in nodes}, edges=edges)
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)

  assert len(new_graph.nodes) == 2
  node_ids: set[str] = set(new_graph.nodes.keys())
  assert "s1" not in node_ids

  n1_node: LogicalNode = new_graph.nodes["n1"]
  assert n1_node.sharding is not None
  assert n1_node.sharding.axes == ("data", None)

  edge_sources: set[tuple[str, str]] = {(e.source, e.target) for e in new_graph.edges}
  assert ("n1", "n2") in edge_sources


def test_parse_partition_spec_various() -> None:
  """Verify parse partition spec various behavior."""
  pass_: ShardingExtractionPass = ShardingExtractionPass()

  assert pass_._parse_partition_spec("foo()") is None
  res1: PartitionSpec | None = pass_._parse_partition_spec("PartitionSpec('tensor')")
  assert res1 is not None and res1.axes == ("tensor",)
  res2: PartitionSpec | None = pass_._parse_partition_spec("NamedSharding('tensor')")
  assert res2 is not None and res2.axes == ("tensor",)

  # Test tuple support
  res3: PartitionSpec | None = pass_._parse_partition_spec("PartitionSpec(('data', 'tensor'), None)")
  assert res3 is not None and res3.axes == (("data", "tensor"), None)

  # Test unknown arg type
  res4: PartitionSpec | None = pass_._parse_partition_spec("PartitionSpec(some_var)")
  assert res4 is not None and res4.axes == (None,)


def test_sharding_extraction_pass_preserve_other_edges() -> None:
  """Verify sharding extraction pass preserve other edges behavior."""
  nodes: list[LogicalNode] = [
    LogicalNode(id="n1", op_type="Linear"),
    LogicalNode(id="s1", op_type="with_sharding_constraint", attributes={"arg_1": "PartitionSpec('data', None)"}),
    LogicalNode(id="n2", op_type="Output"),
    LogicalNode(id="other1", op_type="Other"),
    LogicalNode(id="other2", op_type="Other"),
  ]
  edges: list[LogicalEdge] = [
    LogicalEdge(source="n1", target="s1"),
    LogicalEdge(source="s1", target="n2"),
    LogicalEdge(source="other1", target="other2"),  # preserved
  ]
  graph: LogicalGraph = LogicalGraph(nodes={n.id: n for n in nodes}, edges=edges)
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  edge_sources: set[tuple[str, str]] = {(e.source, e.target) for e in new_graph.edges}
  assert ("other1", "other2") in edge_sources


def test_sharding_extraction_pass_metadata_fallback() -> None:
  """Verifies sharding extraction reads PartitionSpec from attributes on LogicalNode."""
  n1 = LogicalNode(id="n1", op_type="Linear")
  s1 = LogicalNode(
    id="s1",
    op_type="with_sharding_constraint",
    attributes={"arg_1": "PartitionSpec('data', None)"},
  )
  nodes = {"n1": n1, "s1": s1}
  edges: list[LogicalEdge] = [LogicalEdge(source="n1", target="s1")]
  graph: LogicalGraph = LogicalGraph(nodes=nodes, edges=edges)
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert new_graph.nodes["n1"].sharding is not None
  assert new_graph.nodes["n1"].sharding.axes == ("data", None)
