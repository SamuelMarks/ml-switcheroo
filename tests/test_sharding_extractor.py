"""Docstring."""

from typing import Optional

from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode, PartitionSpec
from ml_switcheroo.core.compiler.sharding_extractor import ShardingExtractionPass


def test_sharding_extraction_pass() -> None:
  """Docstring."""
  nodes = {
    "node1": LogicalNode(id="node1", op_type="Linear"),
    "shard_node": LogicalNode(
      id="shard_node", op_type="with_sharding_constraint", attributes={"arg_1": "PartitionSpec('data', None)"}
    ),
    "node2": LogicalNode(id="node2", op_type="Other"),
  }
  edges = [
    LogicalEdge(source="node1", target="shard_node"),
    LogicalEdge(source="shard_node", target="node2"),
  ]
  graph: LogicalGraph = LogicalGraph(nodes=nodes, edges=edges)
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)

  assert len(getattr(new_graph, "nodes")) == 2
  assert "shard_node" not in new_graph.nodes

  node1: LogicalNode = new_graph.nodes["node1"]
  assert getattr(node1, "sharding") == PartitionSpec(axes=("data", None))

  assert any(getattr(e, "source") == "node1" and getattr(e, "target") == "node2" for e in getattr(new_graph, "edges"))


def test_sharding_extraction_pass_no_match() -> None:
  """Docstring."""
  graph: LogicalGraph = LogicalGraph(nodes={"node1": LogicalNode(id="node1", op_type="Linear")}, edges=[])
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert len(getattr(new_graph, "nodes")) == 1


def test_sharding_extraction_no_source() -> None:
  """Docstring."""
  nodes = {
    "shard_node": LogicalNode(
      id="shard_node", op_type="with_sharding_constraint", attributes={"arg_1": "PartitionSpec('data', None)"}
    ),
  }
  graph: LogicalGraph = LogicalGraph(nodes=nodes, edges=[])
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert len(getattr(new_graph, "nodes")) == 1


def test_sharding_extraction_source_not_in_nodes() -> None:
  """Docstring."""
  nodes = {
    "shard_node": LogicalNode(
      id="shard_node", op_type="with_sharding_constraint", attributes={"arg_1": "PartitionSpec('data', None)"}
    ),
  }
  edges = [LogicalEdge(source="node1", target="shard_node")]
  graph: LogicalGraph = LogicalGraph(nodes=nodes, edges=edges)
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert len(getattr(new_graph, "nodes")) == 1


def test_sharding_extraction_parse_fail() -> None:
  """Docstring."""
  nodes = {
    "node1": LogicalNode(id="node1", op_type="Linear"),
    "shard_node": LogicalNode(
      id="shard_node", op_type="with_sharding_constraint", attributes={"arg_1": "PartitionSpec(invalid code)"}
    ),
  }
  edges = [LogicalEdge(source="node1", target="shard_node")]
  graph: LogicalGraph = LogicalGraph(nodes=nodes, edges=edges)
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert len(getattr(new_graph, "nodes")) == 2  # node1 and shard_node


def test_parse_partition_spec_tuple() -> None:
  """Docstring."""
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  spec: Optional[PartitionSpec] = pass_._parse_partition_spec("PartitionSpec(('data', 'tensor'), None)")
  assert spec is not None
  assert getattr(spec, "axes") == (("data", "tensor"), None)


def test_parse_partition_spec_invalid() -> None:
  """Docstring."""
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  spec: Optional[PartitionSpec] = pass_._parse_partition_spec("SomeOtherCode()")
  assert spec is None


def test_parse_partition_spec_fallback() -> None:
  """Docstring."""
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  spec: Optional[PartitionSpec] = pass_._parse_partition_spec("PartitionSpec([1, 2, 3])")
  assert spec is not None
  assert getattr(spec, "axes") == (None,)


def test_parse_partition_spec_exception() -> None:
  """Docstring."""
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  pass_._parse_partition_spec("PartitionSpec(foo=1/0)")
  pass


def test_edges_target_removal_map() -> None:
  """Docstring."""
  nodes = {
    "node1": LogicalNode(id="node1", op_type="Linear"),
    "shard_node": LogicalNode(
      id="shard_node", op_type="with_sharding_constraint", attributes={"arg_1": "PartitionSpec('data', None)"}
    ),
    "node2": LogicalNode(id="node2", op_type="Other"),
    "node3": LogicalNode(id="node3", op_type="Other"),
  }
  edges = [
    LogicalEdge(source="node1", target="shard_node"),
    LogicalEdge(source="shard_node", target="node2"),
    LogicalEdge(source="node2", target="node3"),  # Neither target nor source in removal map
  ]
  graph: LogicalGraph = LogicalGraph(nodes=nodes, edges=edges)
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert any(getattr(e, "source") == "node2" and getattr(e, "target") == "node3" for e in getattr(new_graph, "edges"))
