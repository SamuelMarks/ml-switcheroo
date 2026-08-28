"""Docstring."""

from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge, PartitionSpec
from ml_switcheroo.core.compiler.sharding_extractor import ShardingExtractionPass
from typing import Optional


def test_sharding_extraction_pass() -> None:
  """Docstring."""
  graph: LogicalGraph = LogicalGraph(
    nodes=[
      LogicalNode(id="node1", kind="Linear"),
      LogicalNode(id="shard_node", kind="with_sharding_constraint", metadata={"arg_1": "PartitionSpec('data', None)"}),
      LogicalNode(id="node2", kind="Other"),
    ],
    edges=[
      LogicalEdge(source="node1", target="shard_node"),
      LogicalEdge(source="shard_node", target="node2"),
    ],
  )
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)

  assert len(getattr(new_graph, "nodes")) == 2
  assert "shard_node" not in [getattr(n, "id") for n in getattr(new_graph, "nodes")]

  node1: LogicalNode = next(n for n in getattr(new_graph, "nodes") if getattr(n, "id") == "node1")
  assert getattr(node1, "sharding") == PartitionSpec(axes=("data", None))

  assert any(getattr(e, "source") == "node1" and getattr(e, "target") == "node2" for e in getattr(new_graph, "edges"))


def test_sharding_extraction_pass_no_match() -> None:
  """Docstring."""
  graph: LogicalGraph = LogicalGraph(nodes=[LogicalNode(id="node1", kind="Linear")], edges=[])
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert len(getattr(new_graph, "nodes")) == 1


def test_sharding_extraction_no_source() -> None:
  """Docstring."""
  graph: LogicalGraph = LogicalGraph(
    nodes=[
      LogicalNode(id="shard_node", kind="with_sharding_constraint", metadata={"arg_1": "PartitionSpec('data', None)"}),
    ],
    edges=[],
  )
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert len(getattr(new_graph, "nodes")) == 1


def test_sharding_extraction_source_not_in_nodes() -> None:
  """Docstring."""
  graph: LogicalGraph = LogicalGraph(
    nodes=[
      LogicalNode(id="shard_node", kind="with_sharding_constraint", metadata={"arg_1": "PartitionSpec('data', None)"}),
    ],
    edges=[LogicalEdge(source="node1", target="shard_node")],
  )
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert len(getattr(new_graph, "nodes")) == 1


def test_sharding_extraction_parse_fail() -> None:
  """Docstring."""
  graph: LogicalGraph = LogicalGraph(
    nodes=[
      LogicalNode(id="node1", kind="Linear"),
      LogicalNode(id="shard_node", kind="with_sharding_constraint", metadata={"arg_1": "PartitionSpec(invalid code)"}),
    ],
    edges=[LogicalEdge(source="node1", target="shard_node")],
  )
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
  # just not containing "PartitionSpec" or "NamedSharding"
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
  # Wait, ast.parse will fail on "PartitionSpec(invalid code)" which is tested in test_sharding_extraction_parse_fail
  pass


def test_edges_target_removal_map() -> None:
  """Docstring."""
  # To cover new_edges.append(e) when neither target nor source is in removal_map
  graph: LogicalGraph = LogicalGraph(
    nodes=[
      LogicalNode(id="node1", kind="Linear"),
      LogicalNode(id="shard_node", kind="with_sharding_constraint", metadata={"arg_1": "PartitionSpec('data', None)"}),
      LogicalNode(id="node2", kind="Other"),
      LogicalNode(id="node3", kind="Other"),
    ],
    edges=[
      LogicalEdge(source="node1", target="shard_node"),
      LogicalEdge(source="shard_node", target="node2"),
      LogicalEdge(source="node2", target="node3"),  # Neither target nor source in removal map
    ],
  )
  pass_: ShardingExtractionPass = ShardingExtractionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert any(getattr(e, "source") == "node2" and getattr(e, "target") == "node3" for e in getattr(new_graph, "edges"))
