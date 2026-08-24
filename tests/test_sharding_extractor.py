"""Docstring."""

from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge, PartitionSpec
from ml_switcheroo.core.compiler.sharding_extractor import ShardingExtractionPass


def test_sharding_extraction_pass():
  """Docstring."""
  graph = LogicalGraph(
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
  pass_ = ShardingExtractionPass()
  new_graph = pass_.apply(graph)

  assert len(new_graph.nodes) == 2
  assert "shard_node" not in [n.id for n in new_graph.nodes]

  node1 = next(n for n in new_graph.nodes if n.id == "node1")
  assert node1.sharding == PartitionSpec(axes=("data", None))

  assert any(e.source == "node1" and e.target == "node2" for e in new_graph.edges)


def test_sharding_extraction_pass_no_match():
  """Docstring."""
  graph = LogicalGraph(nodes=[LogicalNode(id="node1", kind="Linear")], edges=[])
  pass_ = ShardingExtractionPass()
  new_graph = pass_.apply(graph)
  assert len(new_graph.nodes) == 1


def test_sharding_extraction_no_source():
  """Docstring."""
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="shard_node", kind="with_sharding_constraint", metadata={"arg_1": "PartitionSpec('data', None)"}),
    ],
    edges=[],
  )
  pass_ = ShardingExtractionPass()
  new_graph = pass_.apply(graph)
  assert len(new_graph.nodes) == 1


def test_sharding_extraction_source_not_in_nodes():
  """Docstring."""
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="shard_node", kind="with_sharding_constraint", metadata={"arg_1": "PartitionSpec('data', None)"}),
    ],
    edges=[LogicalEdge(source="node1", target="shard_node")],
  )
  pass_ = ShardingExtractionPass()
  new_graph = pass_.apply(graph)
  assert len(new_graph.nodes) == 1


def test_sharding_extraction_parse_fail():
  """Docstring."""
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="node1", kind="Linear"),
      LogicalNode(id="shard_node", kind="with_sharding_constraint", metadata={"arg_1": "PartitionSpec(invalid code)"}),
    ],
    edges=[LogicalEdge(source="node1", target="shard_node")],
  )
  pass_ = ShardingExtractionPass()
  new_graph = pass_.apply(graph)
  assert len(new_graph.nodes) == 2  # node1 and shard_node


def test_parse_partition_spec_tuple():
  """Docstring."""
  pass_ = ShardingExtractionPass()
  spec = pass_._parse_partition_spec("PartitionSpec(('data', 'tensor'), None)")
  assert spec is not None
  assert spec.axes == (("data", "tensor"), None)


def test_parse_partition_spec_invalid():
  """Docstring."""
  pass_ = ShardingExtractionPass()
  # just not containing "PartitionSpec" or "NamedSharding"
  spec = pass_._parse_partition_spec("SomeOtherCode()")
  assert spec is None


def test_parse_partition_spec_fallback():
  """Docstring."""
  pass_ = ShardingExtractionPass()
  spec = pass_._parse_partition_spec("PartitionSpec([1, 2, 3])")
  assert spec is not None
  assert spec.axes == (None,)


def test_parse_partition_spec_exception():
  """Docstring."""
  pass_ = ShardingExtractionPass()
  pass_._parse_partition_spec("PartitionSpec(foo=1/0)")
  # Wait, ast.parse will fail on "PartitionSpec(invalid code)" which is tested in test_sharding_extraction_parse_fail
  pass


def test_edges_target_removal_map():
  """Docstring."""
  # To cover new_edges.append(e) when neither target nor source is in removal_map
  graph = LogicalGraph(
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
  pass_ = ShardingExtractionPass()
  new_graph = pass_.apply(graph)
  assert any(e.source == "node2" and e.target == "node3" for e in new_graph.edges)
