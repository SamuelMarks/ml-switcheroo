"""Test suite for the Sharding Extractor module."""

from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.core.compiler.sharding_extractor import ShardingExtractionPass


def test_sharding_extraction_pass():
  """Verifies the behavior of sharding extraction pass."""
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="x", kind="Input"),
      LogicalNode(id="fc", kind="Linear"),
      LogicalNode(
        id="func_sharding",
        kind="jax.lax.with_sharding_constraint",
        metadata={"arg_0": "x", "arg_1": "jax.sharding.PartitionSpec('data', None)"},
      ),
      LogicalNode(id="out", kind="Output"),
    ],
    edges=[LogicalEdge("x", "fc"), LogicalEdge("fc", "func_sharding"), LogicalEdge("func_sharding", "out")],
  )
  pass_ = ShardingExtractionPass()
  extracted_graph = pass_.apply(graph)
  node_ids = {n.id for n in extracted_graph.nodes}
  assert "func_sharding" not in node_ids
  assert "fc" in node_ids
  edges = [(e.source, e.target) for e in extracted_graph.edges]
  assert ("fc", "out") in edges
  fc_node = next((n for n in extracted_graph.nodes if n.id == "fc"))
  assert fc_node.sharding is not None
  assert fc_node.sharding.axes == ("data", None)


def test_sharding_extraction_pass_complex_spec():
  """Verifies the behavior of sharding extraction pass complex spec."""
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="fc", kind="Linear"),
      LogicalNode(
        id="func_sharding",
        kind="with_sharding_constraint",
        metadata={"arg_1": "PartitionSpec('data', ('model', 'tensor'))"},
      ),
    ],
    edges=[LogicalEdge("fc", "func_sharding")],
  )
  pass_ = ShardingExtractionPass()
  extracted_graph = pass_.apply(graph)
  fc_node = next((n for n in extracted_graph.nodes if n.id == "fc"))
  assert fc_node.sharding.axes == ("data", ("model", "tensor"))


def test_sharding_extraction_pass_no_match():
  """Verifies the behavior of sharding extraction pass no match."""
  graph = LogicalGraph(nodes=[LogicalNode(id="fc", kind="Linear")])
  pass_ = ShardingExtractionPass()
  extracted_graph = pass_.apply(graph)
  assert len(extracted_graph.nodes) == 1


def test_sharding_extraction_pass_invalid_ast():
  """Verifies the behavior of sharding extraction pass invalid AST."""
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="fc", kind="Linear"),
      LogicalNode(id="func_sharding", kind="with_sharding_constraint", metadata={"arg_1": "PartitionSpec('data', "}),
    ],
    edges=[LogicalEdge("fc", "func_sharding")],
  )
  pass_ = ShardingExtractionPass()
  extracted_graph = pass_.apply(graph)
  fc_node = next((n for n in extracted_graph.nodes if n.id == "fc"))
  assert fc_node.sharding is None


def test_sharding_extraction_pass_fallback_arg():
  """Verifies the behavior of sharding extraction pass fallback argument."""
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="fc", kind="Linear"),
      LogicalNode(
        id="func_sharding", kind="with_sharding_constraint", metadata={"arg_1": "PartitionSpec('data', [1, 2])"}
      ),
    ],
    edges=[LogicalEdge("fc", "func_sharding")],
  )
  pass_ = ShardingExtractionPass()
  extracted_graph = pass_.apply(graph)
  fc_node = next((n for n in extracted_graph.nodes if n.id == "fc"))
  assert fc_node.sharding.axes == ("data", None)


def test_sharding_extraction_pass_no_partition_spec():
  """Verifies the behavior of sharding extraction pass no partition spec."""
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="fc", kind="Linear"),
      LogicalNode(id="func_sharding", kind="with_sharding_constraint", metadata={"arg_1": "SomeOtherConstraint()"}),
    ],
    edges=[LogicalEdge("fc", "func_sharding")],
  )
  pass_ = ShardingExtractionPass()
  extracted_graph = pass_.apply(graph)
  assert "func_sharding" in {n.id for n in extracted_graph.nodes}
