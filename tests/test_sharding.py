"""Docstring."""

from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, PartitionSpec
from ml_switcheroo.core.compiler.sharding import ShardingInferencePass


def test_sharding_inference_pass() -> None:
  """Docstring."""
  nodes = {
    "q_proj_1": LogicalNode(id="q_proj_1", op_type="Linear"),
    "o_proj_1": LogicalNode(id="o_proj_1", op_type="Linear"),
    "embed_1": LogicalNode(id="embed_1", op_type="Embedding"),
    "conv2d_1": LogicalNode(id="conv2d_1", op_type="Conv2d"),
    "other_node": LogicalNode(id="other_node", op_type="Other"),
  }
  graph: LogicalGraph = LogicalGraph(
    nodes=nodes,
    edges=[],
  )
  pass_: ShardingInferencePass = ShardingInferencePass()
  new_graph: LogicalGraph = pass_.apply(graph)

  assert getattr(new_graph, "mesh") is not None
  assert getattr(getattr(new_graph, "mesh"), "shape") == {"data": 1, "tensor": 1}

  q_proj_node: LogicalNode = new_graph.nodes["q_proj_1"]
  assert getattr(q_proj_node, "sharding") == PartitionSpec(axes=(None, "tensor"))

  o_proj_node: LogicalNode = new_graph.nodes["o_proj_1"]
  assert getattr(o_proj_node, "sharding") == PartitionSpec(axes=("tensor", None))

  embed_node: LogicalNode = new_graph.nodes["embed_1"]
  assert getattr(embed_node, "sharding") == PartitionSpec(axes=("tensor", None))

  conv_node: LogicalNode = new_graph.nodes["conv2d_1"]
  assert getattr(conv_node, "sharding") == PartitionSpec(axes=("data", None))

  other_node: LogicalNode = new_graph.nodes["other_node"]
  assert getattr(other_node, "sharding") is None
