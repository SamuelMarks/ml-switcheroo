"""Docstring."""

from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, PartitionSpec
from ml_switcheroo.core.compiler.sharding import ShardingInferencePass


def test_sharding_inference_pass() -> None:
  """Docstring."""
  graph: LogicalGraph = LogicalGraph(
    nodes=[
      LogicalNode(id="q_proj_1", kind="Linear"),
      LogicalNode(id="o_proj_1", kind="Linear"),
      LogicalNode(id="embed_1", kind="Embedding"),
      LogicalNode(id="conv2d_1", kind="Conv2d"),
      LogicalNode(id="other_node", kind="Other"),
    ],
    edges=[],
  )
  pass_: ShardingInferencePass = ShardingInferencePass()
  new_graph: LogicalGraph = pass_.apply(graph)

  assert getattr(new_graph, "mesh") is not None
  assert getattr(getattr(new_graph, "mesh"), "shape") == {"data": 1, "tensor": 1}

  q_proj_node: LogicalNode = next(n for n in getattr(new_graph, "nodes") if getattr(n, "id") == "q_proj_1")
  assert getattr(q_proj_node, "sharding") == PartitionSpec(axes=(None, "tensor"))

  o_proj_node: LogicalNode = next(n for n in getattr(new_graph, "nodes") if getattr(n, "id") == "o_proj_1")
  assert getattr(o_proj_node, "sharding") == PartitionSpec(axes=("tensor", None))

  embed_node: LogicalNode = next(n for n in getattr(new_graph, "nodes") if getattr(n, "id") == "embed_1")
  assert getattr(embed_node, "sharding") == PartitionSpec(axes=("tensor", None))

  conv_node: LogicalNode = next(n for n in getattr(new_graph, "nodes") if getattr(n, "id") == "conv2d_1")
  assert getattr(conv_node, "sharding") == PartitionSpec(axes=("data", None))

  other_node: LogicalNode = next(n for n in getattr(new_graph, "nodes") if getattr(n, "id") == "other_node")
  assert getattr(other_node, "sharding") is None
