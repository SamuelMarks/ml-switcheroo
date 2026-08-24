"""Docstring."""

from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, PartitionSpec
from ml_switcheroo.core.compiler.sharding import ShardingInferencePass


def test_sharding_inference_pass():
  """Docstring."""
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="q_proj_1", kind="Linear"),
      LogicalNode(id="o_proj_1", kind="Linear"),
      LogicalNode(id="embed_1", kind="Embedding"),
      LogicalNode(id="conv2d_1", kind="Conv2d"),
      LogicalNode(id="other_node", kind="Other"),
    ],
    edges=[],
  )
  pass_ = ShardingInferencePass()
  new_graph = pass_.apply(graph)

  assert new_graph.mesh is not None
  assert new_graph.mesh.shape == {"data": 1, "tensor": 1}

  q_proj_node = next(n for n in new_graph.nodes if n.id == "q_proj_1")
  assert q_proj_node.sharding == PartitionSpec(axes=(None, "tensor"))

  o_proj_node = next(n for n in new_graph.nodes if n.id == "o_proj_1")
  assert o_proj_node.sharding == PartitionSpec(axes=("tensor", None))

  embed_node = next(n for n in new_graph.nodes if n.id == "embed_1")
  assert embed_node.sharding == PartitionSpec(axes=("tensor", None))

  conv_node = next(n for n in new_graph.nodes if n.id == "conv2d_1")
  assert conv_node.sharding == PartitionSpec(axes=("data", None))

  other_node = next(n for n in new_graph.nodes if n.id == "other_node")
  assert other_node.sharding is None
