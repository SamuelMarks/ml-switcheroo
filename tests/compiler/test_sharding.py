"""Test suite for the Sharding module."""

from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode
from ml_switcheroo.core.compiler.sharding import ShardingInferencePass


def test_sharding_inference_heuristics():
  """Verifies the behavior of sharding inference heuristics."""
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="q_proj", kind="Linear"),
      LogicalNode(id="o_proj", kind="Linear"),
      LogicalNode(id="up_proj", kind="Linear"),
      LogicalNode(id="embed", kind="Embedding"),
      LogicalNode(id="some_layer", kind="Linear"),
      LogicalNode(id="activation", kind="Relu"),
    ]
  )
  pass_ = ShardingInferencePass()
  annotated_graph = pass_.apply(graph)
  assert annotated_graph.mesh is not None
  assert annotated_graph.mesh.shape["tensor"] == 1
  for node in annotated_graph.nodes:
    if node.id in ["q_proj", "up_proj"]:
      assert node.sharding.axes == (None, "tensor")
    elif node.id in ["o_proj", "embed"]:
      assert node.sharding.axes == ("tensor", None)
    elif node.id == "some_layer":
      assert node.sharding.axes == ("data", None)
    else:
      assert node.sharding is None
