"""Test suite for the Sharding module."""

import typing

from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode
from ml_switcheroo.core.compiler.sharding import ShardingInferencePass


def test_sharding_inference_heuristics() -> None:
  """Verifies the behavior of sharding inference heuristics."""
  nodes: typing.Any = {
    "q_proj": LogicalNode(id="q_proj", op_type="Linear"),
    "o_proj": LogicalNode(id="o_proj", op_type="Linear"),
    "up_proj": LogicalNode(id="up_proj", op_type="Linear"),
    "embed": LogicalNode(id="embed", op_type="Embedding"),
    "some_layer": LogicalNode(id="some_layer", op_type="Linear"),
    "activation": LogicalNode(id="activation", op_type="Relu"),
  }
  graph: typing.Any = LogicalGraph(nodes=nodes)
  pass_: typing.Any = ShardingInferencePass()
  annotated_graph: typing.Any = pass_.apply(graph)
  assert annotated_graph.mesh is not None
  assert annotated_graph.mesh.shape["tensor"] == 1
  for node in annotated_graph.nodes.values():
    if node.id in ["q_proj", "up_proj"]:
      assert node.sharding.axes == (None, "tensor")
    elif node.id in ["o_proj", "embed"]:
      assert node.sharding.axes == ("tensor", None)
    elif node.id == "some_layer":
      assert node.sharding.axes == ("data", None)
    else:
      assert node.sharding is None
