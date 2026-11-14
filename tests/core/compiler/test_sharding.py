"""Unit tests for compiler sharding passes and analytical memory/comm estimators (`ml_switcheroo.core.compiler.sharding`)."""

from unittest.mock import patch

from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalMesh, LogicalNode, PartitionSpec
from ml_switcheroo.core.compiler.sharding import (
  MlxDistributedShardingPass,
  PyTorchDTensorShardingPass,
  ShardingInferencePass,
  apply_sharding_passes,
  estimate_tensor_memory,
)


def test_sharding_inference_dtensor_mlx_and_pipeline() -> None:
  """Verify `ShardingInferencePass`, `PyTorchDTensorShardingPass`, `MlxDistributedShardingPass`, and pipeline."""
  assert ShardingInferencePass().mesh.shape == {"data": 1, "tensor": 1}
  custom_mesh = LogicalMesh(shape={"data": 2, "tensor": 4})
  assert ShardingInferencePass(mesh=custom_mesh).mesh.shape == {"data": 2, "tensor": 4}

  nodes = [
    LogicalNode(id="q_proj", op_type="Linear"),
    LogicalNode(id="k_proj_layer", op_type="Linear"),
    LogicalNode(id="V_PROJ", op_type="Linear"),
    LogicalNode(id="gate_proj", op_type="Linear"),
    LogicalNode(id="up_proj", op_type="Linear"),
    LogicalNode(id="o_proj", op_type="Linear"),
    LogicalNode(id="down_proj", op_type="Linear"),
    LogicalNode(id="embed_tokens", op_type="Embedding"),
    LogicalNode(id="my_embedding_layer", op_type="Embedding"),
    LogicalNode(id="fc1", op_type="Linear"),
    LogicalNode(id="conv1", op_type="Conv2d"),
    LogicalNode(id="conv3", op_type="Conv3d"),
    LogicalNode(id="relu", op_type="ReLU"),
  ]
  graph = LogicalGraph(nodes={n.id: n for n in nodes}, edges=[])
  inferred = ShardingInferencePass().apply(graph)
  assert inferred.mesh == ShardingInferencePass().mesh

  for col_id in ("q_proj", "k_proj_layer", "V_PROJ", "gate_proj", "up_proj"):
    assert inferred.nodes[col_id].sharding == PartitionSpec(axes=(None, "tensor"))
  for row_id in ("o_proj", "down_proj", "embed_tokens", "my_embedding_layer"):
    assert inferred.nodes[row_id].sharding == PartitionSpec(axes=("tensor", None))
  for dp_id in ("fc1", "conv1", "conv3"):
    assert inferred.nodes[dp_id].sharding == PartitionSpec(axes=("data", None))
  assert inferred.nodes["relu"].sharding is None

  # PyTorch DTensor / FSDP2 pass
  dt_graph = PyTorchDTensorShardingPass(world_size=4).apply(
    LogicalGraph(
      nodes={
        "q_proj": LogicalNode("q_proj", "Linear"),
        "o_proj": LogicalNode("o_proj", "Linear"),
        "fc_layer": LogicalNode("fc_layer", "Linear"),
        "relu": LogicalNode("relu", "ReLU"),
      }
    )
  )
  assert dt_graph.nodes["q_proj"].attributes["dtensor_placement"] == "Shard(dim=1)"
  assert dt_graph.nodes["q_proj"].attributes["dtensor_world_size"] == 4
  assert dt_graph.nodes["o_proj"].attributes["dtensor_placement"] == "Shard(dim=0)"
  assert dt_graph.nodes["fc_layer"].attributes["dtensor_placement"] == "FSDP(dim=0)"
  assert "dtensor_placement" not in dt_graph.nodes["relu"].attributes

  # MLX distributed pass
  mlx_pass = MlxDistributedShardingPass(num_nodes=2)
  mlx_graph = mlx_pass.apply(
    LogicalGraph(nodes={"embed": LogicalNode("embed", "Embedding"), "relu": LogicalNode("relu", "ReLU")})
  )
  assert mlx_graph.nodes["embed"].attributes["mlx_distributed"] is True
  assert mlx_graph.nodes["embed"].attributes["mlx_nodes"] == 2
  assert len(mlx_pass.diagnostics) == 1
  assert "mlx_distributed" not in mlx_graph.nodes["relu"].attributes

  # `apply_sharding_passes` unfused and fused
  tiny = LogicalGraph(nodes={"q_proj": LogicalNode("q_proj", "Linear")})
  assert apply_sharding_passes(tiny, fuse=False).nodes["q_proj"].sharding == PartitionSpec(axes=(None, "tensor"))
  assert apply_sharding_passes(tiny, fuse=True).mesh is not None


def test_memory_communication_and_device_placement_estimators() -> None:
  """Verify `estimate_tensor_memory`, graph memory/communication estimators, and `infer_device_placement`."""
  assert estimate_tensor_memory([10, 20], dtype="float32") == 800.0
  assert estimate_tensor_memory(["B", 128], dtype="float32", symbolic_map={"B": 4}) == 2048.0
  assert estimate_tensor_memory([None, "64", "-1", "unknown", 2.5], dtype="float32") > 0.0

  with patch("ml_ecosystem_snapshots.compliance.estimate_tensor_memory_bytes", side_effect=RuntimeError("err")):
    assert estimate_tensor_memory([10, 20], dtype="float64") == 1600.0
    assert estimate_tensor_memory([10, 20], dtype="float16") == 400.0
    assert estimate_tensor_memory([10, 20], dtype="bfloat16") == 400.0
    assert estimate_tensor_memory([10, 20], dtype="int8") == 200.0
    assert estimate_tensor_memory([10, 20], dtype="int4") == 200.0
    assert estimate_tensor_memory([], dtype="float32") == 0.0

  pass_ = ShardingInferencePass()
  mem_graph = LogicalGraph(
    nodes={
      "w1": LogicalNode("w1", "Linear", attributes={"shape": [10, 10], "dtype": "float32"}),
      "act1": LogicalNode("act1", "Relu", attributes={"shape": [10, 10], "dtype": "float32"}),
      "noshape": LogicalNode("noshape", "Relu"),
    }
  )
  assert (
    pass_.estimate_graph_memory_bytes(
      mem_graph, include_activations=False, include_gradients=False, workspace_overhead_ratio=0.0
    )
    == 400.0
  )
  assert (
    pass_.estimate_graph_memory_bytes(
      mem_graph, include_activations=True, include_gradients=True, workspace_overhead_ratio=0.1
    )
    == 1320.0
  )

  comm_graph = LogicalGraph(
    nodes={
      "c1": LogicalNode(
        "c1",
        "all_reduce",
        shape_metadata=[10, 10],
        attributes={"shape": [10, 10], "dtype": "float32", "device_count": 2},
      ),
      "r1": LogicalNode("r1", "Relu"),
      "c2": LogicalNode("c2", "collective.all_gather"),
    }
  )
  assert pass_.estimate_graph_communication_bytes(comm_graph) > 0.0
  with patch("ml_switcheroo_ir.validator.estimate_graph_communication_volume", side_effect=RuntimeError("err")):
    assert pass_.estimate_graph_communication_bytes(comm_graph) == 400.0

  place_nodes = {
    f"node_{i}": LogicalNode(f"node_{i}", "Linear", attributes={"shape": [10, 10], "dtype": "float32"}) for i in range(5)
  }
  place_nodes["no_shape"] = LogicalNode("no_shape", "Identity")
  place_graph = LogicalGraph(nodes=place_nodes)
  placements = pass_.infer_device_placement(place_graph, device_capacity_bytes=700.0, num_devices=3)
  assert (placements["node_0"], placements["node_1"], placements["node_4"]) == (0, 1, 2)
  assert (place_graph.nodes["node_0"].device, place_graph.nodes["node_4"].device) == ("cuda:0", "cuda:2")
