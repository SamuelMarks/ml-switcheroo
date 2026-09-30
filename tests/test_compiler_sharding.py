"""Test suite for the compiler sharding pass."""

from typing import Dict, List

from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalMesh, LogicalNode, PartitionSpec
from ml_switcheroo.core.compiler.sharding import (
  MlxDistributedShardingPass,
  PyTorchDTensorShardingPass,
  ShardingInferencePass,
  estimate_tensor_memory,
)


def test_sharding_inference_pass_default_mesh() -> None:
  """Docstring."""
  pass_: ShardingInferencePass = ShardingInferencePass()
  assert pass_.mesh.shape == {"data": 1, "tensor": 1}


def test_sharding_inference_pass_custom_mesh() -> None:
  """Docstring."""
  mesh: LogicalMesh = LogicalMesh(shape={"data": 2, "tensor": 4})
  pass_: ShardingInferencePass = ShardingInferencePass(mesh=mesh)
  assert pass_.mesh.shape == {"data": 2, "tensor": 4}


def test_sharding_inference_pass_apply() -> None:
  """Docstring."""
  nodes: List[LogicalNode] = [
    # Column Parallel matches
    LogicalNode(id="q_proj", op_type="Linear"),
    LogicalNode(id="k_proj_layer", op_type="Linear"),
    LogicalNode(id="V_PROJ", op_type="Linear"),
    LogicalNode(id="gate_proj", op_type="Linear"),
    LogicalNode(id="up_proj", op_type="Linear"),
    # Row Parallel matches
    LogicalNode(id="o_proj", op_type="Linear"),
    LogicalNode(id="down_proj", op_type="Linear"),
    LogicalNode(id="embed_tokens", op_type="Embedding"),
    # Fallback FSDP matches
    LogicalNode(id="fc1", op_type="Linear"),
    LogicalNode(id="conv1", op_type="Conv2d"),
    LogicalNode(id="conv3", op_type="Conv3d"),
    LogicalNode(id="my_embedding_layer", op_type="Embedding"),
    # Ignored node
    LogicalNode(id="relu", op_type="ReLU"),
  ]
  graph: LogicalGraph = LogicalGraph(nodes={n.id: n for n in nodes}, edges=[])
  pass_: ShardingInferencePass = ShardingInferencePass()
  new_graph: LogicalGraph = pass_.apply(graph)

  # Check if mesh is attached
  assert new_graph.mesh == pass_.mesh

  node_dict: Dict[str, LogicalNode] = dict(new_graph.nodes)

  # Column parallel
  assert node_dict["q_proj"].sharding == PartitionSpec(axes=(None, "tensor"))
  assert node_dict["k_proj_layer"].sharding == PartitionSpec(axes=(None, "tensor"))
  assert node_dict["V_PROJ"].sharding == PartitionSpec(axes=(None, "tensor"))
  assert node_dict["gate_proj"].sharding == PartitionSpec(axes=(None, "tensor"))
  assert node_dict["up_proj"].sharding == PartitionSpec(axes=(None, "tensor"))

  # Row parallel
  assert node_dict["o_proj"].sharding == PartitionSpec(axes=("tensor", None))
  assert node_dict["down_proj"].sharding == PartitionSpec(axes=("tensor", None))
  assert node_dict["embed_tokens"].sharding == PartitionSpec(axes=("tensor", None))
  assert node_dict["my_embedding_layer"].sharding == PartitionSpec(axes=("tensor", None))

  # Fallback
  assert node_dict["fc1"].sharding == PartitionSpec(axes=("data", None))
  assert node_dict["conv1"].sharding == PartitionSpec(axes=("data", None))
  assert node_dict["conv3"].sharding == PartitionSpec(axes=("data", None))

  # Ignored
  assert node_dict["relu"].sharding is None


def test_estimate_tensor_memory_and_graph() -> None:
  """Test analytical memory estimation on tensors and graphs."""
  from unittest.mock import patch
  from ml_switcheroo.core.compiler.sharding import estimate_tensor_memory

  # Normal computation
  bytes_f32 = estimate_tensor_memory([10, 20], dtype="float32")
  assert bytes_f32 == 800.0

  # Fallback branch
  with patch("ml_ecosystem_snapshots.compliance.estimate_tensor_memory_bytes", side_effect=Exception("Err")):
    fallback_64 = estimate_tensor_memory([10, 20], dtype="float64")
    assert fallback_64 == 1600.0
    fallback_16 = estimate_tensor_memory([10, 20], dtype="float16")
    assert fallback_16 == 400.0
    fallback_8 = estimate_tensor_memory([10, 20], dtype="int8")
    assert fallback_8 == 200.0
    fallback_empty = estimate_tensor_memory([], dtype="float32")
    assert fallback_empty == 0.0

  pass_ = ShardingInferencePass()
  node = LogicalNode(id="n1", op_type="Linear", attributes={"shape": [4, 8], "dtype": "float32"})
  node_no_shape = LogicalNode(id="n2", op_type="Relu", attributes={})
  graph = LogicalGraph(nodes={"n1": node, "n2": node_no_shape}, edges=[])
  # 128 bytes param + 128 bytes grad + 0 act = 256 + 10% workspace = 281.6
  assert pass_.estimate_graph_memory_bytes(graph, include_gradients=False, workspace_overhead_ratio=0.0) == 128.0


def test_estimate_tensor_memory_symbolic_and_types() -> None:
  """Verifies symbolic dimensions and edge-case datatypes in memory estimation."""
  from unittest.mock import patch

  # Symbolic dimension resolution
  bytes_sym = estimate_tensor_memory(["B", 128], dtype="float32", symbolic_map={"B": 4})
  assert bytes_sym == 2048.0

  # Symbolic with None, string integer, negative string fallback, unknown symbol fallback
  bytes_mixed = estimate_tensor_memory([None, "64", "-1", "unknown", object()], dtype="float32")
  assert bytes_mixed > 0.0

  # Test fallback branch datatypes: bfloat16, 4-bit
  with patch("ml_ecosystem_snapshots.compliance.estimate_tensor_memory_bytes", side_effect=Exception("Err")):
    bytes_bf16 = estimate_tensor_memory([10, 20], dtype="bfloat16")
    assert bytes_bf16 == 400.0
    bytes_int4 = estimate_tensor_memory([10, 20], dtype="int4")
    assert bytes_int4 == 200.0


def test_estimate_graph_memory_extended() -> None:
  """Verifies graph memory estimation with activations, gradients, and workspace overhead."""
  pass_ = ShardingInferencePass()
  node_param = LogicalNode(id="w1", op_type="Linear", attributes={"shape": [10, 10], "dtype": "float32"})
  node_act = LogicalNode(id="act1", op_type="Relu", attributes={"shape": [10, 10], "dtype": "float32"})
  graph = LogicalGraph(nodes={"w1": node_param, "act1": node_act}, edges=[])

  # Param: 400, Grad: 400, Act: 400 => Base: 1200 + 10% workspace = 1320
  total = pass_.estimate_graph_memory_bytes(
    graph,
    include_activations=True,
    include_gradients=True,
    workspace_overhead_ratio=0.1,
  )
  assert total == 1320.0


def test_infer_device_placement() -> None:
  """Verifies threshold-based layer splitting across devices."""
  pass_ = ShardingInferencePass()
  # Each node has 400 bytes, plus one node without shape
  nodes = {
    f"node_{i}": LogicalNode(id=f"node_{i}", op_type="Linear", attributes={"shape": [10, 10], "dtype": "float32"})
    for i in range(5)
  }
  nodes["no_shape"] = LogicalNode(id="no_shape", op_type="Identity", attributes={})
  graph = LogicalGraph(nodes=nodes, edges=[])

  # Capacity = 700 bytes (each node is 400 bytes, so 1 node per device before exceeding)
  placements = pass_.infer_device_placement(
    graph,
    device_capacity_bytes=700.0,
    num_devices=3,
  )
  assert placements["node_0"] == 0
  assert placements["node_1"] == 1
  assert placements["node_2"] == 2
  assert placements["node_3"] == 2
  assert placements["node_4"] == 2
  assert graph.nodes["node_0"].device == "cuda:0"
  assert graph.nodes["node_4"].device == "cuda:2"


def test_pytorch_dtensor_sharding_pass() -> None:
  """Verifies PyTorch DTensor / FSDP2 sharding annotation placement."""
  dtensor_pass = PyTorchDTensorShardingPass(world_size=4)
  nodes = {
    "q_proj": LogicalNode(id="q_proj", op_type="Linear"),
    "o_proj": LogicalNode(id="o_proj", op_type="Linear"),
    "fc_layer": LogicalNode(id="fc_layer", op_type="Linear"),
    "relu": LogicalNode(id="relu", op_type="ReLU"),
  }
  graph = LogicalGraph(nodes=nodes, edges=[])
  updated_graph = dtensor_pass.apply(graph)

  assert updated_graph.nodes["q_proj"].attributes["dtensor_placement"] == "Shard(dim=1)"
  assert updated_graph.nodes["q_proj"].attributes["dtensor_world_size"] == 4
  assert updated_graph.nodes["o_proj"].attributes["dtensor_placement"] == "Shard(dim=0)"
  assert updated_graph.nodes["fc_layer"].attributes["dtensor_placement"] == "FSDP(dim=0)"
  assert "dtensor_placement" not in updated_graph.nodes["relu"].attributes


def test_mlx_distributed_sharding_pass() -> None:
  """Verifies Apple MLX distributed array annotations and warning diagnostics."""
  mlx_pass = MlxDistributedShardingPass(num_nodes=2)
  nodes = {
    "embed": LogicalNode(id="embed", op_type="Embedding"),
    "relu": LogicalNode(id="relu", op_type="ReLU"),
  }
  graph = LogicalGraph(nodes=nodes, edges=[])
  updated_graph = mlx_pass.apply(graph)

  assert updated_graph.nodes["embed"].attributes["mlx_distributed"] is True
  assert updated_graph.nodes["embed"].attributes["mlx_nodes"] == 2
  assert len(mlx_pass.diagnostics) == 1
  assert "MLX distributed sharding active" in mlx_pass.diagnostics[0]
  assert "mlx_distributed" not in updated_graph.nodes["relu"].attributes
