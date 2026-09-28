"""Test suite for the compiler sharding pass."""

from typing import Dict, List

from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalMesh, LogicalNode, PartitionSpec
from ml_switcheroo.core.compiler.sharding import ShardingInferencePass


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
  assert pass_.estimate_graph_memory_bytes(graph) == 128.0
