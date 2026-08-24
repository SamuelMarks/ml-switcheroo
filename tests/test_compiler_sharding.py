"""Test suite for the compiler sharding pass."""

from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalMesh, PartitionSpec
from ml_switcheroo.core.compiler.sharding import ShardingInferencePass


def test_sharding_inference_pass_default_mesh():
  """Test ShardingInferencePass with default mesh."""
  pass_ = ShardingInferencePass()
  assert pass_.mesh.shape == {"data": 1, "tensor": 1}


def test_sharding_inference_pass_custom_mesh():
  """Test ShardingInferencePass with custom mesh."""
  mesh = LogicalMesh(shape={"data": 2, "tensor": 4})
  pass_ = ShardingInferencePass(mesh=mesh)
  assert pass_.mesh.shape == {"data": 2, "tensor": 4}


def test_sharding_inference_pass_apply():
  """Test ShardingInferencePass apply method on various nodes."""
  nodes = [
    # Column Parallel matches
    LogicalNode(id="q_proj", kind="Linear"),
    LogicalNode(id="k_proj_layer", kind="Linear"),
    LogicalNode(id="V_PROJ", kind="Linear"),
    LogicalNode(id="gate_proj", kind="Linear"),
    LogicalNode(id="up_proj", kind="Linear"),
    # Row Parallel matches
    LogicalNode(id="o_proj", kind="Linear"),
    LogicalNode(id="down_proj", kind="Linear"),
    LogicalNode(id="embed_tokens", kind="Embedding"),
    # Fallback FSDP matches
    LogicalNode(id="fc1", kind="Linear"),
    LogicalNode(id="conv1", kind="Conv2d"),
    LogicalNode(id="conv3", kind="Conv3d"),
    LogicalNode(id="my_embedding_layer", kind="Embedding"),
    # Ignored node
    LogicalNode(id="relu", kind="ReLU"),
  ]
  graph = LogicalGraph(nodes=nodes, edges=[])
  pass_ = ShardingInferencePass()
  new_graph = pass_.apply(graph)

  # Check if mesh is attached
  assert new_graph.mesh == pass_.mesh

  node_dict = {n.id: n for n in new_graph.nodes}

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
