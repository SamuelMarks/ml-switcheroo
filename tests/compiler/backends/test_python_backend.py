"""
Tests for Python Backend.
"""

import pytest
import ast
import libcst as cst
from ml_switcheroo.compiler.backends.python import PythonBackend
from ml_switcheroo.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge


@pytest.fixture
def backend() -> PythonBackend:
  """Fixture for backend."""
  return PythonBackend()


def validate_python(code: str) -> None:
  """Validate python syntax."""
  try:
    ast.parse(code)
  except SyntaxError as e:
    pytest.fail(f"Generated Invalid Python:\n{e}\n\nCode:\n{code}")


def test_compile_interface_implementation(backend: PythonBackend) -> None:
  """Test compile interface."""
  g = LogicalGraph()
  res = backend.compile(g)
  assert isinstance(res, str)
  # LogicalGraph defaults name to "Model", so class will be "class Model" unless overwritten
  # backend.compile uses g.name if present.
  assert "class Model" in res


def test_synthesize_torch_chain(backend: PythonBackend) -> None:
  """Test torch code gen."""
  g = LogicalGraph(
    nodes=[
      LogicalNode("x", "Input"),
      LogicalNode("conv1", "Conv2d"),
      LogicalNode("output", "Output"),
    ],
    edges=[LogicalEdge("x", "conv1"), LogicalEdge("conv1", "output")],
  )
  code = backend.generate(g, "SimpleNet")
  validate_python(code)
  assert "import torch" in code
  assert "class SimpleNet(nn.Module):" in code
  assert "self.conv1 = nn.Conv2d()" in code
  assert "return x" in code


def test_synthesize_flax_chain() -> None:
  """Test flax nnx code gen."""
  backend = PythonBackend(framework="flax_nnx")
  g = LogicalGraph(
    nodes=[
      LogicalNode("x", "Input"),
      LogicalNode("fc", "Linear", {"out": "10"}),
    ]
  )
  code = backend.generate(g, "FlaxNet")
  validate_python(code)
  assert "class FlaxNet(nnx.Module):" in code
  assert "self.fc = nnx.Linear(out=10, rngs=rngs)" in code


def test_context_preservation(backend: PythonBackend) -> None:
  """Test class context prep."""
  orig = "class MyNet(nn.Module): pass"
  tree = cst.parse_module(orig)
  g = LogicalGraph(nodes=[LogicalNode("x", "Input")])
  code = backend.generate(g, class_name="MyNet", original_tree=tree)
  validate_python(code)
  assert "class MyNet" in code


def test_python_backend_sharding():
  """Verify sharding constraint generation in jax/flax framework."""
  from ml_switcheroo.compiler.ir import PartitionSpec

  graph = LogicalGraph(name="ShardedNet")
  graph.nodes = [
    LogicalNode(id="x", kind="Input"),
    LogicalNode(id="fc1", kind="Linear", sharding=PartitionSpec(axes=("data", ("model", "tensor")))),
    LogicalNode(id="out", kind="Output"),
  ]
  graph.edges = [LogicalEdge("x", "fc1"), LogicalEdge("fc1", "out")]

  backend = PythonBackend(framework="flax_nnx")
  code = backend.compile(graph)

  assert "jax.lax.with_sharding_constraint" in code
  assert "jax.sharding.PartitionSpec('data', ('model', 'tensor'))" in code


def test_python_backend_sharding_none():
  """Verify None axes format correctly."""
  from ml_switcheroo.compiler.ir import PartitionSpec

  graph = LogicalGraph(name="ShardedNet")
  graph.nodes = [
    LogicalNode(id="x", kind="Input"),
    LogicalNode(id="fc1", kind="Linear", sharding=PartitionSpec(axes=(None, "tensor"))),
    LogicalNode(id="out", kind="Output"),
  ]
  graph.edges = [LogicalEdge("x", "fc1"), LogicalEdge("fc1", "out")]

  backend = PythonBackend(framework="jax")
  code = backend.compile(graph)

  assert "jax.lax.with_sharding_constraint" in code
  assert "jax.sharding.PartitionSpec(None, 'tensor')" in code


def test_python_backend_sharding_torch():
  """Verify sharding constraint generation in PyTorch DTensor format."""
  from ml_switcheroo.compiler.ir import PartitionSpec

  graph = LogicalGraph(name="ShardedNet")
  graph.nodes = [
    LogicalNode(id="x", kind="Input"),
    LogicalNode(id="fc1", kind="Linear", sharding=PartitionSpec(axes=("data", None))),
    LogicalNode(id="out", kind="Output"),
  ]
  graph.edges = [LogicalEdge("x", "fc1"), LogicalEdge("fc1", "out")]

  backend = PythonBackend(framework="torch")
  code = backend.compile(graph)

  assert "distribute_tensor" in code
  assert "Shard(0)" in code
  assert "Replicate()" in code


def test_python_backend_sharding_tf_mlx():
  """Verify sharding constraint generation in TF and MLX."""
  from ml_switcheroo.compiler.ir import PartitionSpec

  graph = LogicalGraph(name="ShardedNet")
  graph.nodes = [
    LogicalNode(id="x", kind="Input"),
    LogicalNode(id="fc1", kind="Linear", sharding=PartitionSpec(axes=("data", ("tensor", "model")))),
    LogicalNode(id="out", kind="Output"),
  ]
  graph.edges = [LogicalEdge("x", "fc1"), LogicalEdge("fc1", "out")]

  backend = PythonBackend(framework="tensorflow")
  code_tf = backend.compile(graph)
  assert "keras.distribution.layout" in code_tf

  backend = PythonBackend(framework="mlx")
  code_mlx = backend.compile(graph)
  assert "mx.distributed.shard" in code_mlx


def test_python_backend_primitive_mapping_mlx():
  """Verify advanced nodes fall back correctly in MLX."""
  from ml_switcheroo.compiler.ir import LogicalGraph, LogicalNode
  from ml_switcheroo.compiler.backends.python import PythonBackend

  graph = LogicalGraph()
  graph.nodes = [
    LogicalNode(id="rope", kind="RoPE"),
    LogicalNode(id="vision", kind="VisionPatchEmbedding"),
    LogicalNode(id="swiglu", kind="SwiGLU"),
  ]

  backend = PythonBackend(framework="mlx")
  code = backend.compile(graph)

  assert "self.rope = nn.RoPE()" in code
  assert "self.vision = nn.Conv2d()" in code
  assert "self.swiglu = nn.silu()" in code
