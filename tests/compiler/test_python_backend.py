"""Tests for Python Backend and Python Snippet Emitter."""

import libcst as cst

from ml_switcheroo.core.compiler.backends.python import PythonBackend
from ml_switcheroo.core.compiler.backends.python_snippet import PythonSnippetEmitter
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode


class MockSharding:
  """Mock sharding class."""

  def __init__(self, axes):
    """Initialize mock sharding."""
    self.axes = axes


def test_python_snippet_emitter_init():
  """Test snippet emitter init."""
  emitter = PythonSnippetEmitter(framework="jax")
  assert emitter.framework == "jax"


def test_python_snippet_emit_init_stateless():
  """Test emit_init with stateless node."""
  emitter = PythonSnippetEmitter()
  node = LogicalNode(id="n1", kind="func_add")
  stmt = emitter.emit_init(node)
  assert isinstance(stmt, cst.SimpleStatementLine)
  assert isinstance(stmt.body[0], cst.Pass)


def test_python_snippet_emit_init_stateful():
  """Test emit_init with stateful node."""
  emitter = PythonSnippetEmitter(framework="torch")
  node = LogicalNode(id="l1", kind="Linear", metadata={"arg_1": "10", "arg_2": "20"})
  stmt = emitter.emit_init(node)

  code = cst.Module(body=[stmt]).code
  assert "self.l1 = nn.Linear(10, 20)" in code


def test_python_snippet_emit_init_flax():
  """Test emit_init with stateful node for flax (rngs inject)."""
  emitter = PythonSnippetEmitter(framework="flax_nnx")
  node = LogicalNode(id="l1", kind="Linear")
  stmt = emitter.emit_init(node)

  code = cst.Module(body=[stmt]).code
  assert "rngs = rngs" in code


def test_python_snippet_emit_call_input():
  """Test emit_call with input node."""
  emitter = PythonSnippetEmitter()
  node = LogicalNode(id="in", kind="Input")
  stmt = emitter.emit_call(node, ["x_in"], "x_out")
  code = cst.Module(body=[stmt]).code
  assert "x_out = x_in" in code

  stmt_same = emitter.emit_call(node, ["x_in"], "x_in")
  code_same = cst.Module(body=[stmt_same]).code
  assert "pass" in code_same


def test_python_snippet_emit_call_expr():
  """Test emit_call with expression."""
  emitter = PythonSnippetEmitter()
  node = LogicalNode(id="add", kind="add", metadata={"alpha": "1.0"})
  stmt = emitter.emit_call(node, ["x", "y"], "out")
  code = cst.Module(body=[stmt]).code
  assert "out = torch.add(x, y, alpha = 1.0)" in code


def test_python_snippet_emit_expression_error():
  """Test emit_expression on parse error."""
  emitter = PythonSnippetEmitter()
  # Invalid python expression string in metadata to cause parser error
  node = LogicalNode(id="bad", kind="bad", metadata={"key": "def class *"})
  expr = emitter.emit_expression(node, [])
  assert isinstance(expr, cst.Name)
  assert expr.value == "None"


def test_python_snippet_resolve_api_name():
  """Test framework specific resolving."""
  _ = LogicalNode(id="1", kind="Linear")
  emitter_torch = PythonSnippetEmitter("torch")
  assert emitter_torch._resolve_api_name("Linear") == "nn.Linear"
  assert emitter_torch._resolve_api_name("add") == "torch.add"
  assert emitter_torch._resolve_api_name("a.b.c") == "a.b.c"

  emitter_jax = PythonSnippetEmitter("jax")
  assert emitter_jax._resolve_api_name("Linear") == "nnx.Linear"
  assert emitter_jax._resolve_api_name("add") == "jnp.add"

  emitter_keras = PythonSnippetEmitter("keras")
  assert emitter_keras._resolve_api_name("Linear") == "keras.layers.Linear"
  assert emitter_keras._resolve_api_name("add") == "keras.ops.add"


def test_python_backend_init():
  """Test init."""
  backend = PythonBackend(framework="torch")
  assert backend.framework == "torch"


def test_python_backend_compile():
  """Test compilation for torch."""
  backend = PythonBackend(framework="torch")
  graph = LogicalGraph(name="MyModel")
  graph.nodes.append(LogicalNode(id="in", kind="Input", metadata={"name": "x"}))
  graph.nodes.append(LogicalNode(id="l1", kind="Linear", metadata={"arg_1": "10"}))
  graph.nodes.append(LogicalNode(id="out", kind="Output"))

  code = backend.compile(graph)
  assert "class MyModel(nn.Module):" in code
  assert "def __init__(self):" in code
  assert "self.l1 = nn.Linear(10)" in code
  assert "def forward(self, x):" in code
  assert "x = self.l1(x)" in code
  assert "return x" in code


def test_python_backend_compile_no_name():
  """Test default name."""
  backend = PythonBackend(framework="torch")
  graph = LogicalGraph()
  graph.nodes.append(LogicalNode(id="out", kind="Output"))
  code = backend.compile(graph)
  assert "class Model(nn.Module):" in code


def test_python_backend_generate_replacing():
  """Test generate with class body replacer."""
  backend = PythonBackend(framework="torch")
  graph = LogicalGraph(name="MyModel")
  graph.nodes.append(LogicalNode(id="l1", kind="Linear"))

  source = "class MyModel:\n  pass"
  tree = cst.parse_module(source)

  code = backend.generate(graph, class_name="MyModel", original_tree=tree)
  assert "self.l1" in code
  assert "def forward" in code

  source2 = "class MyModel:\n  def forward(self): pass"
  tree2 = cst.parse_module(source2)
  code2 = backend.generate(graph, class_name="MyModel", original_tree=tree2)
  assert "self.l1" in code2

  source3 = "class MyModel(nn.Module): pass\n"
  # SimpleStatementSuite one-liner
  tree3 = cst.parse_module(source3)
  code3 = backend.generate(graph, class_name="MyModel", original_tree=tree3)
  assert "self.l1" in code3


def test_python_backend_framework_imports():
  """Test imports generation."""
  assert "import torch" in cst.Module(body=PythonBackend("torch")._generate_imports()).code
  assert "jax.numpy" in cst.Module(body=PythonBackend("jax")._generate_imports()).code
  assert "mlx.core" in cst.Module(body=PythonBackend("mlx")._generate_imports()).code
  assert "keras" in cst.Module(body=PythonBackend("keras")._generate_imports()).code
  assert "praxis" in cst.Module(body=PythonBackend("paxml")._generate_imports()).code
  assert PythonBackend("unknown")._generate_imports() == []


def test_python_backend_build_init_paxml():
  """Test init for paxml."""
  backend = PythonBackend("paxml")
  node = LogicalNode(id="l1", kind="Linear", metadata={"arg_0": "10"})
  init_def = backend._build_init([node])
  code = cst.Module(body=[init_def]).code
  assert "def setup(self):" in code
  assert "self.create_child('l1', pl.Linear(10))" in code


def test_python_backend_build_init_flax():
  """Test init for flax."""
  backend = PythonBackend("flax_nnx")
  node = LogicalNode(id="l1", kind="Linear")
  init_def = backend._build_init([node])
  code = cst.Module(body=[init_def]).code
  assert "rngs: nnx.Rngs" in code
  assert "rngs=rngs" in code


def test_python_backend_build_forward_sharding():
  """Test forward with sharding constraints."""
  node = LogicalNode(id="l1", kind="Linear", metadata={"arg_0": "10"})
  node.sharding = MockSharding(["batch", None])

  # torch
  backend_torch = PythonBackend("torch")
  fwd_torch = backend_torch._build_forward([node])
  code_torch = cst.Module(body=[fwd_torch]).code
  assert "distribute_tensor(x, self.mesh, [Shard(0), Replicate()])" in code_torch

  # jax
  backend_jax = PythonBackend("jax")
  fwd_jax = backend_jax._build_forward([node])
  code_jax = cst.Module(body=[fwd_jax]).code
  assert "jax.lax.with_sharding_constraint(x, jax.sharding.PartitionSpec('batch', None))" in code_jax

  # keras
  backend_keras = PythonBackend("keras")
  fwd_keras = backend_keras._build_forward([node])
  code_keras = cst.Module(body=[fwd_keras]).code
  assert "keras.distribution.layout(['batch', None])" in code_keras

  # mlx
  backend_mlx = PythonBackend("mlx")
  fwd_mlx = backend_mlx._build_forward([node])
  code_mlx = cst.Module(body=[fwd_mlx]).code
  assert "mx.distributed.shard" in code_mlx

  # tuple in jax
  node2 = LogicalNode(id="l2", kind="Linear")
  node2.sharding = MockSharding([("a", "b")])
  fwd_jax2 = backend_jax._build_forward([node2])
  code_jax2 = cst.Module(body=[fwd_jax2]).code
  assert "('a', 'b')" in code_jax2


def test_python_backend_build_layer_init_mlx_specials():
  """Test layer init mapping in mlx."""
  backend = PythonBackend("mlx")
  node = LogicalNode(id="s", kind="SwiGLU")
  stmt = backend._generate_layer_init(node)
  assert "silu" in cst.Module(body=[stmt]).code

  node2 = LogicalNode(id="r", kind="RoPE")
  stmt2 = backend._generate_layer_init(node2)
  assert "nn.RoPE" in cst.Module(body=[stmt2]).code

  node3 = LogicalNode(id="v", kind="VisionPatchEmbedding")
  stmt3 = backend._generate_layer_init(node3)
  assert "nn.Conv2d" in cst.Module(body=[stmt3]).code
