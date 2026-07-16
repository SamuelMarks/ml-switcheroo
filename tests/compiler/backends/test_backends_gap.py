"""Auto-generated doc."""

import libcst as cst
from ml_switcheroo.core.compiler.backends.python import PythonBackend, ClassBodyReplacer
from ml_switcheroo.core.compiler.backends.python_snippet import PythonSnippetEmitter
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, PartitionSpec, LogicalEdge


def test_python_backend_class_body_replacer():
  """Auto-generated doc."""
  # Test ClassBodyReplacer for one-line class "class A: pass"
  code = "class A: pass"
  tree = cst.parse_module(code)

  init_func = cst.FunctionDef(
    name=cst.Name("__init__"),
    params=cst.Parameters(),
    body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
  )
  forward_func = cst.FunctionDef(
    name=cst.Name("forward"),
    params=cst.Parameters(),
    body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
  )

  replacer = ClassBodyReplacer("A", init_func, forward_func)
  tree.visit(replacer)
  assert replacer.found

  # Test ClassBodyReplacer for not found class
  replacer2 = ClassBodyReplacer("B", init_func, forward_func)
  tree.visit(replacer2)
  assert not replacer2.found


def test_python_backend_imports():
  """Auto-generated doc."""
  # coverage for different frameworks
  backend_keras = PythonBackend(framework="keras")
  assert "keras.Model" in backend_keras.compile(LogicalGraph())

  backend_mlx = PythonBackend(framework="mlx")
  graph = LogicalGraph()
  graph.nodes = [LogicalNode(id="x", kind="Input"), LogicalNode(id="out", kind="Output")]
  code = backend_mlx.compile(graph)
  assert "import mlx.core as mx" in code

  backend_tf = PythonBackend(framework="tensorflow")
  assert "import tensorflow as tf" in backend_tf.compile(graph)


def test_python_backend_build_init():
  """Auto-generated doc."""
  # coverage for empty init
  backend = PythonBackend(framework="mlx")
  graph = LogicalGraph()
  graph.nodes = [LogicalNode(id="x", kind="Input")]
  code = backend.compile(graph)
  assert "def __init__(self):" in code


def test_python_backend_build_forward():
  """Auto-generated doc."""
  # coverage for sharding with tuples
  backend = PythonBackend(framework="jax")
  graph = LogicalGraph()
  graph.nodes = [
    LogicalNode(id="x", kind="Input"),
    LogicalNode(id="conv", kind="Conv2d", sharding=PartitionSpec(axes=(("a", "b"), "c"))),
    LogicalNode(id="out", kind="Output"),
  ]
  code = backend.compile(graph)
  assert "jax.sharding.PartitionSpec(('a', 'b'), 'c')" in code

  # coverage for sharding formatting exceptions TF
  backend_tf = PythonBackend(framework="tensorflow")
  graph.nodes[1].sharding = PartitionSpec(axes=(None, "a", 1))  # to hit the `else` case in _format_partition_spec_tf
  code_tf = backend_tf.compile(graph)
  assert "[None, 'a', '*']" in code_tf


def test_python_snippet_emitter():
  """Auto-generated doc."""
  emitter = PythonSnippetEmitter(framework="torch")

  # Test non-stateful layer emit_init
  node_input = LogicalNode("x", "Input")
  stmt1 = emitter.emit_init(node_input)
  assert "pass" in cst.Module(body=[stmt1]).code

  # Test stateful emit_call input pass through
  node_pass = LogicalNode("x", "Input")
  stmt2 = emitter.emit_call(node_pass, ["y"], "x")
  assert "x = y" in cst.Module(body=[stmt2]).code

  stmt3 = emitter.emit_call(node_pass, ["x"], "x")
  assert "pass" in cst.Module(body=[stmt3]).code

  # Test expression syntax error coverage
  node_bad = LogicalNode("bad", "1bad_name")
  expr = emitter.emit_expression(node_bad, [])
  assert "None" == cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=expr)])]).code.strip()

  # Test _is_stateful_layer various branches
  assert not emitter._is_stateful_layer(LogicalNode("func_1", "func_x"))
  assert not emitter._is_stateful_layer(LogicalNode("f", "functional.relu"))
  assert not emitter._is_stateful_layer(LogicalNode("o", "ops.add"))
  assert emitter._is_stateful_layer(LogicalNode("Conv", "Conv2d"))
  assert not emitter._is_stateful_layer(LogicalNode("lower", "relu"))

  # Test _resolve_layer_name
  assert emitter._resolve_api_name("func_relu") == "torch.relu"
  assert emitter._resolve_api_name("nn.Module") == "nn.Module"

  # Test jax api name
  emitter_jax = PythonSnippetEmitter(framework="jax")
  assert emitter_jax._resolve_api_name("Linear") == "nnx.Linear"
  assert emitter_jax._resolve_api_name("relu") == "jnp.relu"

  # Test keras api name
  emitter_keras = PythonSnippetEmitter(framework="keras")
  assert emitter_keras._resolve_api_name("Linear") == "keras.layers.Linear"
  assert emitter_keras._resolve_api_name("relu") == "keras.ops.relu"

  # Test functional call with metadata
  node_func = LogicalNode("f", "func_relu", {"arg_0": "True", "dim": 1})
  expr_func = emitter.emit_expression(node_func, ["x"])
  assert (
    "relu(x, True, dim=1)" in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=expr_func)])]).code.strip()
  )


def test_python_backend_class_body_replacer_methods():
  """Auto-generated doc."""
  code = "class A:\n    def __init__(self):\n        pass\n    def forward(self, x):\n        pass\n    def other(self):\n        pass"
  tree = cst.parse_module(code)

  init_func = cst.FunctionDef(
    name=cst.Name("__init__"),
    params=cst.Parameters(),
    body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
  )
  forward_func = cst.FunctionDef(
    name=cst.Name("forward"),
    params=cst.Parameters(),
    body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
  )

  replacer = ClassBodyReplacer("A", init_func, forward_func)
  new_tree = tree.visit(replacer)
  assert replacer.found

  # check if 'other' method is still present
  assert "def other(self):" in new_tree.code

  # multiple of same name
  code2 = "class A:\n    def __init__(self):\n        pass\n    def __init__(self):\n        pass"
  tree2 = cst.parse_module(code2)
  replacer2 = ClassBodyReplacer("A", init_func, forward_func)
  tree2.visit(replacer2)


def test_python_backend_functional_nodes():
  """Auto-generated doc."""
  backend = PythonBackend(framework="torch")
  graph = LogicalGraph()
  graph.nodes = [
    LogicalNode(id="x", kind="Input"),
    LogicalNode(id="relu", kind="torch.relu", metadata={"arg_0": "True", "inplace": "True"}),
    LogicalNode(id="out", kind="Output"),
  ]
  graph.edges = [LogicalEdge("x", "relu"), LogicalEdge("relu", "out")]
  code = backend.compile(graph)
  assert "relu(x, True, inplace=True)" in code


def test_python_backend_is_stateful_layer():
  """Auto-generated doc."""
  backend = PythonBackend(framework="torch")
  assert not backend._is_stateful_layer(LogicalNode("o", "torch.relu"))


def test_python_backend_generate_layer_init_mlx():
  """Auto-generated doc."""
  backend = PythonBackend(framework="mlx")
  graph = LogicalGraph()
  graph.nodes = [
    LogicalNode(id="x", kind="Input"),
    LogicalNode(id="fc", kind="Linear"),
    LogicalNode(id="out", kind="Output"),
  ]
  graph.edges = [LogicalEdge("x", "fc"), LogicalEdge("fc", "out")]
  code = backend.compile(graph)
  assert "self.fc = nn.Linear()" in code


def test_python_backend_format_args():
  """Auto-generated doc."""
  backend = PythonBackend()
  assert backend._format_args_from_metadata({"arg_0": "val", "k": "v"}) == "val, k=v"


def test_python_snippet_emitter_gap():
  """Auto-generated doc."""
  emitter = PythonSnippetEmitter(framework="mlx")
  assert emitter._resolve_api_name("relu") == "relu"
  assert emitter._format_args_from_metadata(None) == ""


def test_python_backend_unknown_fw_import():
  """Auto-generated doc."""
  from ml_switcheroo.core.compiler.backends.python import PythonBackend

  backend = PythonBackend(framework="unknown_fw")
  assert backend._generate_imports() == []


def test_python_backend_keras_layer_kind():
  """Auto-generated doc."""
  from ml_switcheroo.core.compiler.backends.python import PythonBackend
  from ml_switcheroo.core.graph import LogicalNode

  backend = PythonBackend(framework="keras")
  node = LogicalNode("test", "Dense")
  node = LogicalNode("test", "Dense")
  assert backend._generate_layer_init(node).body[0].value.func.value.value.value == "keras"
  assert backend._generate_layer_init(node).body[0].value.func.value.attr.value == "layers"
  assert backend._generate_layer_init(node).body[0].value.func.attr.value == "Dense"

  backend_torch = PythonBackend(framework="torch")
  assert backend_torch._generate_layer_init(node).body[0].value.func.value.value == "nn"

  backend_jax = PythonBackend(framework="jax")
  assert backend_jax._generate_layer_init(node).body[0].value.func.value.value == "nnx"

  backend_mlx = PythonBackend(framework="mlx")
  assert backend_mlx._generate_layer_init(node).body[0].value.func.value.value == "nn"


def test_rdna_macros_linear():
  """Auto-generated doc."""
  from ml_switcheroo.core.compiler.backends.rdna.macros import expand_linear
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RegisterAllocator

  allocator = RegisterAllocator()
  nodes = expand_linear(allocator, "test_lin", {"in_features": 64, "bias": True})
  assert len(nodes) > 10

  # 69, 81, 84 in synthesizer: invalid types/missing args
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaBackend
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode

  backend = RdnaBackend()

  # 69: get_vector_register without fallback
  pass

  # 172-193: _synthesize_unmapped_op
  # 201-216: _synthesize_layer
  node_unmap = LogicalNode("unmap", "this_op_does_not_exist_in_the_universe", {"arg_1": "v1", "arg_2": "v2"})
  node_layer = LogicalNode("lin", "Linear", {"in_features": 64})

  # Needs to process graph with these nodes
  graph = LogicalGraph(nodes=[node_unmap, node_layer], edges=[])
  code = backend.compile(graph)
  assert "Linear" in code
  assert "Unmapped Op:" in code
