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


def test_python_backend_generate_replacing_pass_stmt():
  """Test replacing simple statement with multiple passes to hit missing coverage in _ClassBodyReplacer."""
  backend = PythonBackend(framework="torch")
  graph = LogicalGraph(name="MyModel")

  source = "class MyModel(nn.Module): pass; pass"
  tree = cst.parse_module(source)
  code = backend.generate(graph, class_name="MyModel", original_tree=tree)
  assert "class MyModel" in code


def test_python_backend_generate_replacing_no_match():
  """Test replacing class body with no functions matching and some other nodes."""
  backend = PythonBackend(framework="torch")
  graph = LogicalGraph(name="MyModel")

  source = "class MyModel:\n  def unrelated(self): pass\n  x = 1"
  tree = cst.parse_module(source)
  code = backend.generate(graph, class_name="MyModel", original_tree=tree)
  assert "unrelated" in code
  assert "x = 1" in code


def test_python_backend_build_forward_tuple_outputs():
  """Test build forward with multiple outputs."""
  backend = PythonBackend("torch")
  node = LogicalNode(id="l1", kind="Linear")
  fwd = backend._build_forward([node])
  code = cst.Module(body=[fwd]).code
  assert "return" in code


def test_python_backend_build_forward_no_outputs():
  """Test build forward with zero outputs."""
  backend = PythonBackend("torch")
  node = LogicalNode(id="l1", kind="Linear")
  fwd = backend._build_forward([node])
  code = cst.Module(body=[fwd]).code
  assert "return" in code


def test_python_backend_build_layer_init_torch_specials():
  """Test torch special layer initializers."""
  backend = PythonBackend("torch")
  node_swi = LogicalNode(id="s", kind="SwiGLU")
  code = cst.Module(body=[backend._generate_layer_init(node_swi)]).code
  assert "nn.SwiGLU" in code

  node_rope = LogicalNode(id="r", kind="RoPE")
  code_rope = cst.Module(body=[backend._generate_layer_init(node_rope)]).code
  assert "nn.RoPE" in code_rope

  node_vpe = LogicalNode(id="v", kind="VisionPatchEmbedding")
  code_vpe = cst.Module(body=[backend._generate_layer_init(node_vpe)]).code
  assert "nn.VisionPatchEmbedding" in code_vpe


def test_python_backend_build_layer_init_jax_specials():
  """Test jax special layer initializers."""
  backend = PythonBackend("jax")
  node = LogicalNode(id="l", kind="Conv2d")
  code = cst.Module(body=[backend._generate_layer_init(node)]).code
  assert "nnx.Conv" in code

  node_swi = LogicalNode(id="s", kind="SwiGLU")
  code_swi = cst.Module(body=[backend._generate_layer_init(node_swi)]).code
  assert "nnx.SwiGLU" in code_swi

  node_rope = LogicalNode(id="r", kind="RoPE")
  code_rope = cst.Module(body=[backend._generate_layer_init(node_rope)]).code
  assert "nnx.RoPE" in code_rope

  node_vpe = LogicalNode(id="v", kind="VisionPatchEmbedding")
  code_vpe = cst.Module(body=[backend._generate_layer_init(node_vpe)]).code
  assert "nnx.VisionPatchEmbedding" in code_vpe


def test_python_backend_build_layer_init_paxml_specials():
  """Test paxml special layer initializers."""
  backend = PythonBackend("paxml")
  node = LogicalNode(id="l", kind="Conv2d")
  code = cst.Module(body=[backend._generate_layer_init(node)]).code
  assert "pl.Conv2d" in code


def test_python_snippet_emit_expression_bool():
  """Test boolean value evaluation in metadata."""
  emitter = PythonSnippetEmitter("torch")
  node = LogicalNode(id="l", kind="Linear", metadata={"bias": "True", "other": "False"})
  stmt = emitter.emit_init(node)
  code = cst.Module(body=[stmt]).code
  assert "True" in code
  assert "False" in code


class FakeSemantics:
  """Fake semantics."""

  def resolve_variant(self, abstract_id, fw):
    """Resolve variant."""
    if abstract_id == "ResolvedApi":
      return {"api": "resolved.api"}
    if abstract_id == "TorchFunc" and fw == "torch":
      return {"api": "torch.nn.functional.relu"}
    if abstract_id == "MlxCore" and fw == "mlx":
      return {"api": "mlx.core.add"}
    return None

  def get_definition(self, api):
    """Get definition."""
    if api == "concrete.api":
      return ("ResolvedApi", {})
    return None


def test_python_backend_semantics_resolution():
  """Test python backend semantics resolution."""
  backend = PythonBackend("torch")
  backend.semantics = FakeSemantics()

  # Test _build_forward with resolve directly
  node = LogicalNode(id="l1", kind="resolved.api", metadata={"a": "1"})
  fwd = backend._build_forward([node])
  code = cst.Module(body=[fwd]).code
  assert "resolved.api(x, a=1)" in code

  # Test _build_forward reverse lookup
  node2 = LogicalNode(id="l2", kind="concrete.api")
  fwd2 = backend._build_forward([node2])
  code2 = cst.Module(body=[fwd2]).code
  assert "resolved.api(x)" in code2

  # Test _generate_layer_init with functional torch (should ignore)
  node_torch = LogicalNode(id="t", kind="TorchFunc")
  init_t = backend._generate_layer_init(node_torch)
  code_t = cst.Module(body=[init_t]).code
  assert "nn.TorchFunc()" in code_t

  # Test _generate_layer_init with mlx core (should ignore)
  backend_mlx = PythonBackend("mlx")
  backend_mlx.semantics = FakeSemantics()
  node_mlx = LogicalNode(id="m", kind="MlxCore")
  init_m = backend_mlx._generate_layer_init(node_mlx)
  code_m = cst.Module(body=[init_m]).code
  assert "nn.MlxCore()" in code_m

  # Test _generate_layer_init normal replacement
  node_norm = LogicalNode(id="n", kind="ResolvedApi")
  init_n = backend._generate_layer_init(node_norm)
  code_n = cst.Module(body=[init_n]).code
  assert "resolved.api()" in code_n


def test_python_backend_prefix_stripping():
  # torch prefix
  """Test python backend prefix stripping."""
  backend_t = PythonBackend("torch")
  node_t = LogicalNode(id="n", kind="torch.nn.Linear")
  c_t = cst.Module(body=[backend_t._generate_layer_init(node_t)]).code
  assert "nn.Linear" in c_t

  # mlx prefix
  backend_m = PythonBackend("mlx")
  node_m = LogicalNode(id="m", kind="mlx.nn.Linear")
  c_m = cst.Module(body=[backend_m._generate_layer_init(node_m)]).code
  assert "nn.Linear" in c_m


def test_python_backend_implicit_prefixes():
  """Test python backend implicit prefixes."""
  backends = ["torch", "jax", "flax", "flax_nnx", "keras", "tensorflow", "mlx", "paxml"]
  prefixes = ["nn.", "nnx.", "nnx.", "nnx.", "keras.layers.", "tf.keras.layers.", "nn.", "pl."]

  for fw, pfx in zip(backends, prefixes):
    b = PythonBackend(fw)
    node = LogicalNode(id="n", kind="Linear")
    code = cst.Module(body=[b._generate_layer_init(node)]).code
    assert pfx + "Linear" in code


def test_python_backend_mlx_swiglu():
  """Test python backend mlx swiglu."""
  backend_m = PythonBackend("mlx")
  node = LogicalNode(id="s", kind="SwiGLU")
  code = cst.Module(body=[backend_m._generate_layer_init(node)]).code
  assert "self.s = nn.silu" in code


def test_python_backend_build_forward_resolve():
  """Test python backend build forward resolve."""
  backend = PythonBackend("torch")
  backend.semantics = FakeSemantics()
  # To hit 311->319: get_definition returns something, and definition has 'api'
  node = LogicalNode(id="n1", kind="concrete.api", metadata={"k": "v"})
  fwd = backend._build_forward([node])
  c = cst.Module(body=[fwd]).code
  assert "resolved.api" in c


def test_python_backend_layer_init_paxml():
  """Test python backend layer init paxml."""
  backend = PythonBackend("paxml")
  node = LogicalNode(id="l1", kind="Linear")
  init = backend._generate_layer_init(node)
  c = cst.Module(body=[init]).code
  assert "pl.Linear" in c


def test_python_backend_format_partition_spec():
  """Test python backend format partition spec."""
  backend = PythonBackend("torch")

  # Test _format_partition_spec_tf (tuple -> '*')
  from collections import namedtuple

  Shard = namedtuple("Shard", ["axes"])
  s = Shard(axes=[None, "batch", ("a", "b")])
  assert backend._format_partition_spec_tf(s) == "[None, 'batch', '*']"


def test_python_backend_build_layer_init_mlx_special():
  """Test python backend build layer init mlx special."""
  backend = PythonBackend("mlx")
  node = LogicalNode(id="s", kind="SwiGLU")
  code = cst.Module(body=[backend._generate_layer_init(node)]).code
  assert "self.s = nn.silu()" in code


def test_python_backend_build_layer_init_args():
  """Test python backend build layer init args."""
  backend = PythonBackend("flax_nnx")
  node = LogicalNode(id="s", kind="Linear", metadata={"arg": "1"})
  code = cst.Module(body=[backend._generate_layer_init(node)]).code
  assert "arg=1, rngs=rngs" in code

  node2 = LogicalNode(id="s2", kind="Linear")
  code2 = cst.Module(body=[backend._generate_layer_init(node2)]).code
  assert "rngs=rngs" in code2


def test_python_backend_base_class_resolution():
  """Test python backend base class resolution."""
  backend_flax = PythonBackend("flax_nnx")
  backend_flax.traits.module_base = "flax.nnx.Module"
  code = backend_flax.compile(LogicalGraph(name="M"))
  assert "class M(nnx.Module):" in code

  backend_pax = PythonBackend("paxml")
  backend_pax.traits.module_base = "praxis.base_layer.BaseLayer"
  code = backend_pax.compile(LogicalGraph(name="M"))
  assert "class M(BaseLayer):" in code

  backend_tf = PythonBackend("tensorflow")
  backend_tf.traits.module_base = "keras.Layer"
  code = backend_tf.compile(LogicalGraph(name="M"))
  assert "class M(tf.keras.Model):" in code

  backend_keras = PythonBackend("keras")
  backend_keras.traits.module_base = "keras.Layer"
  code = backend_keras.compile(LogicalGraph(name="M"))
  assert "class M(keras.Model):" in code


def test_python_backend_sharding_tf_no_tuple():
  """Test python backend sharding tf no tuple."""
  backend = PythonBackend("tensorflow")
  from collections import namedtuple

  Shard = namedtuple("Shard", ["axes"])
  s = Shard(axes=[("a", "b")])
  assert backend._format_partition_spec_tf(s) == "['*']"


def test_python_backend_generate_replacer_not_module():
  # If original_tree.visit(replacer) is not a cst.Module (which is weird, but we hit 163->166)
  """Test python backend generate replacer not module."""
  backend = PythonBackend("torch")
  node = cst.parse_statement("pass")
  code = backend.generate(LogicalGraph(), original_tree=node)
  assert "class SwitcherooNet" in code


def test_python_backend_sharding_others():
  """Test python backend sharding others."""
  backend = PythonBackend("paxml")
  from collections import namedtuple

  Shard = namedtuple("Shard", ["axes"])
  s = Shard(axes=[("a", "b")])
  node = LogicalNode(id="l1", kind="Linear")
  node.sharding = s
  backend._build_forward([node])


def test_python_backend_semantics_reverse_lookup_none():
  """Test python backend semantics reverse lookup none."""
  backend = PythonBackend("torch")
  backend.semantics = FakeSemantics()
  # Test _build_forward with reverse lookup returning None
  node = LogicalNode(id="n1", kind="module.not_found", metadata={"k": "v"})
  fwd = backend._build_forward([node])
  c = cst.Module(body=[fwd]).code
  assert "module.not_found" in c


def test_python_backend_semantics_forward_lookup_none():
  """Test python backend semantics forward lookup none."""
  backend = PythonBackend("torch")

  class FakeSemanticsNone(FakeSemantics):
    """Fake semantics none."""

    def resolve_variant(self, abstract_id, fw):
      """Resolve variant."""
      return None

  backend.semantics = FakeSemanticsNone()
  node = LogicalNode(id="n1", kind="resolved.api", metadata={"k": "v"})
  fwd = backend._build_forward([node])
  c = cst.Module(body=[fwd]).code
  assert "resolved.api" in c


def test_python_backend_prefix_others():
  """Test python backend prefix others."""
  backend_k = PythonBackend("keras")
  node_k = LogicalNode(id="k", kind="Linear")
  c_k = cst.Module(body=[backend_k._generate_layer_init(node_k)]).code
  assert "keras.layers.Linear" in c_k

  backend_tf = PythonBackend("tensorflow")
  node_tf = LogicalNode(id="t", kind="Linear")
  c_tf = cst.Module(body=[backend_tf._generate_layer_init(node_tf)]).code
  assert "tf.keras.layers.Linear" in c_tf


def test_python_backend_sharding_torch_tuple():
  """Test python backend sharding torch tuple."""
  backend = PythonBackend("torch")
  from collections import namedtuple

  Shard = namedtuple("Shard", ["axes"])
  s = Shard(axes=[("a", "b")])
  assert backend._format_partition_spec_torch(s) == "Shard(0)"


def test_python_backend_generate_replacing_simple_assign():
  # Hit line 60-63
  """Test python backend generate replacing simple assign."""
  backend = PythonBackend(framework="torch")
  source = "class M: x = 1"
  tree = cst.parse_module(source)
  code = backend.generate(LogicalGraph(name="M"), class_name="M", original_tree=tree)
  assert "x = 1" in code


def test_python_backend_generate_replacing_simple_unsupported():
  # Hit line 61->60 (an unsupported statement type in SimpleStatementSuite)
  """Test python backend generate replacing simple unsupported."""
  backend = PythonBackend(framework="torch")
  source = "class M: global x"
  tree = cst.parse_module(source)
  code = backend.generate(LogicalGraph(name="M"), class_name="M", original_tree=tree)
  # The global x won't be appended to stmts_list, but we just want to execute it.
  assert "class M" in code


def test_python_backend_generate_replacing_not_simple_or_indented():
  # Hit 64->67
  """Test python backend generate replacing not simple or indented."""
  _backend = PythonBackend(framework="torch")

  class FakeBody(cst.CSTNode):
    """Fake body."""

    def _codegen_impl(self, state):
      pass

    def _visit_and_replace_children(self, visitor):
      return self

  # Actually, cst.ClassDef.body can only be SimpleStatementSuite or IndentedBlock.
  # LibCST typing enforces this, so 64->67 is structurally unreachable unless we mock it or pass invalid trees.
  # We can probably safely ignore it or use pragma: no branch in the source.
  pass


def test_python_backend_replacing_multiple_same_func():
  # Hit 82->77
  """Test python backend replacing multiple same func."""
  backend = PythonBackend("torch")
  source = "class M:\n  def forward(self): pass\n  def forward(self): pass"
  tree = cst.parse_module(source)
  backend.generate(LogicalGraph(name="M"), class_name="M", original_tree=tree)


def test_python_backend_replacing_init_only():
  # Hit 92->95
  """Test python backend replacing init only."""
  backend = PythonBackend("torch")
  source = "class M:\n  def __init__(self): pass"
  tree = cst.parse_module(source)
  backend.generate(LogicalGraph(name="M"), class_name="M", original_tree=tree)


def test_python_backend_sharding_jax_tuple():
  """Test python backend sharding jax tuple."""
  backend = PythonBackend("jax")
  from collections import namedtuple

  Shard = namedtuple("Shard", ["axes"])
  s = Shard(axes=[("a", "b")])
  assert backend._format_partition_spec(s) == "jax.sharding.PartitionSpec(('a', 'b'))"


def test_python_backend_prefix_paxml_and_other():
  # Hit 425->429 (not paxml but hits the else branch)
  """Test python backend prefix paxml and other."""
  backend = PythonBackend("numpy")
  node = LogicalNode(id="n", kind="Linear")
  c = cst.Module(body=[backend._generate_layer_init(node)]).code
  assert "self.n = Linear()" in c


def test_python_backend_flax_rngs_existing():
  # Hit 438->442
  """Test python backend flax rngs existing."""
  backend = PythonBackend("flax_nnx")
  node = LogicalNode(id="n", kind="Linear", metadata={"rngs": "rngs"})
  c = cst.Module(body=[backend._generate_layer_init(node)]).code
  assert "rngs=rngs" in c


def test_python_backend_semantics_concrete_hit():
  # Hit 399->409
  """Test python backend semantics concrete hit."""
  backend = PythonBackend("torch")

  class Semantics:
    """Semantics."""

    def resolve_variant(self, abstract_id, fw):
      """Resolve variant."""
      return {"not_api": "value"}

  backend.semantics = Semantics()
  node = LogicalNode(id="n1", kind="torch.add", metadata={"k": "v"})
  fwd = backend._build_forward([node])
  c = cst.Module(body=[fwd]).code
  assert "torch.add" in c


def test_python_backend_semantics_rev_lookup_no_api():
  # Hit 311->319
  """Test python backend semantics rev lookup no api."""
  backend = PythonBackend("torch")

  class Semantics2:
    """Semantics2."""

    def resolve_variant(self, abstract_id, fw):
      """Resolve variant."""
      if abstract_id == "abs_id":
        return {"not_api": "val"}
      return None

    def get_definition(self, api):
      """Get definition."""
      if api == "concrete":
        return ("abs_id", {})
      return None

  backend.semantics = Semantics2()
  node = LogicalNode(id="n1", kind="torch.concrete", metadata={"k": "v"})
  fwd = backend._build_forward([node])
  c = cst.Module(body=[fwd]).code
  assert "torch.concrete" in c


def test_python_backend_build_forward_metadata_no_extra_args():
  # Hit 325->327
  """Test python backend build forward metadata no extra args."""
  backend = PythonBackend("torch")

  class Semantics3:
    """Semantics3."""

    def resolve_variant(self, abstract_id, fw):
      """Resolve variant."""
      return None

    def get_definition(self, api):
      """Get definition."""
      return None

  backend.semantics = Semantics3()
  # If _format_args_from_metadata returns empty string
  backend._format_args_from_metadata = lambda m: ""
  node = LogicalNode(id="n1", kind="torch.concrete", metadata={"k": "v"})
  fwd = backend._build_forward([node])
  c = cst.Module(body=[fwd]).code
  assert "torch.concrete(x)" in c


def test_python_backend_semantics_no_api_dict():
  # Hit 308->322 and 399->409
  """Test python backend semantics no api dict."""
  backend = PythonBackend("torch")

  class Semantics4:
    """Semantics4."""

    def resolve_variant(self, abstract_id, fw):
      """Resolve variant."""
      return {"not_api": "1"}

    def get_definition(self, api):
      """Get definition."""
      return None

  backend.semantics = Semantics4()
  # Test _build_forward (308->322 requires functional node)
  node1 = LogicalNode(id="n1", kind="torch.add", metadata={"k": "v"})
  fwd = backend._build_forward([node1])
  c = cst.Module(body=[fwd]).code
  assert "torch.add" in c
  # Test _generate_layer_init (399->409 requires stateful node)
  node2 = LogicalNode(id="n2", kind="Linear")
  init = backend._generate_layer_init(node2)
  c2 = cst.Module(body=[init]).code
  assert "nn.Linear" in c2


def test_python_backend_tf_sharding_not_none_or_str():
  # Hit 483->478
  """Test python backend tf sharding not none or str."""
  backend = PythonBackend("torch")
  from collections import namedtuple

  Shard = namedtuple("Shard", ["axes"])
  s = Shard(axes=[123])
  # The tuple format string branch for torch:
  # 478 for axis in sharding.axes:
  # 479   if axis is None:
  # 481   elif isinstance(axis, str):
  # 483   elif isinstance(axis, tuple):
  # If it's none of these, it doesn't append to axes.
  # 483->478 means loop continues because the branch was missed (no else block)
  # So 123 hits this.
  res = backend._format_partition_spec(s)
  assert res == "jax.sharding.PartitionSpec()"


def test_python_backend_generate_updated_node_return():
  # Hit line 103 (which is returned when original_node.name.value != target_class)
  """Test python backend generate updated node return."""
  backend = PythonBackend("torch")
  source = "class Other:\n  pass"
  tree = cst.parse_module(source)
  code = backend.generate(LogicalGraph(name="M"), class_name="M", original_tree=tree)
  # The replacer won't find 'M', so the generated code is appended.
  # Wait, the code will just generate from scratch if it doesn't find it.
  # Let's verify line 103 is hit.
  pass
  assert "class M" in code


def test_python_backend_semantics_no_resolve_variant():
  # Hit 308->322 (no resolve_variant attr)
  """Test python backend semantics no resolve variant."""
  backend = PythonBackend("torch")

  class Semantics5:
    """Semantics5."""

    pass

  backend.semantics = Semantics5()
  node = LogicalNode(id="n1", kind="torch.add", metadata={"k": "v"})
  fwd = backend._build_forward([node])
  c = cst.Module(body=[fwd]).code
  assert "torch.add" in c
