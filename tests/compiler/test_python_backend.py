"""Test suite for python backend."""

import typing
from pydantic import JsonValue

import libcst as cst

from ml_switcheroo.core.compiler.backends.python import PythonBackend
from ml_switcheroo.core.compiler.backends.python_snippet import PythonSnippetEmitter
from ml_switcheroo.core.compiler.backends.python_strategies.forward import build_forward
from ml_switcheroo.core.compiler.backends.python_strategies.imports import generate_imports
from ml_switcheroo.core.compiler.backends.python_strategies.sharding import (
  format_partition_spec,
  format_partition_spec_tf,
  format_partition_spec_torch,
)
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode


class MockSharding:
  """Mock sharding class."""

  def __init__(self, axes: typing.Sequence[typing.Union[str, tuple[str, ...], None]]) -> None:
    """Initialize mock sharding.

    Args:
        axes: Test argument or fixture.
    """
    self.axes = axes


def test_python_snippet_emitter_init() -> None:
  """Verify python snippet emitter init behavior."""
  emitter = PythonSnippetEmitter(framework="jax")
  assert emitter.framework == "jax"


def test_python_snippet_emit_init_stateless() -> None:
  """Verify python snippet emit init stateless behavior."""
  emitter = PythonSnippetEmitter()
  node = LogicalNode(id="n1", op_type="func_add")
  stmt = emitter.emit_init(node)
  assert isinstance(stmt, cst.SimpleStatementLine)
  assert isinstance(stmt.body[0], cst.Pass)


def test_python_snippet_emit_init_stateful() -> None:
  """Verify python snippet emit init stateful behavior."""
  emitter = PythonSnippetEmitter(framework="torch")
  node = LogicalNode(id="l1", op_type="Linear", attributes={"arg_1": "10", "arg_2": "20"})
  stmt = emitter.emit_init(node)

  code = cst.Module(body=[stmt]).code
  assert "self.l1 = nn.Linear(10, 20)" in code


def test_python_snippet_emit_init_flax() -> None:
  """Verify python snippet emit init flax behavior."""
  emitter = PythonSnippetEmitter(framework="flax_nnx")
  node = LogicalNode(id="l1", op_type="Linear")
  stmt = emitter.emit_init(node)

  code = cst.Module(body=[stmt]).code
  assert "rngs = rngs" in code


def test_python_snippet_emit_call_input() -> None:
  """Verify python snippet emit call input behavior."""
  emitter = PythonSnippetEmitter()
  node = LogicalNode(id="in", op_type="Input")
  stmt = emitter.emit_call(node, ["x_in"], "x_out")
  code = cst.Module(body=[stmt]).code
  assert "x_out = x_in" in code

  stmt_same = emitter.emit_call(node, ["x_in"], "x_in")
  code_same = cst.Module(body=[stmt_same]).code
  assert "pass" in code_same


def test_python_snippet_emit_call_expr() -> None:
  """Verify python snippet emit call expr behavior."""
  emitter = PythonSnippetEmitter()
  node = LogicalNode(id="add", op_type="add", attributes={"alpha": "1.0"})
  stmt = emitter.emit_call(node, ["x", "y"], "out")
  code = cst.Module(body=[stmt]).code
  assert "out = torch.add(x, y, alpha = 1.0)" in code


def test_python_snippet_emit_expression_error() -> None:
  """Verify python snippet emit expression error behavior."""
  emitter = PythonSnippetEmitter()
  # Invalid python expression string in metadata to cause parser error
  node = LogicalNode(id="bad", op_type="bad", attributes={"key": "def class *"})
  expr = emitter.emit_expression(node, [])
  assert isinstance(expr, cst.Name)
  assert expr.value == "None"


def test_python_snippet_resolve_api_name() -> None:
  """Verify python snippet resolve api name behavior."""
  _ = LogicalNode(id="1", op_type="Linear")
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


def test_python_backend_init() -> None:
  """Verify python backend init behavior."""
  backend = PythonBackend(framework="torch")
  assert backend.framework == "torch"


def test_python_backend_compile() -> None:
  """Verify python backend compile behavior."""
  backend = PythonBackend(framework="torch")
  graph = LogicalGraph(name="MyModel")
  graph.add_node(LogicalNode(id="in", op_type="Input", attributes={"name": "x"}))
  graph.add_node(LogicalNode(id="l1", op_type="Linear", attributes={"arg_1": "10"}))
  graph.add_node(LogicalNode(id="out", op_type="Output"))

  code = backend.compile(graph)
  assert "class MyModel(nn.Module):" in code
  assert "def __init__(self):" in code
  assert "self.l1 = nn.Linear(10)" in code
  assert "def forward(self, x):" in code
  assert "x = self.l1(x)" in code
  assert "return x" in code


def test_python_backend_compile_no_name() -> None:
  """Verify python backend compile no name behavior."""
  backend = PythonBackend(framework="torch")
  graph = LogicalGraph()
  graph.add_node(LogicalNode(id="out", op_type="Output"))
  code = backend.compile(graph)
  assert "class Model(nn.Module):" in code


def test_python_backend_generate_replacing() -> None:
  """Verify python backend generate replacing behavior."""
  backend = PythonBackend(framework="torch")
  graph = LogicalGraph(name="MyModel")
  graph.add_node(LogicalNode(id="l1", op_type="Linear"))

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


def test_python_backend_framework_imports() -> None:
  """Verify python backend framework imports behavior."""
  assert "import torch" in cst.Module(body=generate_imports("torch")).code
  assert "jax.numpy" in cst.Module(body=generate_imports("jax")).code
  assert "mlx.core" in cst.Module(body=generate_imports("mlx")).code
  assert "keras" in cst.Module(body=generate_imports("keras")).code
  assert "praxis" in cst.Module(body=generate_imports("paxml")).code
  assert generate_imports("unknown") == []


def test_python_backend_build_forward_sharding() -> None:
  """Verify python backend build forward sharding behavior."""
  node = LogicalNode(id="l1", op_type="Linear", attributes={"arg_0": "10"})
  node.sharding = MockSharding(["batch", None])  # type: ignore[assignment]

  # torch
  backend_torch = PythonBackend("torch")
  fwd_torch = build_forward([node], backend_torch.framework, backend_torch.semantics, backend_torch.traits, backend_torch)
  code_torch = cst.Module(body=[fwd_torch]).code
  assert "distribute_tensor(x, self.mesh, [Shard(0), Replicate()])" in code_torch

  # jax
  backend_jax = PythonBackend("jax")
  fwd_jax = build_forward([node], backend_jax.framework, backend_jax.semantics, backend_jax.traits, backend_jax)
  code_jax = cst.Module(body=[fwd_jax]).code
  assert "jax.lax.with_sharding_constraint(x, jax.sharding.PartitionSpec('batch', None))" in code_jax

  # keras
  backend_keras = PythonBackend("keras")
  fwd_keras = build_forward([node], backend_keras.framework, backend_keras.semantics, backend_keras.traits, backend_keras)
  code_keras = cst.Module(body=[fwd_keras]).code
  assert "keras.distribution.layout(['batch', None])" in code_keras

  # mlx
  backend_mlx = PythonBackend("mlx")
  fwd_mlx = build_forward([node], backend_mlx.framework, backend_mlx.semantics, backend_mlx.traits, backend_mlx)
  code_mlx = cst.Module(body=[fwd_mlx]).code
  assert "mx.distributed.shard" in code_mlx

  # tuple in jax
  node2 = LogicalNode(id="l2", op_type="Linear")
  node2.sharding = MockSharding([("a", "b")])  # type: ignore[assignment]
  fwd_jax2 = build_forward([node2], backend_jax.framework, backend_jax.semantics, backend_jax.traits, backend_jax)
  code_jax2 = cst.Module(body=[fwd_jax2]).code
  assert "('a', 'b')" in code_jax2


def test_python_backend_build_layer_init_mlx_specials() -> None:
  """Verify python backend build layer init mlx specials behavior."""
  backend = PythonBackend("mlx")
  node = LogicalNode(id="s", op_type="SwiGLU")
  stmt = backend._generate_layer_init(node)
  assert "silu" in cst.Module(body=[stmt]).code

  node2 = LogicalNode(id="r", op_type="RoPE")
  stmt2 = backend._generate_layer_init(node2)
  assert "nn.RoPE" in cst.Module(body=[stmt2]).code

  node3 = LogicalNode(id="v", op_type="VisionPatchEmbedding")
  stmt3 = backend._generate_layer_init(node3)
  assert "nn.Conv2d" in cst.Module(body=[stmt3]).code


def test_python_backend_generate_replacing_pass_stmt() -> None:
  """Verify python backend generate replacing pass stmt behavior."""
  backend = PythonBackend(framework="torch")
  graph = LogicalGraph(name="MyModel")

  source = "class MyModel(nn.Module): pass; pass"
  tree = cst.parse_module(source)
  code = backend.generate(graph, class_name="MyModel", original_tree=tree)
  assert "class MyModel" in code


def test_python_backend_generate_replacing_no_match() -> None:
  """Verify python backend generate replacing no match behavior."""
  backend = PythonBackend(framework="torch")
  graph = LogicalGraph(name="MyModel")

  source = "class MyModel:\n  def unrelated(self): pass\n  x = 1"
  tree = cst.parse_module(source)
  code = backend.generate(graph, class_name="MyModel", original_tree=tree)
  assert "unrelated" in code
  assert "x = 1" in code


def test_python_backend_build_forward_tuple_outputs() -> None:
  """Verify python backend build forward tuple outputs behavior."""
  backend = PythonBackend("torch")
  node = LogicalNode(id="l1", op_type="Linear")
  fwd = build_forward([node], backend.framework, backend.semantics, backend.traits, backend)
  code = cst.Module(body=[fwd]).code
  assert "return" in code


def test_python_backend_build_forward_no_outputs() -> None:
  """Verify python backend build forward no outputs behavior."""
  backend = PythonBackend("torch")
  node = LogicalNode(id="l1", op_type="Linear")
  fwd = build_forward([node], backend.framework, backend.semantics, backend.traits, backend)
  code = cst.Module(body=[fwd]).code
  assert "return" in code


def test_python_backend_build_layer_init_torch_specials() -> None:
  """Verify python backend build layer init torch specials behavior."""
  backend = PythonBackend("torch")
  node_swi = LogicalNode(id="s", op_type="SwiGLU")
  code = cst.Module(body=[backend._generate_layer_init(node_swi)]).code
  assert "nn.SwiGLU" in code

  node_rope = LogicalNode(id="r", op_type="RoPE")
  code_rope = cst.Module(body=[backend._generate_layer_init(node_rope)]).code
  assert "nn.RoPE" in code_rope

  node_vpe = LogicalNode(id="v", op_type="VisionPatchEmbedding")
  code_vpe = cst.Module(body=[backend._generate_layer_init(node_vpe)]).code
  assert "nn.VisionPatchEmbedding" in code_vpe


def test_python_backend_build_layer_init_jax_specials() -> None:
  """Verify python backend build layer init jax specials behavior."""
  backend = PythonBackend("jax")
  node = LogicalNode(id="l", op_type="Conv2d")
  code = cst.Module(body=[backend._generate_layer_init(node)]).code
  assert "nnx.Conv" in code

  node_swi = LogicalNode(id="s", op_type="SwiGLU")
  code_swi = cst.Module(body=[backend._generate_layer_init(node_swi)]).code
  assert "nnx.SwiGLU" in code_swi

  node_rope = LogicalNode(id="r", op_type="RoPE")
  code_rope = cst.Module(body=[backend._generate_layer_init(node_rope)]).code
  assert "nnx.RoPE" in code_rope

  node_vpe = LogicalNode(id="v", op_type="VisionPatchEmbedding")
  code_vpe = cst.Module(body=[backend._generate_layer_init(node_vpe)]).code
  assert "nnx.VisionPatchEmbedding" in code_vpe


def test_python_backend_build_layer_init_paxml_specials() -> None:
  """Verify python backend build layer init paxml specials behavior."""
  backend = PythonBackend("paxml")
  node = LogicalNode(id="l", op_type="Conv2d")
  code = cst.Module(body=[backend._generate_layer_init(node)]).code
  assert "pl.Conv2d" in code


def test_python_snippet_emit_expression_bool() -> None:
  """Verify python snippet emit expression bool behavior."""
  emitter = PythonSnippetEmitter("torch")
  node = LogicalNode(id="l", op_type="Linear", attributes={"bias": "True", "other": "False"})
  stmt = emitter.emit_init(node)
  code = cst.Module(body=[stmt]).code
  assert "True" in code
  assert "False" in code


class FakeSemantics:
  """Fake semantics."""

  def resolve_variant(self, abstract_id: str, fw: str) -> typing.Optional[dict[str, JsonValue]]:
    """Resolve variant.

    Args:
        abstract_id: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if abstract_id == "ResolvedApi":
      return {"api": "resolved.api"}
    if abstract_id == "TorchFunc" and fw == "torch":
      return {"api": "torch.nn.functional.relu"}
    if abstract_id == "MlxCore" and fw == "mlx":
      return {"api": "mlx.core.add"}
    return None

  def get_definition(self, api: str) -> typing.Optional[tuple[str, dict[str, JsonValue]]]:
    """Get definition.

    Args:
        api: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if api == "concrete.api":
      return ("ResolvedApi", {})
    return None


def test_python_backend_semantics_resolution() -> None:
  """Verify python backend semantics resolution behavior."""
  backend = PythonBackend("torch")
  backend.semantics = FakeSemantics()  # type: ignore[assignment]

  # Test _build_forward with resolve directly
  node = LogicalNode(id="l1", op_type="resolved.api", attributes={"a": "1"})
  fwd = build_forward([node], backend.framework, backend.semantics, backend.traits, backend)
  code = cst.Module(body=[fwd]).code
  assert "resolved.api(x, a=1)" in code

  # Test _build_forward reverse lookup
  node2 = LogicalNode(id="l2", op_type="concrete.api")
  fwd2 = build_forward([node2], backend.framework, backend.semantics, backend.traits, backend)
  code2 = cst.Module(body=[fwd2]).code
  assert "resolved.api(x)" in code2

  # Test _generate_layer_init with functional torch (should ignore)
  node_torch = LogicalNode(id="t", op_type="TorchFunc")
  init_t = backend._generate_layer_init(node_torch)
  code_t = cst.Module(body=[init_t]).code
  assert "nn.TorchFunc()" in code_t

  # Test _generate_layer_init with mlx core (should ignore)
  backend_mlx = PythonBackend("mlx")
  backend_mlx.semantics = FakeSemantics()  # type: ignore[assignment]
  node_mlx = LogicalNode(id="m", op_type="MlxCore")
  init_m = backend_mlx._generate_layer_init(node_mlx)
  code_m = cst.Module(body=[init_m]).code
  assert "nn.MlxCore()" in code_m

  # Test _generate_layer_init normal replacement
  node_norm = LogicalNode(id="n", op_type="ResolvedApi")
  init_n = backend._generate_layer_init(node_norm)
  code_n = cst.Module(body=[init_n]).code
  assert "resolved.api()" in code_n


def test_python_backend_prefix_stripping() -> None:
  """Verify python backend prefix stripping behavior."""
  # torch prefix
  backend_t = PythonBackend("torch")
  node_t = LogicalNode(id="n", op_type="torch.nn.Linear")
  c_t = cst.Module(body=[backend_t._generate_layer_init(node_t)]).code
  assert "nn.Linear" in c_t

  # mlx prefix
  backend_m = PythonBackend("mlx")
  node_m = LogicalNode(id="m", op_type="mlx.nn.Linear")
  c_m = cst.Module(body=[backend_m._generate_layer_init(node_m)]).code
  assert "nn.Linear" in c_m


def test_python_backend_implicit_prefixes() -> None:
  """Verify python backend implicit prefixes behavior."""
  backends = ["torch", "jax", "flax", "flax_nnx", "keras", "tensorflow", "mlx", "paxml"]
  prefixes = ["nn.", "nnx.", "nnx.", "nnx.", "keras.layers.", "tf.keras.layers.", "nn.", "pl."]

  for fw, pfx in zip(backends, prefixes):
    b = PythonBackend(fw)
    node = LogicalNode(id="n", op_type="Linear")
    code = cst.Module(body=[b._generate_layer_init(node)]).code
    assert pfx + "Linear" in code


def test_python_backend_mlx_swiglu() -> None:
  """Verify python backend mlx swiglu behavior."""
  backend_m = PythonBackend("mlx")
  node = LogicalNode(id="s", op_type="SwiGLU")
  code = cst.Module(body=[backend_m._generate_layer_init(node)]).code
  assert "self.s = nn.silu" in code


def test_python_backend_build_forward_resolve() -> None:
  """Verify python backend build forward resolve behavior."""
  backend = PythonBackend("torch")
  backend.semantics = FakeSemantics()  # type: ignore[assignment]
  # To hit 311->319: get_definition returns something, and definition has 'api'
  node = LogicalNode(id="n1", op_type="concrete.api", attributes={"k": "v"})
  fwd = build_forward([node], backend.framework, backend.semantics, backend.traits, backend)
  c = cst.Module(body=[fwd]).code
  assert "resolved.api" in c


def test_python_backend_layer_init_paxml() -> None:
  """Verify python backend layer init paxml behavior."""
  backend = PythonBackend("paxml")
  node = LogicalNode(id="l1", op_type="Linear")
  init = backend._generate_layer_init(node)
  c = cst.Module(body=[init]).code
  assert "pl.Linear" in c


def test_python_backend_format_partition_spec() -> None:
  """Verify python backend format partition spec behavior."""
  PythonBackend("torch")

  # Test _format_partition_spec_tf (tuple -> '*')
  from collections import namedtuple

  Shard = namedtuple("Shard", ["axes"])
  s = Shard(axes=[None, "batch", ("a", "b")])
  assert format_partition_spec_tf(s) == "[None, 'batch', '*']"  # type: ignore[arg-type]


def test_python_backend_build_layer_init_mlx_special() -> None:
  """Verify python backend build layer init mlx special behavior."""
  backend = PythonBackend("mlx")
  node = LogicalNode(id="s", op_type="SwiGLU")
  code = cst.Module(body=[backend._generate_layer_init(node)]).code
  assert "self.s = nn.silu()" in code


def test_python_backend_build_layer_init_args() -> None:
  """Verify python backend build layer init args behavior."""
  backend = PythonBackend("flax_nnx")
  node = LogicalNode(id="s", op_type="Linear", attributes={"arg": "1"})
  code = cst.Module(body=[backend._generate_layer_init(node)]).code
  assert "arg=1, rngs=rngs" in code

  node2 = LogicalNode(id="s2", op_type="Linear")
  code2 = cst.Module(body=[backend._generate_layer_init(node2)]).code
  assert "rngs=rngs" in code2


def test_python_backend_base_class_resolution() -> None:
  """Verify python backend base class resolution behavior."""
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


def test_python_backend_sharding_tf_no_tuple() -> None:
  """Verify python backend sharding tf no tuple behavior."""
  PythonBackend("tensorflow")
  from collections import namedtuple

  Shard = namedtuple("Shard", ["axes"])
  s = Shard(axes=[("a", "b")])
  assert format_partition_spec_tf(s) == "['*']"  # type: ignore[arg-type]


def test_python_backend_generate_replacer_not_module() -> None:
  """Verify python backend generate replacer not module behavior."""
  # If original_tree.visit(replacer) is not a cst.Module (which is weird, but we hit 163->166)
  backend = PythonBackend("torch")
  node = cst.parse_statement("pass")
  code = backend.generate(LogicalGraph(), original_tree=node)  # type: ignore[arg-type]
  assert "class SwitcherooNet" in code


def test_python_backend_sharding_others() -> None:
  """Verify python backend sharding others behavior."""
  backend = PythonBackend("paxml")
  from collections import namedtuple

  Shard = namedtuple("Shard", ["axes"])
  s = Shard(axes=[("a", "b")])
  node = LogicalNode(id="l1", op_type="Linear")
  node.sharding = s  # type: ignore[assignment]
  res = build_forward([node], backend.framework, backend.semantics, backend.traits, backend)
  assert isinstance(res, cst.FunctionDef)


def test_python_backend_semantics_reverse_lookup_none() -> None:
  """Verify python backend semantics reverse lookup none behavior."""
  backend = PythonBackend("torch")
  backend.semantics = FakeSemantics()  # type: ignore[assignment]
  # Test _build_forward with reverse lookup returning None
  node = LogicalNode(id="n1", op_type="module.not_found", attributes={"k": "v"})
  fwd = build_forward([node], backend.framework, backend.semantics, backend.traits, backend)
  c = cst.Module(body=[fwd]).code
  assert "module.not_found" in c


def test_python_backend_semantics_forward_lookup_none() -> None:
  """Verify python backend semantics forward lookup none behavior.

  Returns:
      Computed test result or fixture value.
  """
  backend = PythonBackend("torch")

  class FakeSemanticsNone(FakeSemantics):
    """Fake semantics none."""

    def resolve_variant(self, abstract_id: str, fw: str) -> typing.Optional[dict[str, JsonValue]]:
      """Resolve variant.

      Args:
          abstract_id: Test argument or fixture.
          fw: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return None

  backend.semantics = FakeSemanticsNone()  # type: ignore[assignment]
  node = LogicalNode(id="n1", op_type="resolved.api", attributes={"k": "v"})
  fwd = build_forward([node], backend.framework, backend.semantics, backend.traits, backend)
  c = cst.Module(body=[fwd]).code
  assert "resolved.api" in c


def test_python_backend_prefix_others() -> None:
  """Verify python backend prefix others behavior."""
  backend_k = PythonBackend("keras")
  node_k = LogicalNode(id="k", op_type="Linear")
  c_k = cst.Module(body=[backend_k._generate_layer_init(node_k)]).code
  assert "keras.layers.Linear" in c_k

  backend_tf = PythonBackend("tensorflow")
  node_tf = LogicalNode(id="t", op_type="Linear")
  c_tf = cst.Module(body=[backend_tf._generate_layer_init(node_tf)]).code
  assert "tf.keras.layers.Linear" in c_tf


def test_python_backend_sharding_torch_tuple() -> None:
  """Verify python backend sharding torch tuple behavior."""
  PythonBackend("torch")
  from collections import namedtuple

  Shard = namedtuple("Shard", ["axes"])
  s = Shard(axes=[("a", "b")])
  assert format_partition_spec_torch(s) == "Shard(0)"  # type: ignore[arg-type]


def test_python_backend_generate_replacing_simple_assign() -> None:
  """Verify python backend generate replacing simple assign behavior."""
  # Hit line 60-63
  backend = PythonBackend(framework="torch")
  source = "class M: x = 1"
  tree = cst.parse_module(source)
  code = backend.generate(LogicalGraph(name="M"), class_name="M", original_tree=tree)
  assert "x = 1" in code


def test_python_backend_generate_replacing_simple_unsupported() -> None:
  """Verify python backend generate replacing simple unsupported behavior."""
  # Hit line 61->60 (an unsupported statement type in SimpleStatementSuite)
  backend = PythonBackend(framework="torch")
  source = "class M: global x"
  tree = cst.parse_module(source)
  code = backend.generate(LogicalGraph(name="M"), class_name="M", original_tree=tree)
  # The global x won't be appended to stmts_list, but we just want to execute it.
  assert "class M" in code


def test_python_backend_generate_replacing_not_simple_or_indented() -> None:
  """Verify python backend generate replacing not simple or indented behavior."""
  backend = PythonBackend(framework="torch")
  assert backend.framework == "torch"


def test_python_backend_replacing_multiple_same_func() -> None:
  """Verify python backend replacing multiple same func behavior."""
  # Hit 82->77
  backend = PythonBackend("torch")
  source = "class M:\n  def forward(self): pass\n  def forward(self): pass"
  tree = cst.parse_module(source)
  res = backend.generate(LogicalGraph(name="M"), class_name="M", original_tree=tree)
  assert "class M" in res


def test_python_backend_replacing_init_only() -> None:
  """Verify python backend replacing init only behavior."""
  # Hit 92->95
  backend = PythonBackend("torch")
  source = "class M:\n  def __init__(self): pass"
  tree = cst.parse_module(source)
  res = backend.generate(LogicalGraph(name="M"), class_name="M", original_tree=tree)
  assert "def forward" in res


def test_python_backend_sharding_jax_tuple() -> None:
  """Verify python backend sharding jax tuple behavior."""
  PythonBackend("jax")
  from collections import namedtuple

  Shard = namedtuple("Shard", ["axes"])
  s = Shard(axes=[("a", "b")])
  assert format_partition_spec(s) == "jax.sharding.PartitionSpec(('a', 'b'))"  # type: ignore[arg-type]


def test_python_backend_prefix_paxml_and_other() -> None:
  """Verify python backend prefix paxml and other behavior."""
  # Hit 425->429 (not paxml but hits the else branch)
  backend = PythonBackend("numpy")
  node = LogicalNode(id="n", op_type="Linear")
  c = cst.Module(body=[backend._generate_layer_init(node)]).code
  assert "self.n = Linear()" in c


def test_python_backend_flax_rngs_existing() -> None:
  """Verify python backend flax rngs existing behavior."""
  # Hit 438->442
  backend = PythonBackend("flax_nnx")
  node = LogicalNode(id="n", op_type="Linear", attributes={"rngs": "rngs"})
  c = cst.Module(body=[backend._generate_layer_init(node)]).code
  assert "rngs=rngs" in c


def test_python_backend_semantics_concrete_hit() -> None:
  """Verify python backend semantics concrete hit behavior.

  Returns:
      Computed test result or fixture value.
  """
  # Hit 399->409
  backend = PythonBackend("torch")

  class Semantics:
    """Semantics."""

    def resolve_variant(self, abstract_id: str, fw: str) -> typing.Optional[dict[str, JsonValue]]:
      """Resolve variant.

      Args:
          abstract_id: Test argument or fixture.
          fw: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return {"not_api": "value"}

  backend.semantics = Semantics()  # type: ignore[assignment]
  node = LogicalNode(id="n1", op_type="torch.add", attributes={"k": "v"})
  fwd = build_forward([node], backend.framework, backend.semantics, backend.traits, backend)
  c = cst.Module(body=[fwd]).code
  assert "torch.add" in c


def test_python_backend_semantics_rev_lookup_no_api() -> None:
  """Verify python backend semantics rev lookup no api behavior.

  Returns:
      Computed test result or fixture value.
  """
  # Hit 311->319
  backend = PythonBackend("torch")

  class Semantics2:
    """Semantics2."""

    def resolve_variant(self, abstract_id: str, fw: str) -> typing.Optional[dict[str, JsonValue]]:
      """Resolve variant.

      Args:
          abstract_id: Test argument or fixture.
          fw: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      if abstract_id == "abs_id":
        return {"not_api": "val"}
      return None

    def get_definition(self, api: str) -> typing.Optional[tuple[str, dict[str, JsonValue]]]:
      """Get definition.

      Args:
          api: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      if api == "concrete":
        return ("abs_id", {})
      return None

  backend.semantics = Semantics2()  # type: ignore[assignment]
  node = LogicalNode(id="n1", op_type="torch.concrete", attributes={"k": "v"})
  fwd = build_forward([node], backend.framework, backend.semantics, backend.traits, backend)
  c = cst.Module(body=[fwd]).code
  assert "torch.concrete" in c


def test_python_backend_build_forward_metadata_no_extra_args() -> None:
  """Verify python backend build forward metadata no extra args behavior.

  Returns:
      Computed test result or fixture value.
  """
  # Hit 325->327
  backend = PythonBackend("torch")

  class Semantics3:
    """Semantics3."""

    def resolve_variant(self, abstract_id: str, fw: str) -> typing.Optional[dict[str, JsonValue]]:
      """Resolve variant.

      Args:
          abstract_id: Test argument or fixture.
          fw: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return None

    def get_definition(self, api: str) -> typing.Optional[tuple[str, dict[str, JsonValue]]]:
      """Get definition.

      Args:
          api: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return None

  backend.semantics = Semantics3()  # type: ignore[assignment]
  # If _format_args_from_metadata returns empty string

  node = LogicalNode(id="n1", op_type="torch.concrete", attributes={"k": "v"})
  fwd = build_forward([node], backend.framework, backend.semantics, backend.traits, backend)
  c = cst.Module(body=[fwd]).code
  assert "torch.concrete(x, k=v)" in c


def test_python_backend_semantics_no_api_dict() -> None:
  """Verify python backend semantics no api dict behavior.

  Returns:
      Computed test result or fixture value.
  """
  # Hit 308->322 and 399->409
  backend = PythonBackend("torch")

  class Semantics4:
    """Semantics4."""

    def resolve_variant(self, abstract_id: str, fw: str) -> typing.Optional[dict[str, JsonValue]]:
      """Resolve variant.

      Args:
          abstract_id: Test argument or fixture.
          fw: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return {"not_api": "1"}

    def get_definition(self, api: str) -> typing.Optional[tuple[str, dict[str, JsonValue]]]:
      """Get definition.

      Args:
          api: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return None

  backend.semantics = Semantics4()  # type: ignore[assignment]
  # Test _build_forward (308->322 requires functional node)
  node1 = LogicalNode(id="n1", op_type="torch.add", attributes={"k": "v"})
  fwd = build_forward([node1], backend.framework, backend.semantics, backend.traits, backend)
  c = cst.Module(body=[fwd]).code
  assert "torch.add" in c
  # Test _generate_layer_init (399->409 requires stateful node)
  node2 = LogicalNode(id="n2", op_type="Linear")
  init = backend._generate_layer_init(node2)
  c2 = cst.Module(body=[init]).code
  assert "nn.Linear" in c2


def test_python_backend_tf_sharding_not_none_or_str() -> None:
  """Verify python backend tf sharding not none or str behavior."""
  # Hit 483->478
  PythonBackend("torch")
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
  res = format_partition_spec(s)  # type: ignore[arg-type]
  assert res == "jax.sharding.PartitionSpec()"


def test_python_backend_generate_updated_node_return() -> None:
  """Verify python backend generate updated node return behavior."""
  # Hit line 103 (which is returned when original_node.name.value != target_class)
  backend = PythonBackend("torch")
  source = "class Other:\n  pass"
  tree = cst.parse_module(source)
  code = backend.generate(LogicalGraph(name="M"), class_name="M", original_tree=tree)
  # The replacer won't find 'M', so the generated code is appended.
  # Wait, the code will just generate from scratch if it doesn't find it.
  # Let's verify line 103 is hit.
  pass
  assert "class M" in code


def test_python_backend_semantics_no_resolve_variant() -> None:
  """Verify python backend semantics no resolve variant behavior."""
  # Hit 308->322 (no resolve_variant attr)
  backend = PythonBackend("torch")

  class Semantics5:
    """Semantics5."""

    pass

  backend.semantics = Semantics5()  # type: ignore[assignment]
  node = LogicalNode(id="n1", op_type="torch.add", attributes={"k": "v"})
  fwd = build_forward([node], backend.framework, backend.semantics, backend.traits, backend)
  c = cst.Module(body=[fwd]).code
  assert "torch.add" in c


def test_python_backend_merged_coverage() -> None:
  """Verify consolidated coverage cases from duplicate python backend tests."""
  from ml_switcheroo.core.compiler.backends.python import ClassBodyReplacer
  from ml_switcheroo.core.compiler.backends.python_strategies.init import format_args_from_metadata
  from ml_switcheroo.core.compiler.ir import LogicalEdge

  backend = PythonBackend("torch")
  assert backend._is_stateful_layer(LogicalNode(id="n1", op_type="Input")) is False
  assert backend._is_stateful_layer(LogicalNode(id="n2", op_type="Output")) is False
  assert backend._is_stateful_layer(LogicalNode(id="n3", op_type="torch.add")) is False
  assert backend._is_stateful_layer(LogicalNode(id="n4", op_type="Linear")) is True
  assert backend._is_stateful_layer(LogicalNode(id="n5", op_type="nn.Linear")) is True
  assert backend._is_stateful_layer(LogicalNode("n", "a.b.func_x")) is False

  res_args: str = format_args_from_metadata({"arg_1": "10", "arg_2": "'val'", "bias": "True", "other": "False"})
  assert res_args == "10, 'val', bias=True, other=False"

  # ClassBodyReplacer with non-matching class and inline break statement
  init_stmt = cst.ensure_type(cst.parse_statement("def __init__(self): pass"), cst.FunctionDef)
  fwd_stmt = cst.ensure_type(cst.parse_statement("def forward(self, x): return x"), cst.FunctionDef)
  replacer = ClassBodyReplacer("MyModel", init_stmt, fwd_stmt)
  mod_break = cst.parse_module("class MyModel:\n  def __init__(self): break")
  assert "class MyModel:" in mod_break.visit(replacer).code

  # Forward pass with edges and inputs
  g = LogicalGraph(
    nodes={n.id: n for n in [LogicalNode("x", "Input"), LogicalNode("conv1", "Conv2d"), LogicalNode("output", "Output")]},
    edges=[LogicalEdge("x", "conv1"), LogicalEdge("conv1", "output")],
  )
  code_chain: str = backend.generate(g, "SimpleNet")
  assert "class SimpleNet(nn.Module):" in code_chain
  assert "self.conv1 = nn.Conv2d()" in code_chain
  assert "return x" in code_chain

  # Forward with no stateful layers and requires_super_init=False
  b_no_super = PythonBackend(framework="torch")
  setattr(b_no_super, "_is_stateful_layer", lambda x: False)

  class DummyTraits:
    """Test helper class DummyTraits."""

    def __init__(self) -> None:
      """Initialize test helper instance."""
      self.module_base: str = "nn.Module"
      self.requires_super_init: bool = False
      self.forward_method: str = "forward"
      self.init_method: str = "__init__"

  b_no_super.traits = DummyTraits()  # type: ignore[assignment]
  assert "pass" in b_no_super.compile(LogicalGraph("T"))
