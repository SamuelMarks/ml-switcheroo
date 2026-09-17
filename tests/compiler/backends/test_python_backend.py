"""Test suite for the Python Backend module."""

import ast
import typing
from typing import Any, Optional

import libcst as cst
import pytest

from ml_switcheroo.core.compiler.backends.python import ClassBodyReplacer, PythonBackend
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def backend() -> PythonBackend:
  """Docstring."""
  return PythonBackend()


def validate_python(code: str) -> None:
  """Validates python."""
  try:
    ast.parse(code)
  except SyntaxError as e:
    pytest.fail(f"Generated Invalid Python:\n{e}\n\nCode:\n{code}")


def test_compile_interface_implementation(backend: PythonBackend) -> None:
  """Compiles interface implementation."""
  g = LogicalGraph()
  res: str = backend.compile(g)
  assert isinstance(res, str)
  assert "class Model" in res


def test_synthesize_torch_chain(backend: PythonBackend) -> None:
  """Verifies the behavior of synthesize PyTorch chain."""
  g = LogicalGraph(
    nodes={n.id: n for n in [LogicalNode("x", "Input"), LogicalNode("conv1", "Conv2d"), LogicalNode("output", "Output")]},
    edges=[LogicalEdge("x", "conv1"), LogicalEdge("conv1", "output")],
  )
  code: str = backend.generate(g, "SimpleNet")
  validate_python(code)
  assert "import torch" in code
  assert "class SimpleNet(nn.Module):" in code
  assert "self.conv1 = nn.Conv2d()" in code
  assert "return x" in code


def test_synthesize_flax_chain() -> None:
  """Verifies the behavior of synthesize Flax chain."""
  backend = PythonBackend(framework="flax_nnx")
  g = LogicalGraph(nodes={n.id: n for n in [LogicalNode("x", "Input"), LogicalNode("fc", "Linear", {"out": "10"})]})
  code: str = backend.generate(g, "FlaxNet")
  validate_python(code)
  assert "class FlaxNet(nnx.Module):" in code
  assert "self.fc = nnx.Linear(out=10, rngs=rngs)" in code


def test_context_preservation(backend: PythonBackend) -> None:
  """Verifies the behavior of context preservation."""
  orig: str = "class MyNet(nn.Module): pass"
  tree: cst.Module = cst.parse_module(orig)
  g = LogicalGraph(nodes={n.id: n for n in [LogicalNode("x", "Input")]})
  code: str = backend.generate(g, class_name="MyNet", original_tree=tree)
  validate_python(code)
  assert "class MyNet" in code


def test_python_backend_sharding() -> None:
  """Verifies the behavior of python backend sharding."""
  from ml_switcheroo.core.compiler.ir import PartitionSpec

  graph = LogicalGraph(
    name="ShardedNet",
    nodes={
      "x": LogicalNode(id="x", op_type="Input"),
      "fc1": LogicalNode(
        id="fc1", op_type="Linear", sharding=PartitionSpec(axes=("data", ("model", "tensor"))), inputs=["x"]
      ),
      "out": LogicalNode(id="out", op_type="Output", inputs=["fc1"]),
    },
  )
  backend = PythonBackend(framework="flax_nnx")
  code: str = backend.compile(graph)
  assert "jax.lax.with_sharding_constraint" in code
  assert "jax.sharding.PartitionSpec('data', ('model', 'tensor'))" in code


def test_python_backend_sharding_none() -> None:
  """Verifies the behavior of python backend sharding none."""
  from ml_switcheroo.core.compiler.ir import PartitionSpec

  graph = LogicalGraph(
    name="ShardedNet",
    nodes={
      "x": LogicalNode(id="x", op_type="Input"),
      "fc1": LogicalNode(id="fc1", op_type="Linear", sharding=PartitionSpec(axes=(None, "tensor")), inputs=["x"]),
      "out": LogicalNode(id="out", op_type="Output", inputs=["fc1"]),
    },
  )
  backend = PythonBackend(framework="jax")
  code: str = backend.compile(graph)
  assert "jax.lax.with_sharding_constraint" in code
  assert "jax.sharding.PartitionSpec(None, 'tensor')" in code


def test_python_backend_sharding_torch() -> None:
  """Verifies the behavior of python backend sharding PyTorch."""
  from ml_switcheroo.core.compiler.ir import PartitionSpec

  graph = LogicalGraph(
    name="ShardedNet",
    nodes={
      "x": LogicalNode(id="x", op_type="Input"),
      "fc1": LogicalNode(id="fc1", op_type="Linear", sharding=PartitionSpec(axes=("data", None)), inputs=["x"]),
      "out": LogicalNode(id="out", op_type="Output", inputs=["fc1"]),
    },
  )
  backend = PythonBackend(framework="torch")
  code: str = backend.compile(graph)
  assert "distribute_tensor" in code
  assert "Shard(0)" in code
  assert "Replicate()" in code


def test_python_backend_sharding_tf_mlx() -> None:
  """Verifies the behavior of python backend sharding tf MLX."""
  from ml_switcheroo.core.compiler.ir import PartitionSpec

  graph = LogicalGraph(
    name="ShardedNet",
    nodes={
      "x": LogicalNode(id="x", op_type="Input"),
      "fc1": LogicalNode(
        id="fc1", op_type="Linear", sharding=PartitionSpec(axes=("data", ("tensor", "model"))), inputs=["x"]
      ),
      "out": LogicalNode(id="out", op_type="Output", inputs=["fc1"]),
    },
  )
  backend = PythonBackend(framework="tensorflow")
  code_tf: str = backend.compile(graph)
  assert "keras.distribution.layout" in code_tf
  backend_mlx = PythonBackend(framework="mlx")
  code_mlx: str = backend_mlx.compile(graph)
  assert "mx.distributed.shard" in code_mlx


def test_python_backend_primitive_mapping_mlx() -> None:
  """Verifies the behavior of python backend primitive mapping MLX."""
  from ml_switcheroo.core.compiler.backends.python import PythonBackend
  from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode

  graph = LogicalGraph()
  graph.nodes = [
    LogicalNode(id="rope", op_type="RoPE"),
    LogicalNode(id="vision", op_type="VisionPatchEmbedding"),
    LogicalNode(id="swiglu", op_type="SwiGLU"),
  ]
  backend = PythonBackend(framework="mlx")
  code: str = backend.compile(graph)
  assert "self.rope = nn.RoPE()" in code
  assert "self.vision = nn.Conv2d()" in code
  assert "self.swiglu = nn.silu()" in code


def test_python_backend_class_updater_inline_body() -> None:
  """Docstring."""
  import libcst as cst

  from ml_switcheroo.core.compiler.backends.python import ClassBodyReplacer

  # A class with inline body methods
  code: str = "class MyModel:\n  def __init__(self): pass\n  def forward(self, x): return x"
  tree: cst.Module = cst.parse_module(code)

  new_init: cst.FunctionDef = typing.cast(
    cst.FunctionDef, cst.parse_module("def __init__(self):\n  self.new_layer = 1").body[0]
  )
  new_forward: cst.FunctionDef = typing.cast(
    cst.FunctionDef, cst.parse_module("def forward(self, x):\n  return x + 1").body[0]
  )

  updater = ClassBodyReplacer("MyModel", new_init, new_forward)
  modified: cst.Module = tree.visit(updater)

  assert "self.new_layer = 1" in modified.code


def test_python_backend_class_updater_inline_body_missing_branch() -> None:
  """Docstring."""
  import libcst as cst

  from ml_switcheroo.core.compiler.backends.python import ClassBodyReplacer

  # A class with inline body that contains a SmallStatement we don't care about or that wraps safely
  code: str = "class MyModel: pass\n"
  tree: cst.Module = cst.parse_module(code)

  new_init: cst.FunctionDef = typing.cast(cst.FunctionDef, cst.parse_module("def __init__(self):\n  pass").body[0])
  new_forward: cst.FunctionDef = typing.cast(
    cst.FunctionDef, cst.parse_module("def forward(self, x):\n  return x").body[0]
  )

  updater = ClassBodyReplacer("MyModel", new_init, new_forward)
  modified: cst.Module = tree.visit(updater)

  assert "def __init__" in modified.code


def test_python_backend_class_updater_inline_body_missing_branch2() -> None:
  """Docstring."""
  import libcst as cst

  from ml_switcheroo.core.compiler.backends.python import ClassBodyReplacer

  # A class with inline body that contains a SmallStatement we don't care about (e.g., break/continue which aren't in the tuple)
  code: str = "class MyModel:\n  def __init__(self): break"
  tree: cst.Module = cst.parse_module(code)

  new_init: cst.FunctionDef = typing.cast(cst.FunctionDef, cst.parse_module("def __init__(self):\n  pass").body[0])
  new_forward: cst.FunctionDef = typing.cast(
    cst.FunctionDef, cst.parse_module("def forward(self, x):\n  return x").body[0]
  )

  updater = ClassBodyReplacer("MyModel", new_init, new_forward)
  modified: cst.Module = tree.visit(updater)

  # Ensure it doesn't crash on the missing branch for 'break' stmt
  assert "class MyModel:" in modified.code


# --- Merged from test_python_backend_missing4.py ---


def test_python_backend_base_class_resolution() -> None:
  """Docstring."""
  b = PythonBackend(framework="paxml")

  class DummyTraits:
    """Docstring."""

    def __init__(self) -> None:
      """Docstring."""
      self.module_base: str = "praxis.base_layer.BaseLayer"
      self.requires_super_init: bool = False
      self.forward_method: str = "__call__"
      self.init_method: str = "__init__"

  b.traits = DummyTraits()  # type: ignore
  assert b.compile(LogicalGraph("T"))

  b = PythonBackend(framework="keras")

  class DummyTraitsKeras:
    """Docstring."""

    def __init__(self) -> None:
      """Docstring."""
      self.module_base: str = "keras.Layer"
      self.requires_super_init: bool = False
      self.forward_method: str = "call"
      self.init_method: str = "__init__"

  b.traits = DummyTraitsKeras()  # type: ignore
  assert b.compile(LogicalGraph("T"))


def test_python_backend_forward_init_fallback() -> None:
  """Docstring."""
  b = PythonBackend(framework="torch")
  b._is_stateful = lambda x: False  # type: ignore
  b._is_stateful_layer = lambda x: False  # type: ignore

  class DummyTraits:
    """Docstring."""

    def __init__(self) -> None:
      """Docstring."""
      self.module_base: str = "nn.Module"
      self.requires_super_init: bool = False
      self.forward_method: str = "forward"
      self.init_method: str = "__init__"

  b.traits = DummyTraits()  # type: ignore

  g = LogicalGraph("T")
  c: str = b.compile(g)
  assert "pass" in c


# --- Merged from test_python_backend_missing3.py ---


def test_python_backend_is_stateful_layer_fallbacks() -> None:
  """Docstring."""
  b = PythonBackend()
  assert not b._is_stateful_layer(LogicalNode("n", "a.b.func_x"))


def test_python_backend_frameworks_base_class() -> None:
  """Docstring."""
  backend = PythonBackend(framework="paxml")

  class DummyTraits:
    """Docstring."""

    def __init__(self) -> None:
      """Docstring."""
      self.module_base: str = "praxis.base_layer.BaseLayer"
      self.requires_super_init: bool = True
      self.forward_method: str = "forward"

  backend.traits = DummyTraits()  # type: ignore
  assert backend.compile(LogicalGraph("T"))

  backend = PythonBackend(framework="keras")

  class DummyTraitsKeras:
    """Docstring."""

    def __init__(self) -> None:
      """Docstring."""
      self.module_base: str = "keras.Layer"
      self.requires_super_init: bool = True
      self.forward_method: str = "call"

  backend.traits = DummyTraitsKeras()  # type: ignore
  assert backend.compile(LogicalGraph("T"))


def test_python_backend_layer_init_resolution() -> None:
  """Docstring."""
  semantics = SemanticsManager()

  def mock_resolve(api: str, fw: str) -> Optional[dict[str, Any]]:
    """Docstring."""
    if api == "Relu":
      if fw == "torch":
        return {"api": "torch.nn.functional.relu"}
      elif fw == "mlx":
        return {"api": "mlx.core.relu"}
    if api == "Linear":
      if fw == "torch":
        return {"api": "torch.nn.Linear"}
      if fw == "mlx":
        return {"api": "mlx.nn.Linear"}
    if api == "KerasDense":
      return {"api": "keras.layers.Dense"}
    if api == "PMLX":
      return {"api": "Linear"}  # test prefix fallback
    if api == "TFLayer":
      if fw == "tensorflow":
        return {"api": "Dense"}
    if api == "MLXSwiGLU":
      return {"api": "SwiGLU"}
    return None

  semantics.resolve_variant = mock_resolve

  b = PythonBackend(framework="torch", semantics=semantics)
  n_relu = LogicalNode("n_relu", "Relu")
  res: typing.Any = b._generate_layer_init(n_relu)
  assert "nn.Relu" in cst.Module(body=[res]).code

  n_linear = LogicalNode("n_linear", "Linear")
  res2: typing.Any = b._generate_layer_init(n_linear)
  assert "nn.Linear" in cst.Module(body=[res2]).code

  b = PythonBackend(framework="mlx", semantics=semantics)
  res3: typing.Any = b._generate_layer_init(n_relu)
  assert "nn.Relu" in cst.Module(body=[res3]).code

  res4: typing.Any = b._generate_layer_init(n_linear)
  assert "nn.Linear" in cst.Module(body=[res4]).code

  b = PythonBackend(framework="keras", semantics=semantics)
  res5: typing.Any = b._generate_layer_init(LogicalNode("n1", "PMLX"))
  assert "keras.layers.Linear" in cst.Module(body=[res5]).code

  b = PythonBackend(framework="tensorflow", semantics=semantics)
  res6: typing.Any = b._generate_layer_init(LogicalNode("n1", "TFLayer"))
  assert "tf.keras.layers.Dense" in cst.Module(body=[res6]).code

  b = PythonBackend(framework="mlx", semantics=semantics)
  res7: typing.Any = b._generate_layer_init(LogicalNode("n1", "MLXSwiGLU"))
  assert "nn.silu" in cst.Module(body=[res7]).code


def test_python_backend_forward_args() -> None:
  """Docstring."""
  b = PythonBackend(framework="torch")

  def mock_is_stateful_layer(node: LogicalNode) -> bool:
    """Docstring."""
    return False

  b._is_stateful_layer = mock_is_stateful_layer

  n = LogicalNode("n", "func_x", attributes={"kwarg_a": "1"})
  g = LogicalGraph("T", nodes={n.id: n for n in [LogicalNode("i", "Input"), n]}, edges=[LogicalEdge("i", "n")])
  c: str = b.compile(g)
  assert "kwarg_a=1" in c


# --- Merged from test_python_backend_missing.py ---


def test_class_body_replacer_else_branch() -> None:
  """Docstring."""
  init_stmt = typing.cast(cst.FunctionDef, cst.parse_statement("def __init__(self): pass"))
  forward_stmt = typing.cast(cst.FunctionDef, cst.parse_statement("def forward(self): pass"))
  replacer = ClassBodyReplacer("X", init_stmt, forward_stmt)
  mod: cst.Module = cst.parse_module(
    "class X:\n    def __init__(self):\n        pass\n    def other(self):\n        print(1)\n        pass\n"
  )
  res: cst.Module = mod.visit(replacer)
  assert "print(1)" in res.code

  class DummyNode(cst.ClassDef):
    """Docstring."""

    def __init__(self) -> None:
      """Docstring."""
      super().__init__(name=cst.Name("Dummy"), body=cst.IndentedBlock(body=[]))

  d = DummyNode()
  assert replacer.leave_ClassDef(d, d) is d


def test_python_backend_imports() -> None:
  """Docstring."""
  pass


def test_python_backend_compile_forward_pass_no_stmts() -> None:
  """Docstring."""
  backend = PythonBackend(framework="torch")
  graph = LogicalGraph(name="Test", nodes={"input_0": LogicalNode(id="input_0", op_type="Input")})
  res: str = backend.compile(graph)
  assert "def forward" in res


def test_python_backend_forward_pass_abstract_resolution() -> None:
  """Docstring."""
  semantics = SemanticsManager()

  original_resolve = semantics.resolve_variant
  original_get = semantics.get_definition

  def mock_resolve(api: str, fw: str) -> typing.Optional[dict[str, typing.Any]]:
    """Docstring."""
    if api == "my_func":
      return {"api": "resolved.my_func"}
    if api == "my_abstract":
      return {"api": "resolved.my_abstract"}
    return original_resolve(api, fw)

  def mock_get(api: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Docstring."""
    if api == "func_concrete_func":
      return ("my_abstract", {})
    return original_get(api)

  semantics.resolve_variant = mock_resolve
  semantics.get_definition = mock_get

  backend = PythonBackend(framework="torch", semantics=semantics)

  # We must patch backend._is_stateful_layer because it decides functional vs object state
  def mock_is_stateful_layer(node: LogicalNode) -> bool:
    """Docstring."""
    return False

  backend._is_stateful_layer = mock_is_stateful_layer

  n0 = LogicalNode(id="n0", op_type="Input")
  n1 = LogicalNode(id="n1", op_type="my_func", inputs=["n0"])
  n2 = LogicalNode(id="n2", op_type="func_concrete_func", inputs=["n1"])
  graph = LogicalGraph(name="Test", nodes={"n0": n0, "n1": n1, "n2": n2})
  code: str = backend.compile(graph)
  assert "resolved.my_func" in code
  assert "resolved.my_abstract" in code


# --- Merged from test_python_backend_missing2.py ---


def test_python_backend_frameworks() -> None:
  """Docstring."""
  PythonBackend(framework="flax_nnx").compile(LogicalGraph("T"))

  semantics = SemanticsManager()

  def mock_resolve(api: str, fw: str) -> Optional[dict[str, Any]]:
    """Docstring."""
    if api == "Relu":
      if fw == "torch":
        return {"api": "torch.nn.functional.relu"}
      elif fw == "mlx":
        return {"api": "mlx.core.relu"}
    return None

  semantics.resolve_variant = mock_resolve

  b = PythonBackend(framework="torch", semantics=semantics)
  c: str = b.compile(LogicalGraph("T", nodes={"n1": LogicalNode("n1", op_type="Relu")}))
  assert "self.n1 = nn.Relu" in c

  b = PythonBackend(framework="mlx", semantics=semantics)
  c = b.compile(LogicalGraph("T", nodes={"n1": LogicalNode("n1", op_type="Relu")}))
  assert "self.n1 = nn.Relu" in c

  b = PythonBackend(framework="torch")
  c = b.compile(LogicalGraph("T", nodes={"n1": LogicalNode("n1", op_type="Linear")}))
  assert "self.n1 = nn.Linear" in c

  b = PythonBackend(framework="mlx")
  c = b.compile(LogicalGraph("T", nodes={"n1": LogicalNode("n1", op_type="Linear")}))
  assert "self.n1 = nn.Linear" in c

  b = PythonBackend(framework="paxml")
  c = b.compile(LogicalGraph("T", nodes={"n1": LogicalNode("n1", op_type="Linear")}))
  assert "pl.Linear" in c

  b = PythonBackend(framework="keras")
  c = b.compile(LogicalGraph("T", nodes={"n1": LogicalNode("n1", op_type="Layer")}))
  assert "self.n1 = keras.layers.Layer" in c


def test_python_backend_sharding_and_metadata() -> None:
  """Docstring."""
  b = PythonBackend(framework="torch")

  class FakeSharding:
    """Docstring."""

    def __init__(self) -> None:
      """Docstring."""
      self.axes: list[str] = ["x"]

  n = LogicalNode("n", "func_x", attributes={"kwarg_a": "1"}, sharding=FakeSharding())  # type: ignore

  def mock_is_stateful_layer(node: LogicalNode) -> bool:
    """Docstring."""
    return False

  b._is_stateful_layer = mock_is_stateful_layer

  g = LogicalGraph("T", nodes={n.id: n for n in [LogicalNode("i", "Input"), n]}, edges=[LogicalEdge("i", "n")])
  c: str = b.compile(g)
  assert "kwarg_a=1" in c
  assert "distribute_tensor" in c


def test_python_backend_sharding_jax() -> None:
  """Docstring."""
  b = PythonBackend(framework="jax")

  class FakeSharding:
    """Docstring."""

    def __init__(self) -> None:
      """Docstring."""
      self.axes: list[str] = ["x"]

  n = LogicalNode("n", "func_x", sharding=FakeSharding())  # type: ignore

  def mock_is_stateful_layer(node: LogicalNode) -> bool:
    """Docstring."""
    return False

  b._is_stateful_layer = mock_is_stateful_layer
  g = LogicalGraph("T", nodes={n.id: n for n in [LogicalNode("i", "Input"), n]}, edges=[LogicalEdge("i", "n")])
  assert "with_sharding_constraint" in b.compile(g)


def test_python_backend_sharding_keras() -> None:
  """Docstring."""
  b = PythonBackend(framework="keras")

  class FakeSharding:
    """Docstring."""

    def __init__(self) -> None:
      """Docstring."""
      self.axes: list[str] = ["x"]

  n = LogicalNode("n", "func_x", sharding=FakeSharding())  # type: ignore

  def mock_is_stateful_layer(node: LogicalNode) -> bool:
    """Docstring."""
    return False

  b._is_stateful_layer = mock_is_stateful_layer
  g = LogicalGraph("T", nodes={n.id: n for n in [LogicalNode("i", "Input"), n]}, edges=[LogicalEdge("i", "n")])
  assert "keras.distribution.layout" in b.compile(g)


def test_python_backend_sharding_mlx() -> None:
  """Docstring."""
  b = PythonBackend(framework="mlx")

  class FakeSharding:
    """Docstring."""

    def __init__(self) -> None:
      """Docstring."""
      self.axes: list[str] = ["x"]

  n = LogicalNode("n", "func_x", sharding=FakeSharding())  # type: ignore

  def mock_is_stateful_layer(node: LogicalNode) -> bool:
    """Docstring."""
    return False

  b._is_stateful_layer = mock_is_stateful_layer
  g = LogicalGraph("T", nodes={n.id: n for n in [LogicalNode("i", "Input"), n]}, edges=[LogicalEdge("i", "n")])
  assert "mx.distributed.shard" in b.compile(g)


def test_python_backend_is_stateful_layer_fallbacks_extra() -> None:
  """Docstring."""
  b = PythonBackend()
  assert not b._is_stateful_layer(LogicalNode("n", "math.add"))
  assert not b._is_stateful_layer(LogicalNode("n", "a.b.func_x"))
  assert not b._is_stateful_layer(LogicalNode("n", "math.add"))
