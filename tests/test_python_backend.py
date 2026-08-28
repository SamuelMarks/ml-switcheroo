"""Docstring."""

from unittest.mock import patch
from ml_switcheroo.core.compiler.backends.python import PythonBackend, ClassBodyReplacer
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, PartitionSpec
import libcst as cst
from typing import List, Optional


def test_class_body_replacer() -> None:
  """Docstring."""
  code: str = "class MyClass: pass"
  tree: cst.Module = cst.parse_module(code)
  new_init: cst.BaseStatement = cst.parse_statement("def __init__(self): pass\n")
  new_forward: cst.BaseStatement = cst.parse_statement("def forward(self, x): return x\n")
  replacer: ClassBodyReplacer = ClassBodyReplacer("MyClass", new_init, new_forward)
  new_tree: cst.CSTNode = tree.visit(replacer)
  assert getattr(replacer, "found", None) is True
  code_out: str = getattr(new_tree, "code", "")
  assert "def __init__(self):" in code_out


def test_class_body_replacer_existing() -> None:
  """Docstring."""
  code: str = "class MyClass:\n    def __init__(self):\n        pass\n    def forward(self, x):\n        pass\n    def other(self):\n        pass\n"
  tree: cst.Module = cst.parse_module(code)
  new_init: cst.BaseStatement = cst.parse_statement("def __init__(self): self.new = 1\n")
  new_forward: cst.BaseStatement = cst.parse_statement("def forward(self, y): return y\n")
  replacer: ClassBodyReplacer = ClassBodyReplacer("MyClass", new_init, new_forward)
  new_tree: cst.CSTNode = tree.visit(replacer)
  code_out: str = getattr(new_tree, "code", "")
  assert "self.new = 1" in code_out
  assert "return y" in code_out


def test_python_backend_init() -> None:
  """Docstring."""
  backend: PythonBackend = PythonBackend("torch")
  assert backend.framework == "torch"


def test_python_backend_compile() -> None:
  """Docstring."""
  graph: LogicalGraph = LogicalGraph(nodes=[LogicalNode(id="input", kind="Input")], edges=[])
  backend: PythonBackend = PythonBackend("torch")
  code: str = backend.compile(graph)
  assert "class" in code


def test_python_backend_generate_with_tree() -> None:
  """Docstring."""
  graph: LogicalGraph = LogicalGraph(nodes=[LogicalNode(id="input", kind="Input")], edges=[])
  backend: PythonBackend = PythonBackend("torch")
  tree: cst.Module = cst.parse_module("class CustomClass: pass")
  code: str = backend.generate(graph, class_name="CustomClass", original_tree=tree)
  assert "def __init__(self)" in code


def test_is_stateful_layer() -> None:
  """Docstring."""
  backend: PythonBackend = PythonBackend("torch")
  assert backend._is_stateful_layer(LogicalNode(id="n1", kind="Input")) is False
  assert backend._is_stateful_layer(LogicalNode(id="n2", kind="Output")) is False
  assert backend._is_stateful_layer(LogicalNode(id="n3", kind="torch.add")) is False
  assert backend._is_stateful_layer(LogicalNode(id="n4", kind="Linear")) is True
  assert backend._is_stateful_layer(LogicalNode(id="n5", kind="nn.Linear")) is True


def test_generate_imports_all() -> None:
  """Docstring."""
  assert len(PythonBackend("torch")._generate_imports()) == 2
  assert len(PythonBackend("jax")._generate_imports()) == 2
  assert len(PythonBackend("mlx")._generate_imports()) == 2
  assert len(PythonBackend("keras")._generate_imports()) == 2
  assert len(PythonBackend("paxml")._generate_imports()) == 3
  assert len(PythonBackend("unknown")._generate_imports()) == 0


def test_format_partition_spec() -> None:
  """Docstring."""
  backend: PythonBackend = PythonBackend("jax")
  ps: PartitionSpec = PartitionSpec(axes=("data", None))
  assert backend._format_partition_spec(ps) == "jax.sharding.PartitionSpec('data', None)"
  ps2: PartitionSpec = PartitionSpec(axes=(("data", "fsdp"), None))
  assert backend._format_partition_spec(ps2) == "jax.sharding.PartitionSpec(('data', 'fsdp'), None)"


def test_format_partition_spec_tf() -> None:
  """Docstring."""
  backend: PythonBackend = PythonBackend("tensorflow")
  ps: PartitionSpec = PartitionSpec(axes=("data", None, 1))
  assert backend._format_partition_spec_tf(ps) == "['data', None, '*']"


def test_format_partition_spec_torch() -> None:
  """Docstring."""
  backend: PythonBackend = PythonBackend("torch")
  ps: PartitionSpec = PartitionSpec(axes=("data", None))
  assert "Shard(0), Replicate()" in backend._format_partition_spec_torch(ps)


def test_format_args_from_metadata() -> None:
  """Docstring."""
  backend: PythonBackend = PythonBackend("torch")
  res: str = backend._format_args_from_metadata({"arg_1": "10", "arg_2": "'val'", "bias": "True", "other": "False"})
  assert res == "10, 'val', bias=True, other=False"


def test_build_forward_with_semantics() -> None:
  """Docstring."""
  backend: PythonBackend = PythonBackend("torch")
  nodes: List[LogicalNode] = [
    LogicalNode(id="in1", kind="Input"),
    LogicalNode(id="out1", kind="Output"),
  ]
  forward_def: cst.BaseStatement = backend._build_forward(nodes)
  assert "return in1" in cst.Module([forward_def]).code


def test_build_forward_complex() -> None:
  """Docstring."""
  backend: PythonBackend = PythonBackend("torch")
  nodes: List[LogicalNode] = [
    LogicalNode(id="in1", kind="Input"),
    LogicalNode(id="layer1", kind="Linear"),
    LogicalNode(id="op1", kind="torch.add", metadata={"arg_1": "1"}),
    LogicalNode(id="out1", kind="Output"),
  ]

  class MockSemantics:
    """Docstring."""

    def resolve_variant(self, kind: str, framework: str) -> Optional[dict]:
      """Docstring."""
      # reverse lookup fallback branch
      return None

    def get_definition(self, kind: str) -> tuple:
      """Docstring."""
      return "Abs", {"api": "torch.add"}

  backend.semantics = MockSemantics()

  forward_def: cst.BaseStatement = backend._build_forward(nodes)
  code: str = cst.Module([forward_def]).code
  assert "self.layer1(in1)" in code
  assert "torch.add(in1, 1)" in code
  assert "return in1" in code


def test_build_forward_with_semantics_reverse_lookup() -> None:
  """Docstring."""
  backend: PythonBackend = PythonBackend("torch")

  class MockSemantics:
    """Docstring."""

    def resolve_variant(self, kind: str, framework: str) -> Optional[dict]:
      """Docstring."""
      if kind == "abstract_id":
        return {"api": "resolved_target"}
      return None

    def get_definition(self, kind: str) -> Optional[tuple]:
      """Docstring."""
      if kind == "torch.concrete_api":
        return "abstract_id", {"some": "data"}
      return None

  backend.semantics = MockSemantics()

  nodes: List[LogicalNode] = [
    LogicalNode(id="in1", kind="Input"),
    LogicalNode(id="op1", kind="torch.concrete_api"),
    LogicalNode(id="out1", kind="Output"),
  ]
  forward_def: cst.BaseStatement = backend._build_forward(nodes)
  assert "resolved_target" in cst.Module([forward_def]).code


def test_build_forward_sharding() -> None:
  """Docstring."""
  nodes: List[LogicalNode] = [
    LogicalNode(id="in1", kind="Input"),
    LogicalNode(id="layer1", kind="Linear", sharding=PartitionSpec(axes=("data", None))),
  ]

  for fw in ["jax", "torch", "tensorflow", "keras", "mlx"]:
    backend: PythonBackend = PythonBackend(fw)
    forward_def: cst.BaseStatement = backend._build_forward(nodes)
    code: str = cst.Module([forward_def]).code
    if fw == "jax":
      assert "jax.lax.with_sharding_constraint" in code
    elif fw == "torch":
      assert "distribute_tensor" in code
    elif fw in ["tensorflow", "keras"]:
      assert "layout" in code
    elif fw == "mlx":
      assert "mx.distributed.shard" in code


def test_generate_layer_init() -> None:
  """Docstring."""
  backend: PythonBackend = PythonBackend("torch")

  class MockSemantics:
    """Docstring."""

    def resolve_variant(self, kind: str, framework: str) -> Optional[dict]:
      """Docstring."""
      if kind == "Linear":
        return {"api": "torch.nn.Linear"}
      elif kind == "Other":
        return {"api": "torch.nn.functional.other"}
      return None

  backend.semantics = MockSemantics()

  node: LogicalNode = LogicalNode(id="layer1", kind="Linear", metadata={"arg_1": "10"})
  stmt: cst.BaseStatement = backend._generate_layer_init(node)
  code: str = cst.Module([stmt]).code
  assert "self.layer1 = nn.Linear(10)" in code

  # testing torch with functional
  node_func: LogicalNode = LogicalNode(id="layer2", kind="Other", metadata={"arg_1": "10"})
  stmt_func: cst.BaseStatement = backend._generate_layer_init(node_func)
  code_func: str = cst.Module([stmt_func]).code
  assert "self.layer2 = nn.Other(10)" in code_func


def test_generate_layer_init_flax() -> None:
  """Docstring."""
  backend: PythonBackend = PythonBackend("flax")

  class MockSemantics:
    """Docstring."""

    def resolve_variant(self, kind: str, framework: str) -> Optional[dict]:
      """Docstring."""
      if kind == "Linear":
        return {"api": "nn.Dense"}
      return None

  backend.semantics = MockSemantics()

  node: LogicalNode = LogicalNode(id="layer1", kind="Linear", metadata={"arg_1": "10"})
  stmt: cst.BaseStatement = backend._generate_layer_init(node)
  code: str = cst.Module([stmt]).code
  assert "nn.Dense(10, rngs=rngs)" in code


def test_generate_layer_init_keras() -> None:
  """Docstring."""
  backend: PythonBackend = PythonBackend("keras")
  node: LogicalNode = LogicalNode(id="layer1", kind="Linear", metadata={"arg_1": "10"})
  stmt: cst.BaseStatement = backend._generate_layer_init(node)
  code: str = cst.Module([stmt]).code
  assert "keras.layers.Linear(10)" in code

  backend2: PythonBackend = PythonBackend("tensorflow")
  stmt2: cst.BaseStatement = backend2._generate_layer_init(node)
  code2: str = cst.Module([stmt2]).code
  assert "tf.keras.layers.Linear(10)" in code2


def test_generate_layer_init_paxml() -> None:
  """Docstring."""
  backend: PythonBackend = PythonBackend("paxml")
  node: LogicalNode = LogicalNode(id="layer1", kind="Linear", metadata={"arg_1": "10"})
  stmt: cst.BaseStatement = backend._generate_layer_init(node)
  code: str = cst.Module([stmt]).code
  assert "self.create_child" in code
  assert "pl.Linear" in code


def test_generate_layer_init_mlx() -> None:
  """Docstring."""
  backend: PythonBackend = PythonBackend("mlx")

  class MockSemantics:
    """Docstring."""

    def resolve_variant(self, kind: str, framework: str) -> Optional[dict]:
      """Docstring."""
      if kind == "SwiGLU":
        return {"api": "mlx.core.silu"}
      return None

  backend.semantics = MockSemantics()

  node: LogicalNode = LogicalNode(id="layer1", kind="VisionPatchEmbedding", metadata={"arg_1": "10"})
  stmt: cst.BaseStatement = backend._generate_layer_init(node)
  code: str = cst.Module([stmt]).code
  assert "nn.Conv2d(10)" in code

  node2: LogicalNode = LogicalNode(id="layer2", kind="RoPE", metadata={"arg_1": "10"})
  stmt2: cst.BaseStatement = backend._generate_layer_init(node2)
  code2: str = cst.Module([stmt2]).code
  assert "nn.RoPE(10)" in code2

  node3: LogicalNode = LogicalNode(id="layer3", kind="SwiGLU", metadata={})
  stmt3: cst.BaseStatement = backend._generate_layer_init(node3)
  code3: str = cst.Module([stmt3]).code
  assert "self.layer3 = nn.silu()" in code3

  node4: LogicalNode = LogicalNode(id="layer4", kind="mlx.nn.Linear", metadata={})
  stmt4: cst.BaseStatement = backend._generate_layer_init(node4)
  code4: str = cst.Module([stmt4]).code
  assert "self.layer4 = nn.Linear()" in code4


def test_generate_layer_init_swiglu() -> None:
  """Docstring."""
  backend: PythonBackend = PythonBackend("torch")
  node: LogicalNode = LogicalNode(id="swiglu", kind="SwiGLU", metadata={})
  stmt: cst.BaseStatement = backend._generate_layer_init(node)
  code: str = cst.Module([stmt]).code
  assert "self.swiglu = nn.SwiGLU()" in code


def test_generate_base_class_formatting() -> None:
  """Docstring."""
  graph: LogicalGraph = LogicalGraph(nodes=[LogicalNode(id="in1", kind="Input")], edges=[])

  class MockAdapter:
    """Docstring."""

    def __init__(self, requires_super_init: bool = True, base_class: str = "torch.nn.Module") -> None:
      """Docstring."""
      self.requires_super_init = requires_super_init
      self.module_base = base_class
      self.forward_method = "forward"

    @property
    def structural_traits(self) -> "MockAdapter":
      """Docstring."""
      return self

  with patch("ml_switcheroo.core.compiler.backends.python.get_adapter") as mock_adapter:
    mock_adapter.return_value = MockAdapter(base_class="torch.nn.Module")
    backend: PythonBackend = PythonBackend("torch")
    code: str = backend.compile(graph)
    assert "nn.Module" in code
    assert "super().__init__()" in code

    mock_adapter.return_value = MockAdapter(base_class="flax.nnx.Module", requires_super_init=False)
    backend_flax: PythonBackend = PythonBackend("flax_nnx")
    code_flax: str = backend_flax.compile(graph)
    assert "nnx.Module" in code_flax
    assert "rngs: nnx.Rngs" in code_flax

    mock_adapter.return_value = MockAdapter(base_class="praxis.base_layer.BaseLayer", requires_super_init=False)
    backend_pax: PythonBackend = PythonBackend("paxml")
    code_pax: str = backend_pax.compile(graph)
    assert "BaseLayer" in code_pax

    mock_adapter.return_value = MockAdapter(base_class="keras.Layer", requires_super_init=False)
    backend_tf: PythonBackend = PythonBackend("tensorflow")
    code_tf: str = backend_tf.compile(graph)
    assert "tf.keras.Model" in code_tf

    mock_adapter.return_value = MockAdapter(base_class="keras.Layer", requires_super_init=False)
    backend_keras: PythonBackend = PythonBackend("keras")
    code_keras: str = backend_keras.compile(graph)
    assert "keras.Model" in code_keras


def test_class_body_replacer_no_match() -> None:
  """Docstring."""
  code: str = "class OtherClass: pass"
  tree: cst.Module = cst.parse_module(code)
  new_init: cst.BaseStatement = cst.parse_statement("pass\n")
  new_forward: cst.BaseStatement = cst.parse_statement("pass\n")
  replacer: ClassBodyReplacer = ClassBodyReplacer("MyClass", new_init, new_forward)
  new_tree: cst.CSTNode = tree.visit(replacer)
  assert getattr(replacer, "found", None) is False
  assert "class OtherClass: pass" in getattr(new_tree, "code", "")


def test_build_init_stateful() -> None:
  """Docstring."""
  graph: LogicalGraph = LogicalGraph(
    nodes=[LogicalNode(id="in1", kind="Input"), LogicalNode(id="layer1", kind="Linear")], edges=[]
  )
  backend: PythonBackend = PythonBackend("torch")
  code: str = backend.compile(graph)
  assert "self.layer1 =" in code


def test_generate_layer_init_flax_nnx_dotless() -> None:
  """Docstring."""
  backend: PythonBackend = PythonBackend("flax_nnx")
  node: LogicalNode = LogicalNode(id="layer1", kind="MyCustomLayer", metadata={"arg_1": "10"})
  stmt: cst.BaseStatement = backend._generate_layer_init(node)
  code: str = cst.Module([stmt]).code
  assert "nnx.MyCustomLayer" in code
