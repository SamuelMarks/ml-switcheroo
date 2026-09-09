"""Test suite for the Backends Gap module."""

import typing

import libcst as cst
import pytest

from ml_switcheroo.core.compiler.backends.python import ClassBodyReplacer, PythonBackend
from ml_switcheroo.core.compiler.backends.python_snippet import PythonSnippetEmitter
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode, PartitionSpec


def test_python_backend_class_body_replacer() -> None:
  """Verifies the behavior of python backend class body replacer."""
  code: str = "class A: pass"
  tree: cst.Module = cst.parse_module(code)
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
  replacer2 = ClassBodyReplacer("B", init_func, forward_func)
  tree.visit(replacer2)
  assert not replacer2.found


def test_python_backend_imports() -> None:
  """Verifies the behavior of python backend imports."""
  backend_keras = PythonBackend(framework="keras")
  assert "keras.Model" in backend_keras.compile(LogicalGraph())
  backend_mlx = PythonBackend(framework="mlx")
  graph = LogicalGraph()
  graph.nodes = [LogicalNode(id="x", kind="Input"), LogicalNode(id="out", kind="Output")]
  code: str = backend_mlx.compile(graph)
  assert "import mlx.core as mx" in code
  backend_tf = PythonBackend(framework="tensorflow")
  assert "import tensorflow as tf" in backend_tf.compile(graph)


def test_python_backend_build_init() -> None:
  """Verifies the behavior of python backend build initialization."""
  backend = PythonBackend(framework="mlx")
  graph = LogicalGraph()
  graph.nodes = [LogicalNode(id="x", kind="Input")]
  code: str = backend.compile(graph)
  assert "def __init__(self):" in code


def test_python_backend_build_forward() -> None:
  """Verifies the behavior of python backend build forward."""
  backend = PythonBackend(framework="jax")
  graph = LogicalGraph()
  graph.nodes = [
    LogicalNode(id="x", kind="Input"),
    LogicalNode(id="conv", kind="Conv2d", sharding=PartitionSpec(axes=(("a", "b"), "c"))),
    LogicalNode(id="out", kind="Output"),
  ]
  code: str = backend.compile(graph)
  assert "jax.sharding.PartitionSpec(('a', 'b'), 'c')" in code
  backend_tf = PythonBackend(framework="tensorflow")
  graph.nodes[1].sharding = PartitionSpec(axes=(None, "a", 1))
  code_tf: str = backend_tf.compile(graph)
  assert "[None, 'a', '*']" in code_tf


def test_python_snippet_emitter() -> None:
  """Verifies the behavior of python snippet emitter."""
  emitter = PythonSnippetEmitter(framework="torch")
  node_input = LogicalNode("x", "Input")
  stmt1: cst.SimpleStatementLine = emitter.emit_init(node_input)
  assert "pass" in cst.Module(body=[stmt1]).code
  node_pass = LogicalNode("x", "Input")
  stmt2: cst.SimpleStatementLine = emitter.emit_call(node_pass, ["y"], "x")
  assert "x = y" in cst.Module(body=[stmt2]).code
  stmt3: cst.SimpleStatementLine = emitter.emit_call(node_pass, ["x"], "x")
  assert "pass" in cst.Module(body=[stmt3]).code
  node_bad = LogicalNode("bad", "1bad_name")
  expr: cst.BaseExpression = emitter.emit_expression(node_bad, [])
  assert "None" == cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=expr)])]).code.strip()
  assert not emitter._is_stateful_layer(LogicalNode("func_1", "func_x"))
  assert not emitter._is_stateful_layer(LogicalNode("f", "functional.relu"))
  assert not emitter._is_stateful_layer(LogicalNode("o", "ops.add"))
  assert emitter._is_stateful_layer(LogicalNode("Conv", "Conv2d"))
  assert not emitter._is_stateful_layer(LogicalNode("lower", "relu"))
  assert emitter._resolve_api_name("func_relu") == "torch.relu"
  assert emitter._resolve_api_name("nn.Module") == "nn.Module"
  emitter_jax = PythonSnippetEmitter(framework="jax")
  assert emitter_jax._resolve_api_name("Linear") == "nnx.Linear"
  assert emitter_jax._resolve_api_name("relu") == "jnp.relu"
  emitter_keras = PythonSnippetEmitter(framework="keras")
  assert emitter_keras._resolve_api_name("Linear") == "keras.layers.Linear"
  assert emitter_keras._resolve_api_name("relu") == "keras.ops.relu"
  node_func = LogicalNode("f", "func_relu", {"arg_0": "True", "dim": 1})
  expr_func: cst.BaseExpression = emitter.emit_expression(node_func, ["x"])
  assert "relu(x,True,dim=1)" in cst.Module(
    body=[cst.SimpleStatementLine(body=[cst.Expr(value=expr_func)])]
  ).code.strip().replace(" ", "")


def test_python_backend_class_body_replacer_methods() -> None:
  """Verifies the behavior of python backend class body replacer methods."""
  code: str = "class A:\n    def __init__(self):\n        pass\n    def forward(self, x):\n        pass\n    def other(self):\n        pass"
  tree: cst.Module = cst.parse_module(code)
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
  new_tree: cst.Module = tree.visit(replacer)
  assert replacer.found
  assert "def other(self):" in new_tree.code
  code2: str = "class A:\n    def __init__(self):\n        pass\n    def __init__(self):\n        pass"
  tree2: cst.Module = cst.parse_module(code2)
  replacer2 = ClassBodyReplacer("A", init_func, forward_func)
  tree2.visit(replacer2)


def test_python_backend_functional_nodes() -> None:
  """Verifies the behavior of python backend functional nodes."""
  backend = PythonBackend(framework="torch")
  graph = LogicalGraph()
  graph.nodes = [
    LogicalNode(id="x", kind="Input"),
    LogicalNode(id="relu", kind="torch.relu", metadata={"arg_0": "True", "inplace": "True"}),
    LogicalNode(id="out", kind="Output"),
  ]
  graph.edges = [LogicalEdge("x", "relu"), LogicalEdge("relu", "out")]
  code: str = backend.compile(graph)
  assert "relu(x, True, inplace=True)" in code


def test_python_backend_is_stateful_layer() -> None:
  """Verifies the behavior of python backend is stateful layer."""
  backend = PythonBackend(framework="torch")
  assert not backend._is_stateful_layer(LogicalNode("o", "torch.relu"))


def test_python_backend_generate_layer_init_mlx() -> None:
  """Verifies the behavior of python backend generate layer initialization MLX."""
  backend = PythonBackend(framework="mlx")
  graph = LogicalGraph()
  graph.nodes = [
    LogicalNode(id="x", kind="Input"),
    LogicalNode(id="fc", kind="Linear"),
    LogicalNode(id="out", kind="Output"),
  ]
  graph.edges = [LogicalEdge("x", "fc"), LogicalEdge("fc", "out")]
  code: str = backend.compile(graph)
  assert "self.fc = nn.Linear()" in code


def test_python_backend_format_args() -> None:
  """Verifies the behavior of python backend format arguments."""
  backend = PythonBackend()
  assert backend._format_args_from_metadata({"arg_0": "val", "k": "v"}) == "val, k=v"


def test_python_snippet_emitter_gap() -> None:
  """Verifies the behavior of python snippet emitter gap."""
  emitter = PythonSnippetEmitter(framework="mlx")
  assert emitter._resolve_api_name("relu") == "relu"
  assert emitter._build_args_from_metadata({}) == []


def test_python_backend_unknown_fw_import() -> None:
  """Verifies the behavior of python backend unknown framework import."""
  from ml_switcheroo.core.compiler.backends.python import PythonBackend

  backend = PythonBackend(framework="unknown_fw")
  assert backend._generate_imports() == []


def test_python_backend_keras_layer_kind() -> None:
  """Verifies the behavior of python backend Keras layer kind."""
  import typing

  from ml_switcheroo.core.compiler.backends.python import PythonBackend
  from ml_switcheroo.core.graph import LogicalNode

  backend = PythonBackend(framework="keras")
  node = LogicalNode("test", "Dense")
  init_stmt: typing.Any = backend._generate_layer_init(node)
  assert init_stmt.body[0].value.func.value.value.value == "keras"
  assert init_stmt.body[0].value.func.value.attr.value == "layers"
  assert init_stmt.body[0].value.func.attr.value == "Dense"
  backend_torch = PythonBackend(framework="torch")
  init_stmt_torch: typing.Any = backend_torch._generate_layer_init(node)
  assert init_stmt_torch.body[0].value.func.value.value == "nn"
  backend_jax = PythonBackend(framework="jax")
  init_stmt_jax: typing.Any = backend_jax._generate_layer_init(node)
  assert init_stmt_jax.body[0].value.func.value.value == "nnx"
  backend_mlx = PythonBackend(framework="mlx")
  init_stmt_mlx: typing.Any = backend_mlx._generate_layer_init(node)
  assert init_stmt_mlx.body[0].value.func.value.value == "nn"


def test_rdna_macros_linear() -> None:
  """Verifies the behavior of RDNA macros linear."""
  from ml_switcheroo.core.compiler.backends.rdna.macros import expand_linear
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RegisterAllocator

  allocator = RegisterAllocator()
  nodes: list = expand_linear(allocator, "test_lin", {"in_features": 64, "bias": True})
  assert len(nodes) > 10
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaBackend
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode

  backend = RdnaBackend()
  pass
  node_unmap = LogicalNode("unmap", "this_op_does_not_exist_in_the_universe", {"arg_1": "v1", "arg_2": "v2"})
  node_layer = LogicalNode("lin", "Linear", {"in_features": 64})
  graph = LogicalGraph(nodes=[node_unmap, node_layer], edges=[])
  code: str = backend.compile(graph)
  assert "Linear" in code
  assert "Unmapped Op:" in code


# --- Merged from test_backends_gap2.py ---


@pytest.mark.skip(reason="torch.add removed")
def test_rdna_synthesizer_gaps() -> None:
  """Verifies the behavior of RDNA synthesizer gaps."""
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer, RegisterAllocator
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode

  alloc = RegisterAllocator()
  for _ in range(256):
    alloc.allocate_vector_temp()

  with pytest.raises(ValueError):
    alloc.allocate_vector_temp()
  for _ in range(106):
    alloc.allocate_scalar_temp()
  with pytest.raises(ValueError):
    alloc.allocate_scalar_temp()
  pass
  from ml_switcheroo.semantics.manager import SemanticsManager

  synth = RdnaSynthesizer(SemanticsManager())
  g = LogicalGraph(nodes=[], edges=[])
  g.nodes.append(LogicalNode("n1", "torch.add", {"arg_1": "a", "arg_2": "b"}))
  nodes: list[typing.Any] = synth.from_graph(g)
  assert len(nodes) > 0
  mod: typing.Any = synth.to_python(nodes)
  from ml_switcheroo.core.compiler.backends.nvidia_sass.backend import NvidiaSassBackend

  NvidiaSassBackend()
  assert "v0 =" in mod.code


def test_rdna_synthesizer_py_translation() -> None:
  """Verifies the behavior of RDNA synthesizer py translation."""
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaImmediate as Immediate
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaInstruction as Instruction
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaLabel as Label
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaMemory as Memory
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaSGPR as SGPR
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaVGPR as VGPR
  from ml_switcheroo.semantics.manager import SemanticsManager

  synth = RdnaSynthesizer(SemanticsManager())
  nodes: list[typing.Any] = [
    Instruction(opcode="v_add_f32", operands=[]),
    Instruction(opcode="store_dword", operands=[VGPR(index=0, count=1), Immediate(value=5, is_hex=True)]),
    Instruction(opcode="branch", operands=[Label(name="L1")]),
    Instruction(opcode="v_mov_b32", operands=[VGPR(index=1, count=1), Immediate(value=3.14, is_hex=False)]),
    Instruction(opcode="v_mov_b32", operands=[VGPR(index=2, count=1), Immediate(value=42, is_hex=False)]),
    Instruction(opcode="s_load", operands=[SGPR(index=0, count=2), Memory(base=VGPR(index=3, count=1))]),
    Label(name="L1"),
  ]
  mod: typing.Any = synth.to_python(nodes)
  from ml_switcheroo.core.compiler.backends.nvidia_sass.backend import NvidiaSassBackend

  NvidiaSassBackend()
  code: str = mod.code
  assert "rdna.v_add_f32" in code
  assert "rdna.store_dword" in code
  assert "rdna.branch" in code
  assert "0x5" in code
  assert "3.14" in code
  assert "42" in code


def test_rdna_synthesizer_io() -> None:
  """Verifies the behavior of RDNA synthesizer I/O."""
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer
  from ml_switcheroo.core.graph import LogicalEdge, LogicalGraph, LogicalNode
  from ml_switcheroo.semantics.manager import SemanticsManager

  synth = RdnaSynthesizer(SemanticsManager())
  g = LogicalGraph(
    nodes=[LogicalNode("in", "Input", {"name": "x"}), LogicalNode("out", "Output")], edges=[LogicalEdge("in", "out")]
  )
  nodes: list[typing.Any] = synth.from_graph(g)
  assert len(nodes) > 0
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaLabelRef as LabelRef

  res: typing.Any = synth._convert_operand_to_py(LabelRef("[var]"))
  assert res.value == "_var"


def test_rdna_synthesizer_misc() -> None:
  """Verifies the behavior of RDNA synthesizer misc."""
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer, RegisterAllocator
  from ml_switcheroo.core.graph import LogicalEdge, LogicalGraph, LogicalNode
  from ml_switcheroo.semantics.manager import SemanticsManager

  alloc = RegisterAllocator()
  alloc._var_to_sgpr["test"] = 0
  s: typing.Any = alloc.get_scalar_register("test")
  assert s.index == 0

  class MockSemantics(SemanticsManager):
    """Docstring."""

    def __init__(self) -> None:
      """Docstring."""
      pass

    def resolve_variant(self, node_id: str, tgt: str) -> dict[str, typing.Any]:
      """Mock implementation of resolve variant."""
      return {"api": "v_add_f32"}

    def get_definition(self, kind: str) -> typing.Optional[typing.Any]:
      """Mock implementation of get definition."""
      return None

  synth = RdnaSynthesizer(MockSemantics())
  g = LogicalGraph(nodes=[LogicalNode("src", "src_op"), LogicalNode("dst", "dst_op")], edges=[LogicalEdge("src", "dst")])
  synth.from_graph(g)


def test_nvidia_sass_macros_linear() -> None:
  """Verifies the behavior of NVIDIA_SASS macros linear."""
  from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import expand_linear
  from ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer import RegisterAllocator

  alloc = RegisterAllocator()
  nodes: list[typing.Any] = expand_linear(alloc, "test_lin", {"in_features": 64, "bias": True})
  assert len(nodes) > 10


def test_nvidia_sass_synthesizer_gaps() -> None:
  """Verifies the behavior of NVIDIA_SASS synthesizer gaps."""
  from ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer import RegisterAllocator, NvidiaSassSynthesizer
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment as Comment
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassImmediate as Immediate
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassInstruction as Instruction
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassLabel as Label
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassMemory as Memory
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassRegister as Register
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode
  from ml_switcheroo.semantics.manager import SemanticsManager

  alloc = RegisterAllocator()
  for _ in range(255):
    alloc.allocate_temp()
  import pytest

  with pytest.raises(ValueError):
    alloc.allocate_temp()

  class MockSemantics(SemanticsManager):
    """Docstring."""

    def __init__(self) -> None:
      """Docstring."""
      pass

    def get_definition(self, kind: str) -> typing.Optional[typing.Any]:
      """Mock implementation of get definition."""
      return None

    def resolve_variant(self, abstract_id: str, tgt: str) -> dict[str, typing.Any]:
      """Mock implementation of resolve variant."""
      if abstract_id == "Missing":
        return {}
      return {"api": "FADD"}

  synth = NvidiaSassSynthesizer(MockSemantics())
  g = LogicalGraph(nodes=[], edges=[])
  g.nodes.append(LogicalNode("n1", "Missing"))
  nodes: list[typing.Any] = synth.from_graph(g)
  assert len(nodes) > 0
  assert "Unmapped Op:" in str(nodes[0])
  nodes = [
    Instruction(opcode="FADD", operands=[]),
    Instruction(opcode="FADD", operands=[Register(name="R1")], predicate="P0"),
    Comment(text="test"),
    Label(name="L1"),
    Instruction(opcode="STG", operands=[Memory(base=Register(name="R1")), Immediate(value=1)]),
    Instruction(opcode="BRA", operands=[Label(name="L1")]),
    Instruction(opcode="MOV", operands=[Register(name="R2"), Immediate(value=1, is_hex=True)]),
    Instruction(opcode="FMUL", operands=[Register(name="R3"), Immediate(value=3.14, is_hex=False)]),
    Instruction(opcode="MOV", operands=[Register(name="R4"), Register(name="R1")]),
  ]
  mod: typing.Any = synth.to_python(nodes)
  from ml_switcheroo.core.compiler.backends.nvidia_sass.backend import NvidiaSassBackend

  NvidiaSassBackend()
  code: str = mod.code
  assert "nvidia_sass.FADD" in code
  assert "Label: L1" in code
  assert "0x1" in code
  assert "3.14" in code
