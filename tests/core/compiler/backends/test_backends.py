"""Test suite for backends."""

import libcst as _cst
import pytest
from pydantic import JsonValue

from ml_switcheroo.core.compiler.backends.python import ClassBodyReplacer, PythonBackend
from ml_switcheroo.core.compiler.backends.python_snippet import PythonSnippetEmitter
from ml_switcheroo.core.compiler.backends.python_strategies.init import format_args_from_metadata
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, PartitionSpec

cst = _cst


def test_python_backend_class_body_replacer() -> None:
  """Verify `FunctionDef` handles python backend class body replacer as expected."""
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
  replaced_tree = tree.visit(replacer)
  assert replacer.found is True
  assert "def __init__():" in replaced_tree.code
  assert "def forward():" in replaced_tree.code
  replacer2 = ClassBodyReplacer("B", init_func, forward_func)
  unchanged_tree = tree.visit(replacer2)
  assert replacer2.found is False
  assert unchanged_tree.code == code


def test_python_backend_imports() -> None:
  """Verify `PythonBackend` emits `keras.Model` for python backend imports."""
  backend_keras = PythonBackend(framework="keras")
  assert "keras.Model" in backend_keras.compile(LogicalGraph())
  backend_mlx = PythonBackend(framework="mlx")
  graph = LogicalGraph(nodes={"x": LogicalNode(id="x", op_type="Input"), "out": LogicalNode(id="out", op_type="Output")})
  code = backend_mlx.compile(graph)
  assert "import mlx.core as mx" in code
  backend_tf = PythonBackend(framework="tensorflow")
  assert "import tensorflow as tf" in backend_tf.compile(graph)


def test_python_backend_build_init() -> None:
  """Verify `PythonBackend` emits `def __init__(self):` for python backend build init."""
  backend = PythonBackend(framework="mlx")
  graph = LogicalGraph(nodes={"x": LogicalNode(id="x", op_type="Input")})
  code = backend.compile(graph)
  assert "def __init__(self):" in code


def test_python_backend_build_forward() -> None:
  """Verify `PythonBackend` emits `[None, 'a', '*']` for python backend build forward."""
  backend = PythonBackend(framework="jax")
  graph = LogicalGraph(
    nodes={
      "x": LogicalNode(id="x", op_type="Input"),
      "conv": LogicalNode(id="conv", op_type="Conv2d", sharding=PartitionSpec(axes=(("a", "b"), "c"))),
      "out": LogicalNode(id="out", op_type="Output"),
    }
  )
  code = backend.compile(graph)
  assert "jax.sharding.PartitionSpec(('a', 'b'), 'c')" in code
  backend_tf = PythonBackend(framework="tensorflow")
  graph.nodes["conv"].sharding = PartitionSpec(axes=(None, "a", ("b", "c")))
  code_tf = backend_tf.compile(graph)
  assert "[None, 'a', '*']" in code_tf


def test_python_snippet_emitter() -> None:
  """Verify `PythonSnippetEmitter` emits `pass` for python snippet emitter."""
  emitter = PythonSnippetEmitter(framework="torch")
  node_input = LogicalNode("x", "Input")
  stmt1 = emitter.emit_init(node_input)
  assert "pass" in cst.Module(body=[stmt1]).code
  node_pass = LogicalNode("x", "Input")
  stmt2 = emitter.emit_call(node_pass, ["y"], "x")
  assert "x = y" in cst.Module(body=[stmt2]).code
  stmt3 = emitter.emit_call(node_pass, ["x"], "x")
  assert "pass" in cst.Module(body=[stmt3]).code
  node_bad = LogicalNode("bad", "1bad_name")
  expr = emitter.emit_expression(node_bad, [])
  assert cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=expr)])]).code.strip() == "None"
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
  node_func = LogicalNode("f", "func_relu", attributes={"arg_0": "True", "dim": 1})
  expr_func = emitter.emit_expression(node_func, ["x"])
  assert "relu(x,True,dim=1)" in cst.Module(
    body=[cst.SimpleStatementLine(body=[cst.Expr(value=expr_func)])]
  ).code.strip().replace(" ", "")


def test_python_backend_class_body_replacer_methods() -> None:
  """Verify `FunctionDef` handles python backend class body replacer methods as expected."""
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
  assert "def other(self):" in new_tree.code
  code2 = "class A:\n    def __init__(self):\n        pass\n    def __init__(self):\n        pass"
  tree2 = cst.parse_module(code2)
  replacer2 = ClassBodyReplacer("A", init_func, forward_func)
  tree2.visit(replacer2)


def test_python_backend_functional_nodes() -> None:
  """Verify `PythonBackend` emits `relu(x, True, inplace=True)` for python backend functional nodes."""
  backend = PythonBackend(framework="torch")
  graph = LogicalGraph(
    nodes={
      "x": LogicalNode(id="x", op_type="Input"),
      "relu": LogicalNode(id="relu", op_type="torch.relu", attributes={"arg_0": "True", "inplace": "True"}, inputs=["x"]),
      "out": LogicalNode(id="out", op_type="Output", inputs=["relu"]),
    }
  )
  code = backend.compile(graph)
  assert "relu(x, True, inplace=True)" in code


def test_python_backend_is_stateful_layer() -> None:
  """Verify `PythonBackend` handles python backend is stateful layer as expected."""
  backend = PythonBackend(framework="torch")
  assert not backend._is_stateful_layer(LogicalNode("o", "torch.relu"))


def test_python_backend_generate_layer_init_mlx() -> None:
  """Verify `PythonBackend` handles python backend generate layer init MLX as expected."""
  backend = PythonBackend(framework="mlx")
  graph = LogicalGraph(
    nodes={
      "x": LogicalNode(id="x", op_type="Input"),
      "fc": LogicalNode(id="fc", op_type="Linear", inputs=["x"]),
      "out": LogicalNode(id="out", op_type="Output", inputs=["fc"]),
    }
  )
  code = backend.compile(graph)
  assert "self.fc = nn.Linear()" in code


def test_python_backend_format_args() -> None:
  """Verify `PythonBackend` handles python backend format args as expected."""
  PythonBackend()
  assert format_args_from_metadata({"arg_0": "val", "k": "v"}) == "val, k=v"


def test_python_snippet_emitter_gap() -> None:
  """Verify `PythonSnippetEmitter` handles python snippet emitter gap as expected."""
  emitter = PythonSnippetEmitter(framework="unknown_fw")
  assert emitter._resolve_api_name("relu") == "relu"
  assert emitter._build_args_from_metadata({}) == []


def test_python_backend_unknown_fw_import() -> None:
  """Verify `PythonBackend` handles python backend unknown fw import as expected."""
  from ml_switcheroo.core.compiler.backends.python import PythonBackend

  backend = PythonBackend(framework="unknown_fw")
  assert backend.framework == "unknown_fw"


def test_python_backend_keras_layer_kind() -> None:
  """Verify `PythonBackend` handles python backend keras layer kind as expected."""
  from ml_switcheroo.core.compiler.backends.python import PythonBackend
  from ml_switcheroo.core.graph import LogicalNode

  backend = PythonBackend(framework="keras")
  node = LogicalNode("test", "Dense")
  init_stmt = backend._generate_layer_init(node)
  assign_keras = cst.ensure_type(init_stmt.body[0], cst.Assign)
  call_keras = cst.ensure_type(assign_keras.value, cst.Call)
  func_keras = cst.ensure_type(call_keras.func, cst.Attribute)
  layers_attr = cst.ensure_type(func_keras.value, cst.Attribute)
  assert cst.ensure_type(layers_attr.value, cst.Name).value == "keras"
  assert layers_attr.attr.value == "layers"
  assert func_keras.attr.value == "Dense"
  backend_torch = PythonBackend(framework="torch")
  init_stmt_torch = backend_torch._generate_layer_init(node)
  assign_torch = cst.ensure_type(init_stmt_torch.body[0], cst.Assign)
  call_torch = cst.ensure_type(assign_torch.value, cst.Call)
  func_torch = cst.ensure_type(call_torch.func, cst.Attribute)
  assert cst.ensure_type(func_torch.value, cst.Name).value == "nn"
  backend_jax = PythonBackend(framework="jax")
  init_stmt_jax = backend_jax._generate_layer_init(node)
  assign_jax = cst.ensure_type(init_stmt_jax.body[0], cst.Assign)
  call_jax = cst.ensure_type(assign_jax.value, cst.Call)
  func_jax = cst.ensure_type(call_jax.func, cst.Attribute)
  assert cst.ensure_type(func_jax.value, cst.Name).value == "nnx"
  backend_mlx = PythonBackend(framework="mlx")
  init_stmt_mlx = backend_mlx._generate_layer_init(node)
  assign_mlx = cst.ensure_type(init_stmt_mlx.body[0], cst.Assign)
  call_mlx = cst.ensure_type(assign_mlx.value, cst.Call)
  func_mlx = cst.ensure_type(call_mlx.func, cst.Attribute)
  assert cst.ensure_type(func_mlx.value, cst.Name).value == "nn"


def test_rdna_macros_linear() -> None:
  """Verify `RegisterAllocator` emits `Linear` for RDNA macros linear."""
  from ml_switcheroo.core.compiler.backends.rdna.macros import expand_linear
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaVGPR
  from ml_switcheroo.core.compiler.isa.allocator import GenericRegisterAllocator as RegisterAllocator

  allocator = RegisterAllocator(256, lambda x: RdnaVGPR(index=x))
  nodes = expand_linear(allocator, "test_lin", {"in_features": 64, "bias": True})
  assert len(nodes) > 10
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaBackend
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode

  backend = RdnaBackend()
  node_unmap = LogicalNode("unmap", "this_op_does_not_exist_in_the_universe", attributes={"arg_1": "v1", "arg_2": "v2"})
  node_layer = LogicalNode("lin", "Linear", attributes={"in_features": 64})
  graph = LogicalGraph(nodes={n.id: n for n in [node_unmap, node_layer]}, edges=[])
  code = backend.compile(graph)
  assert "Linear" in code
  assert "Unmapped Op:" in code


@pytest.mark.skip(reason="torch.add removed")
def test_rdna_synthesizer_gaps() -> None:
  """Verify `RegisterAllocator` raises `ValueError` during RDNA synthesizer gaps."""
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer
  from ml_switcheroo.core.compiler.isa.allocator import GenericRegisterAllocator as RegisterAllocator
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode

  alloc = RegisterAllocator(256, lambda x: None)
  for _ in range(256):
    alloc.allocate_temp()

  with pytest.raises(ValueError, match="Register overflow"):
    alloc.allocate_temp()
  for _ in range(106):
    alloc.allocate_temp()
  with pytest.raises(ValueError, match="Register overflow"):
    alloc.allocate_temp()
  from ml_switcheroo.semantics.manager import SemanticsManager

  synth = RdnaSynthesizer(SemanticsManager())
  g = LogicalGraph()
  g.add_node(LogicalNode("n1", "torch.add", attributes={"arg_1": "a", "arg_2": "b"}))
  nodes = synth.from_graph(g)
  assert len(nodes) > 0
  mod = synth.to_python(nodes)
  from ml_switcheroo.core.compiler.backends.nvidia_sass.backend import NvidiaSassBackend

  NvidiaSassBackend()
  assert "v0 =" in mod.code


def test_rdna_synthesizer_py_translation() -> None:
  """Verify `RdnaSynthesizer` emits `rdna.v_add_f32` for RDNA synthesizer py translation."""
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaImmediate as Immediate
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaInstruction as Instruction
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaLabel as Label
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaMemory as Memory
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaSGPR as SGPR
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaVGPR as VGPR
  from ml_switcheroo.semantics.manager import SemanticsManager

  synth = RdnaSynthesizer(SemanticsManager())
  nodes = [
    Instruction(opcode="v_add_f32", operands=[]),
    Instruction(opcode="store_dword", operands=[VGPR(index=0, count=1), Immediate(value=5, is_hex=True)]),
    Instruction(opcode="branch", operands=[Label(name="L1")]),
    Instruction(opcode="v_mov_b32", operands=[VGPR(index=1, count=1), Immediate(value=3.14, is_hex=False)]),
    Instruction(opcode="v_mov_b32", operands=[VGPR(index=2, count=1), Immediate(value=42, is_hex=False)]),
    Instruction(opcode="s_load", operands=[SGPR(index=0, count=2), Memory(base=VGPR(index=3, count=1))]),
    Label(name="L1"),
  ]
  mod = synth.to_python(nodes)
  from ml_switcheroo.core.compiler.backends.nvidia_sass.backend import NvidiaSassBackend

  NvidiaSassBackend()
  code = mod.code
  assert "rdna.v_add_f32" in code
  assert "rdna.store_dword" in code
  assert "rdna.branch" in code
  assert "0x5" in code
  assert "3.14" in code
  assert "42" in code


def test_rdna_synthesizer_io() -> None:
  """Verify `RdnaSynthesizer` handles RDNA synthesizer I/O as expected."""
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer
  from ml_switcheroo.core.graph import LogicalEdge, LogicalGraph, LogicalNode
  from ml_switcheroo.semantics.manager import SemanticsManager

  synth = RdnaSynthesizer(SemanticsManager())
  g = LogicalGraph(
    nodes={n.id: n for n in [LogicalNode("in", "Input", attributes={"name": "x"}), LogicalNode("out", "Output")]},
    edges=[LogicalEdge("in", "out")],
  )
  nodes = synth.from_graph(g)
  assert len(nodes) > 0
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaLabelRef as LabelRef

  res = synth._convert_operand_to_py(LabelRef("[var]"))
  assert cst.ensure_type(res, cst.Name).value == "_var"


@pytest.mark.skip(reason="Tests old allocator mock")
def test_rdna_synthesizer_misc() -> None:
  """Verify `RegisterAllocator` handles RDNA synthesizer misc as expected."""
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaSGPR
  from ml_switcheroo.core.compiler.isa.allocator import GenericRegisterAllocator as RegisterAllocator
  from ml_switcheroo.core.graph import LogicalEdge, LogicalGraph, LogicalNode
  from ml_switcheroo.semantics.manager import SemanticsManager

  alloc = RegisterAllocator(256, lambda x: RdnaSGPR(index=x))
  alloc._var_to_reg_id["test"] = 0
  s = alloc.get_register("test")
  assert s.index == 0

  class MockSemantics(SemanticsManager):
    """Test double `MockSemantics` used in backends tests."""

    def __init__(self) -> None:
      """Initialize test helper instance."""
      super().__init__(load_defaults=False)

    def resolve_variant(self, node_id: str, tgt: str) -> dict[str, JsonValue] | None:
      """Mock implementation of resolve variant.

      Args:
          node_id: Input `node_id` parameter value (`str`).
          tgt: Input `tgt` parameter value (`str`).

      Returns:
          Resulting `dict[str, JsonValue] | None` from resolve variant.
      """
      return {"api": "v_add_f32"}

    def get_definition(self, kind: str) -> tuple[str, dict[str, JsonValue]] | None:
      """Mock implementation of get definition.

      Args:
          kind: Input `kind` parameter value (`str`).

      Returns:
          Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
      """
      return None

  synth = RdnaSynthesizer(MockSemantics())
  g = LogicalGraph(
    nodes={n.id: n for n in [LogicalNode("src", "src_op"), LogicalNode("dst", "dst_op")]},
    edges=[LogicalEdge("src", "dst")],
  )
  synth.from_graph(g)


def test_nvidia_sass_macros_linear() -> None:
  """Verify `RegisterAllocator` handles nvidia SASS macros linear as expected."""
  from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import expand_linear
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassRegister
  from ml_switcheroo.core.compiler.isa.allocator import GenericRegisterAllocator as RegisterAllocator

  alloc = RegisterAllocator(256, lambda x: NvidiaSassRegister(name=f"R{x}"))
  nodes = expand_linear(alloc, "test_lin", {"in_features": 64, "bias": True})
  assert len(nodes) > 10


@pytest.mark.skip(reason="Tests old allocator mock")
def test_nvidia_sass_synthesizer_gaps() -> None:
  """Verify `RegisterAllocator` raises `ValueError` during nvidia SASS synthesizer gaps."""
  from ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer import NvidiaSassSynthesizer
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment as Comment
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassImmediate as Immediate
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassInstruction as Instruction
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassLabel as Label
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassMemory as Memory
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassPredicate as Predicate
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassRegister as Register
  from ml_switcheroo.core.compiler.isa.allocator import GenericRegisterAllocator as RegisterAllocator
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode
  from ml_switcheroo.semantics.manager import SemanticsManager

  alloc = RegisterAllocator(256, lambda x: None)
  for _ in range(255):
    alloc.allocate_temp()
  import pytest

  with pytest.raises(ValueError, match="Register overflow"):
    alloc.allocate_temp()

  class MockSemantics(SemanticsManager):
    """Test double `MockSemantics` used in backends tests."""

    def __init__(self) -> None:
      """Initialize test helper instance."""
      super().__init__(load_defaults=False)

    def get_definition(self, kind: str) -> tuple[str, dict[str, JsonValue]] | None:
      """Mock implementation of get definition.

      Args:
          kind: Input `kind` parameter value (`str`).

      Returns:
          Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
      """
      return None

    def resolve_variant(self, abstract_id: str, tgt: str) -> dict[str, JsonValue] | None:
      """Mock implementation of resolve variant.

      Args:
          abstract_id: Canonical abstract operation identifier (`str`).
          tgt: Input `tgt` parameter value (`str`).

      Returns:
          Resulting `dict[str, JsonValue] | None` from resolve variant.
      """
      if abstract_id == "Missing":
        return {}
      return {"api": "FADD"}

  synth = NvidiaSassSynthesizer(MockSemantics())
  g = LogicalGraph()
  g.add_node(LogicalNode(id="n1", op_type="Missing"))
  nodes = synth.from_graph(g)
  assert len(nodes) > 0
  assert "Unmapped Op:" in str(nodes[0])
  nodes = [
    Instruction(opcode="FADD", operands=[]),
    Instruction(
      opcode="FADD",
      operands=[Register(name="R1")],
      predicate=Predicate(name="P0"),
    ),
    Comment(text="test"),
    Label(name="L1"),
    Instruction(opcode="STG", operands=[Memory(base=Register(name="R1")), Immediate(value=1)]),
    Instruction(opcode="BRA", operands=[Label(name="L1")]),
    Instruction(opcode="MOV", operands=[Register(name="R2"), Immediate(value=1, is_hex=True)]),
    Instruction(opcode="FMUL", operands=[Register(name="R3"), Immediate(value=3.14, is_hex=False)]),
    Instruction(opcode="MOV", operands=[Register(name="R4"), Register(name="R1")]),
  ]
  mod = synth.to_python(nodes)
  from ml_switcheroo.core.compiler.backends.nvidia_sass.backend import NvidiaSassBackend

  NvidiaSassBackend()
  code = mod.code
  assert "nvidia_sass.FADD" in code
  assert "Label: L1" in code
  assert "0x1" in code
  assert "3.14" in code
