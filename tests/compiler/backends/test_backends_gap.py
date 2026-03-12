import pytest
import libcst as cst
from ml_switcheroo.compiler.backends.python import PythonBackend, ClassBodyReplacer
from ml_switcheroo.compiler.backends.python_snippet import PythonSnippetEmitter
from ml_switcheroo.compiler.ir import LogicalGraph, LogicalNode, PartitionSpec, LogicalEdge


def test_python_backend_class_body_replacer():
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
  new_tree = tree.visit(replacer)
  assert replacer.found

  # Test ClassBodyReplacer for not found class
  replacer2 = ClassBodyReplacer("B", init_func, forward_func)
  tree.visit(replacer2)
  assert not replacer2.found


def test_python_backend_imports():
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
  # coverage for empty init
  backend = PythonBackend(framework="mlx")
  graph = LogicalGraph()
  graph.nodes = [LogicalNode(id="x", kind="Input")]
  code = backend.compile(graph)
  assert "def __init__(self):" in code


def test_python_backend_build_forward():
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
  backend = PythonBackend(framework="torch")
  assert not backend._is_stateful_layer(LogicalNode("o", "torch.relu"))


def test_python_backend_generate_layer_init_mlx():
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
  backend = PythonBackend()
  assert backend._format_args_from_metadata({"arg_0": "val", "k": "v"}) == "val, k=v"


def test_python_snippet_emitter_gap():
  emitter = PythonSnippetEmitter(framework="mlx")
  assert emitter._resolve_api_name("relu") == "relu"
  assert emitter._format_args_from_metadata(None) == ""


def test_python_backend_unknown_fw_import():
  from ml_switcheroo.compiler.backends.python import PythonBackend
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode

  backend = PythonBackend(framework="unknown_fw")
  assert backend._generate_imports() == []


def test_python_backend_keras_layer_kind():
  from ml_switcheroo.compiler.backends.python import PythonBackend
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode

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
  from ml_switcheroo.compiler.backends.rdna.macros import expand_linear
  from ml_switcheroo.compiler.backends.rdna.synthesizer import RegisterAllocator

  allocator = RegisterAllocator()
  nodes = expand_linear(allocator, "test_lin", {"in_features": 64, "bias": True})
  assert len(nodes) > 10

  # 69, 81, 84 in synthesizer: invalid types/missing args
  from ml_switcheroo.compiler.backends.rdna.synthesizer import RdnaBackend
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


def test_rdna_synthesizer_gaps():
  from ml_switcheroo.compiler.backends.rdna.synthesizer import RegisterAllocator, RdnaSynthesizer
  from ml_switcheroo.compiler.frontends.rdna.nodes import VGPR, SGPR, Instruction, Label
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode

  # RegisterAllocator fallbacks
  alloc = RegisterAllocator()
  for _ in range(256):
    alloc.allocate_vector_temp()
  import pytest

  with pytest.raises(ValueError):
    v = alloc.allocate_vector_temp()

  for _ in range(106):
    alloc.allocate_scalar_temp()
  with pytest.raises(ValueError):
    s = alloc.allocate_scalar_temp()

  # 84: release scalar
  pass

  # RdnaSynthesizer compilation gaps
  from ml_switcheroo.semantics.manager import SemanticsManager

  synth = RdnaSynthesizer(SemanticsManager())
  g = LogicalGraph(nodes=[], edges=[])
  # 172-193: Unmapped node
  g.nodes.append(LogicalNode("n1", "torch.add", {"arg_1": "a", "arg_2": "b"}))
  # 201-216: rdna_nodes to cst conversion with label
  nodes = synth.from_graph(g)
  assert len(nodes) > 0
  # Also compile to python explicitly to hit 201-216 block
  mod = synth.to_python(nodes)
  from ml_switcheroo.compiler.backends.sass.synthesizer import SassBackend

  SassBackend()
  assert "v0 =" in mod.code


def test_rdna_synthesizer_py_translation():
  from ml_switcheroo.compiler.backends.rdna.synthesizer import RdnaSynthesizer
  from ml_switcheroo.compiler.frontends.rdna.nodes import Instruction, Label, Immediate, VGPR, SGPR, Memory

  from ml_switcheroo.semantics.manager import SemanticsManager

  synth = RdnaSynthesizer(SemanticsManager())

  # 201-216, 225-257, 266-281
  nodes = [
    Instruction("v_add_f32", []),  # 226
    Instruction("store_dword", [VGPR(0, 1), Immediate(5, True)]),  # 236 is_store
    Instruction("branch", [Label("L1")]),  # 236 is_branch
    Instruction("v_mov_b32", [VGPR(1, 1), Immediate(3.14, False)]),  # float immediate
    Instruction("v_mov_b32", [VGPR(2, 1), Immediate(42, False)]),  # int immediate
    Instruction("s_load", [SGPR(0, 2), Memory(VGPR(3, 1))]),  # bracket string
    Label("L1"),
  ]

  mod = synth.to_python(nodes)
  from ml_switcheroo.compiler.backends.sass.synthesizer import SassBackend

  SassBackend()
  code = mod.code
  assert "rdna.v_add_f32" in code
  assert "rdna.store_dword" in code
  assert "rdna.branch" in code
  assert "0x5" in code
  assert "3.14" in code
  assert "42" in code


def test_rdna_synthesizer_io():
  from ml_switcheroo.compiler.backends.rdna.synthesizer import RdnaSynthesizer
  from ml_switcheroo.semantics.manager import SemanticsManager
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode, LogicalEdge

  synth = RdnaSynthesizer(SemanticsManager())

  # 140-142 edges
  # 147-150 Input
  # 154-158 Output

  g = LogicalGraph(
    nodes=[LogicalNode("in", "Input", {"name": "x"}), LogicalNode("out", "Output")], edges=[LogicalEdge("in", "out")]
  )
  nodes = synth.from_graph(g)
  assert len(nodes) > 0

  # 275-276 convert operand string with bracket
  from ml_switcheroo.compiler.frontends.rdna.nodes import LabelRef

  # "LabelRef" __str__ might return `str(self.name)`? Let's use something that returns a string with `[`
  res = synth._convert_operand_to_py(LabelRef("[var]"))
  assert res.value == "_var"


def test_rdna_synthesizer_misc():
  from ml_switcheroo.compiler.backends.rdna.synthesizer import RegisterAllocator, RdnaSynthesizer
  from ml_switcheroo.semantics.manager import SemanticsManager
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode, LogicalEdge

  alloc = RegisterAllocator()
  # populate var_to_sgpr
  alloc._var_to_sgpr["test"] = 0
  s = alloc.get_scalar_register("test")
  assert s.index == 0

  synth = RdnaSynthesizer(SemanticsManager())

  # 189-190: op with a mapped variant and sources.
  # we need a node that is mapped to rdna so variant gets found. Let's mock the semantics manager.
  class MockSemantics:
    def resolve_variant(self, node_id, tgt):
      return {"api": "v_add_f32"}

    def get_definition(self, kind):
      return None

  synth = RdnaSynthesizer(MockSemantics())
  g = LogicalGraph(nodes=[LogicalNode("src", "src_op"), LogicalNode("dst", "dst_op")], edges=[LogicalEdge("src", "dst")])
  synth.from_graph(g)


def test_sass_macros_linear():
  from ml_switcheroo.compiler.backends.sass.macros import expand_linear
  from ml_switcheroo.compiler.backends.sass.synthesizer import RegisterAllocator

  alloc = RegisterAllocator()
  nodes = expand_linear(alloc, "test_lin", {"in_features": 64, "bias": True})
  assert len(nodes) > 10


def test_sass_synthesizer_gaps():
  from ml_switcheroo.compiler.backends.sass.synthesizer import RegisterAllocator, SassSynthesizer
  from ml_switcheroo.compiler.frontends.sass.nodes import Instruction, Label, Comment, Immediate, Register, Memory
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode

  alloc = RegisterAllocator()
  for _ in range(256):
    alloc.allocate_temp()
  import pytest

  with pytest.raises(ValueError):
    alloc.allocate_temp()

  class MockSemantics:
    def get_definition(self, kind):
      return None

    def resolve_variant(self, abstract_id, tgt):
      if abstract_id == "Missing":
        return {}
      return {"api": "FADD"}

  synth = SassSynthesizer(MockSemantics())
  g = LogicalGraph(nodes=[], edges=[])
  g.nodes.append(LogicalNode("n1", "Missing"))
  nodes = synth.from_graph(g)
  assert len(nodes) > 0
  assert "Unmapped Op:" in str(nodes[0])

  # 267-289: Python emission
  nodes = [
    Instruction("FADD", []),
    Instruction("FADD", [Register("R1")], predicate="P0"),
    Comment("test"),
    Label("L1"),
    Instruction("STG", [Memory(Register("R1")), Immediate(1)]),
    Instruction("BRA", [Label("L1")]),
    Instruction("MOV", [Register("R2"), Immediate(1, True)]),  # 380 Hex
    Instruction("FMUL", [Register("R3"), Immediate(3.14, False)]),  # 382 float
    Instruction("MOV", [Register("R4"), Register("R1")]),  # alphanumeric
  ]
  mod = synth.to_python(nodes)
  from ml_switcheroo.compiler.backends.sass.synthesizer import SassBackend

  SassBackend()
  code = mod.code
  assert "sass.FADD" in code
  assert "Label: L1" in code
  assert "0x1" in code
  assert "3.14" in code
