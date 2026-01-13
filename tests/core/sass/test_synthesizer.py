import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.compiler.backends.sass.synthesizer import RegisterAllocator, SassSynthesizer, MAX_REGISTERS
from ml_switcheroo.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.compiler.frontends.sass.nodes import Instruction, Register, Immediate, Comment, Label
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.compiler.backends.sass.macros import expand_conv2d, expand_linear


def test_allocator_sequential():
  alloc = RegisterAllocator()
  r1 = alloc.get_register("x")
  r2 = alloc.get_register("y")
  assert r1.name == "R0"
  assert r2.name == "R1"


def test_allocator_reuse():
  alloc = RegisterAllocator()
  r1 = alloc.get_register("x")
  r2 = alloc.get_register("x")
  assert r1.name == "R0"
  assert r2.name == "R0"


def test_allocator_overflow():
  alloc = RegisterAllocator()
  alloc._next_idx = MAX_REGISTERS + 1
  with pytest.raises(ValueError, match="Register overflow"):
    alloc.get_register("overflow")


def test_allocator_temp():
  alloc = RegisterAllocator()
  t1 = alloc.allocate_temp()
  t2 = alloc.allocate_temp()
  assert t1.name != t2.name
  assert t1.name.startswith("R")
  assert t2.name.startswith("R")


def test_allocator_reset():
  alloc = RegisterAllocator()
  alloc.get_register("x")
  assert alloc._next_idx == 1
  alloc.reset()
  assert alloc._next_idx == 0
  assert alloc._var_to_reg == {}


@pytest.fixture
def mock_semantics():
  mgr = MagicMock(spec=SemanticsManager)

  def resolve(kind, target):
    if target != "sass":
      return None
    if kind == "Add":
      return {"api": "FADD"}
    if kind == "Mul":
      return {"api": "FMUL"}
    return None

  mgr.resolve_variant.side_effect = resolve

  def get_def(kind):
    if "Conv2d" in kind:
      return ("Conv2d", {})
    if "Linear" in kind:
      return ("Linear", {})
    return None

  mgr.get_definition.side_effect = get_def
  return mgr


def test_graph_to_sass_linear_flow(mock_semantics):
  synth = SassSynthesizer(mock_semantics)
  g = LogicalGraph()
  g.nodes = [
    LogicalNode("x", "Input", {}),
    LogicalNode("y", "Input", {}),
    LogicalNode("z", "Add", {}),
  ]
  g.edges = [LogicalEdge("x", "z"), LogicalEdge("y", "z")]

  nodes = synth.from_graph(g)
  assert len(nodes) == 3
  assert isinstance(nodes[0], Comment)
  assert "Input x -> R0" in str(nodes[0])
  assert isinstance(nodes[1], Comment)
  assert "Input y -> R1" in str(nodes[1])

  inst = nodes[2]
  assert isinstance(inst, Instruction)
  assert inst.opcode == "FADD"
  assert inst.operands[0].name == "R2"
  assert inst.operands[1].name == "R0"
  assert inst.operands[2].name == "R1"


def test_graph_to_sass_unmapped_op(mock_semantics):
  synth = SassSynthesizer(mock_semantics)
  g = LogicalGraph()
  g.nodes = [LogicalNode("n1", "UnknownOp", {})]
  nodes = synth.from_graph(g)
  assert len(nodes) == 1
  assert isinstance(nodes[0], Comment)
  assert "Unmapped Op: UnknownOp" in str(nodes[0])


def test_graph_to_sass_macro_expansion(mock_semantics):
  synth = SassSynthesizer(mock_semantics)
  g = LogicalGraph()
  g.nodes = [LogicalNode("conv1", "Conv2d", {"k": 3})]
  nodes = synth.from_graph(g)
  assert len(nodes) > 10
  comments = [n.text for n in nodes if isinstance(n, Comment)]
  assert "BEGIN Conv2d (conv1)" in comments
  labels = [n.name for n in nodes if isinstance(n, Label)]
  assert any("L_KY" in l for l in labels)
  opcodes = [n.opcode for n in nodes if isinstance(n, Instruction)]
  assert "IMAD" in opcodes
  assert "FFMA" in opcodes


def test_graph_to_sass_output_node(mock_semantics):
  synth = SassSynthesizer(mock_semantics)
  g = LogicalGraph()
  g.nodes = [LogicalNode("in1", "Input", {}), LogicalNode("out1", "Output", {})]
  g.edges = [LogicalEdge("in1", "out1")]
  nodes = synth.from_graph(g)
  assert len(nodes) == 2
  assert "Return: R0" in str(nodes[1])


def test_sass_to_python_instruction():
  synth = SassSynthesizer(MagicMock())
  inst = Instruction("FADD", [Register("R0"), Register("R1"), Register("R2")])
  mod = synth.to_python([inst])
  code = mod.code
  assert "R0 = sass.FADD(R1, R2)" in code


def test_sass_to_python_immediates():
  synth = SassSynthesizer(MagicMock())
  inst = Instruction("MOV", [Register("R0"), Immediate(16, is_hex=True)])
  mod = synth.to_python([inst])
  code = mod.code
  assert "R0 = sass.MOV(0x10)" in code


def test_sass_to_python_no_dest():
  synth = SassSynthesizer(MagicMock())

  class LabelRef:
    def __str__(self):
      return "L_TARGET"

  inst = Instruction("BRA", [LabelRef()])
  mod = synth.to_python([inst])
  code = mod.code
  assert "sass.BRA('L_TARGET')" in code
  assert "=" not in code


def test_sass_to_python_complex_operand():
  synth = SassSynthesizer(MagicMock())

  class ComplexMem:
    def __str__(self):
      return "[R1 + 0x4]"

  inst = Instruction("LD", [Register("R0"), ComplexMem()])
  mod = synth.to_python([inst])
  code = mod.code
  assert "R0 = sass.LD('[R1 + 0x4]')" in code
