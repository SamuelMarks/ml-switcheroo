"""
Tests for SASS Synthesizer and Register Allocator.

Verifies:
1.  **Register Allocation**: Sequential assignment, overflow handling, reuse.
2.  **Graph -> SASS (1:1)**: Mapping Logic, Opcode Resolution.
3.  **Graph -> SASS (Macro)**: Macro registration and expansion (Conv2d, Linear).
4.  **SASS -> Python**: AST transformation, format handling.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.sass.synthesizer import RegisterAllocator, SassSynthesizer, MAX_REGISTERS
from ml_switcheroo.core.graph import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.core.sass.nodes import Instruction, Register, Immediate, Comment, Label
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.sass.macros import expand_conv2d, expand_linear

# --- 1. Register Allocator Tests ---


def test_allocator_sequential():
  """Verify registers are handed out in order (R0, R1, ...)."""
  alloc = RegisterAllocator()
  r1 = alloc.get_register("x")
  r2 = alloc.get_register("y")

  assert r1.name == "R0"
  assert r2.name == "R1"


def test_allocator_reuse():
  """Verify idempotency for same variable name."""
  alloc = RegisterAllocator()
  r1 = alloc.get_register("x")
  r2 = alloc.get_register("x")

  assert r1.name == "R0"
  assert r2.name == "R0"


def test_allocator_overflow():
  """Verify ValueError raised when exceeding MAX_REGISTERS."""
  alloc = RegisterAllocator()
  # Fill up
  alloc._next_idx = MAX_REGISTERS + 1

  with pytest.raises(ValueError) as exc:
    alloc.get_register("overflow")
  assert "Register overflow" in str(exc.value)


def test_allocator_temp():
  """Verify temp allocation creates fresh registers."""
  alloc = RegisterAllocator()
  t1 = alloc.allocate_temp()
  t2 = alloc.allocate_temp()

  assert t1.name != t2.name
  assert t1.name.startswith("R")
  assert t2.name.startswith("R")


def test_allocator_reset():
  """Verify reset clears state."""
  alloc = RegisterAllocator()
  alloc.get_register("x")
  assert alloc._next_idx == 1

  alloc.reset()
  assert alloc._next_idx == 0
  assert alloc._var_to_reg == {}


# --- 2. Target Synthesis Tests (Graph -> SASS) ---


@pytest.fixture
def mock_semantics():
  mgr = MagicMock(spec=SemanticsManager)

  # Define resolver logic: Map 'Add' -> 'FADD', 'Mul' -> 'FMUL'
  def resolve(kind, target):
    if target != "sass":
      return None
    if kind == "Add":
      return {"api": "FADD"}
    if kind == "Mul":
      return {"api": "FMUL"}
    return None

  mgr.resolve_variant.side_effect = resolve

  # Define get_definition logic for "Conv2d" to ensure it matches Abstract ID "Conv2d"
  # This allows synthesizer to fetch ID even if node.kind is "torch.nn.Conv2d"
  def get_def(kind):
    if "Conv2d" in kind:
      return ("Conv2d", {})
    if "Linear" in kind:
      return ("Linear", {})
    return None

  mgr.get_definition.side_effect = get_def

  return mgr


def test_graph_to_sass_linear_flow(mock_semantics):
  """
  Scenario: x -> Add(y) -> z (1:1 Op Mapping)
  Graph:
    Nodes: Input(x), Input(y), Add(z)
    Edges: x->z, y->z
  Expectation:
    // Input x -> R0
    // Input y -> R1
    FADD R2, R0, R1;
  """
  synth = SassSynthesizer(mock_semantics)

  g = LogicalGraph()
  # Note: Sorting depends on edges. Add depends on Inputs.
  g.nodes = [
    LogicalNode("x", "Input", {}),
    LogicalNode("y", "Input", {}),
    LogicalNode("z", "Add", {}),
  ]
  g.edges = [LogicalEdge("x", "z"), LogicalEdge("y", "z")]

  nodes = synth.from_graph(g)

  # Analyze output Nodes
  assert len(nodes) == 3

  # 1. Comments for inputs
  assert isinstance(nodes[0], Comment)
  assert "Input x -> R0" in str(nodes[0])

  assert isinstance(nodes[1], Comment)
  assert "Input y -> R1" in str(nodes[1])

  # 2. Instruction for Add
  # Sort order ensures Inputs processed before Add
  inst = nodes[2]
  assert isinstance(inst, Instruction)
  assert inst.opcode == "FADD"

  # R2 = R0 + R1
  # Operands: [Dest(R2), Src1(R0), Src2(R1)]
  assert len(inst.operands) == 3
  assert inst.operands[0].name == "R2"  # z
  assert inst.operands[1].name == "R0"  # x
  assert inst.operands[2].name == "R1"  # y


def test_graph_to_sass_unmapped_op(mock_semantics):
  """Verify handling of ops missing from semantics and no macro."""
  synth = SassSynthesizer(mock_semantics)

  g = LogicalGraph()
  g.nodes = [LogicalNode("n1", "UnknownOp", {})]

  nodes = synth.from_graph(g)

  assert len(nodes) == 1
  assert isinstance(nodes[0], Comment)
  assert "Unmapped Op: UnknownOp" in str(nodes[0])


def test_graph_to_sass_macro_expansion(mock_semantics):
  """
  Scenario: Conv2d node triggers Macro Expansion logic.
  Expectation: Output contains loop labels and IMAD instructions instead of single op.
  """
  synth = SassSynthesizer(mock_semantics)

  # 1. Create Graph with Conv2d
  g = LogicalGraph()
  g.nodes = [LogicalNode("conv1", "Conv2d", {"k": 3})]

  # 2. Run Synthesizer
  nodes = synth.from_graph(g)

  # 3. Validation
  # Should not be 1 instruction
  assert len(nodes) > 10

  # Check for markers from macro
  comments = [n.text for n in nodes if isinstance(n, Comment)]
  assert "BEGIN Conv2d (conv1)" in comments

  # Check for labels (Loop Lables)
  labels = [n.name for n in nodes if isinstance(n, Label)]
  assert any("L_KY" in l for l in labels)

  # Check for instructions specific to macro (IMAD, LDG)
  opcodes = [n.opcode for n in nodes if isinstance(n, Instruction)]
  assert "IMAD" in opcodes
  assert "FFMA" in opcodes


def test_graph_to_sass_output_node(mock_semantics):
  """Verify Output node generates return comment."""
  synth = SassSynthesizer(mock_semantics)
  g = LogicalGraph()
  g.nodes = [LogicalNode("in1", "Input", {}), LogicalNode("out1", "Output", {})]
  g.edges = [LogicalEdge("in1", "out1")]

  nodes = synth.from_graph(g)

  # Input comment + Output comment
  assert len(nodes) == 2
  assert "Return: R0" in str(nodes[1])


# --- 3. Source Synthesis Tests (SASS -> Python) ---


def test_sass_to_python_instruction():
  """
  Input: FADD R0, R1, R2;
  Output: R0 = sass.FADD(R1, R2)
  """
  synth = SassSynthesizer(MagicMock())

  inst = Instruction("FADD", [Register("R0"), Register("R1"), Register("R2")])

  mod = synth.to_python([inst])
  code = mod.code

  assert "R0 = sass.FADD(R1, R2)" in code


def test_sass_to_python_immediates():
  """
  Input: MOV R0, 0x10;
  Output: R0 = sass.MOV(0x10)
  """
  synth = SassSynthesizer(MagicMock())

  inst = Instruction("MOV", [Register("R0"), Immediate(16, is_hex=True)])

  mod = synth.to_python([inst])
  code = mod.code

  assert "R0 = sass.MOV(0x10)" in code


def test_sass_to_python_no_dest():
  """
  Input: BRA L_TARGET;
  Output: sass.BRA('L_TARGET')
  """
  synth = SassSynthesizer(MagicMock())

  # We use a custom object that stringifies to L_TARGET to simulate LabelRef
  class LabelRef:
    def __str__(self):
      return "L_TARGET"

  inst = Instruction("BRA", [LabelRef()])

  mod = synth.to_python([inst])
  code = mod.code

  # Should be expression statement, not assignment
  assert "sass.BRA('L_TARGET')" in code
  assert "=" not in code


def test_sass_to_python_complex_operand():
  """
  Input: LD R0, [R1 + 0x4]
  Output: R0 = sass.LD('[R1 + 0x4]')
  """
  synth = SassSynthesizer(MagicMock())

  # Simulate complex Memory operand
  class ComplexMem:
    def __str__(self):
      return "[R1 + 0x4]"

  inst = Instruction("LD", [Register("R0"), ComplexMem()])

  mod = synth.to_python([inst])
  code = mod.code

  assert "R0 = sass.LD('[R1 + 0x4]')" in code
