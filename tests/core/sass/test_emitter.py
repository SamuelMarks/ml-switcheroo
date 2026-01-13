import pytest
from unittest.mock import MagicMock

from ml_switcheroo.compiler.backends.sass.emitter import SassEmitter
from ml_switcheroo.compiler.frontends.sass.nodes import Instruction, Register, Comment, Label, Directive, Immediate
from ml_switcheroo.compiler.backends.sass.synthesizer import SassSynthesizer
from ml_switcheroo.compiler.ir import LogicalGraph, LogicalNode
from ml_switcheroo.semantics.manager import SemanticsManager


def test_emit_basic_instruction():
  """
  Requirement: Add emits `FADD R0, R1, R2;` (indented).
  """
  emitter = SassEmitter()
  inst = Instruction(opcode="FADD", operands=[Register("R0"), Register("R1"), Register("R2")])

  output = emitter.emit([inst])

  # Check indentation (4 spaces)
  assert output.startswith("    ")
  # Check syntax
  assert "FADD R0, R1, R2;" in output
  assert output.endswith("\n")


def test_emit_label_flush_left():
  """
  Requirement: Labels must be flush-left.
  """
  emitter = SassEmitter()
  block = [Label("L_START"), Instruction("MOV", [Register("R0"), Register("RZ")])]

  output = emitter.emit(block)
  lines = output.strip().split("\n")

  # Label: No indent
  assert lines[0] == "L_START:"
  # Instruction: Indented
  assert lines[1] == "    MOV R0, RZ;"


def test_emit_unmapped_op_fallback():
  """
  Requirement: If an op isn't in sass.json, output `// Op: {Name}`.
  """
  # 1. Setup Mock Semantics (Empty for 'Conv2d')
  mgr = MagicMock(spec=SemanticsManager)
  mgr.resolve_variant.return_value = None

  # 2. Synthesize Graph
  synth = SassSynthesizer(mgr)
  graph = LogicalGraph(nodes=[LogicalNode(id="conv1", kind="Conv2d", metadata={})])
  ast_nodes = synth.from_graph(graph)

  # 3. Emit
  emitter = SassEmitter()
  output = emitter.emit(ast_nodes)

  # 4. Verify Output
  assert "// Unmapped Op: Conv2d" in output
  assert output.strip().startswith("//")


def test_emit_manual_directives():
  """Verify directives are formatted correctly."""
  emitter = SassEmitter()
  nodes = [Directive("headerflags", params=["@0x100"]), Comment("Start of block")]

  output = emitter.emit(nodes)
  assert "    .headerflags @0x100" in output
  assert "    // Start of block" in output


def test_emit_immediate_values():
  """Verify immediate operands handling."""
  emitter = SassEmitter()
  inst = Instruction("MOV", [Register("R0"), Immediate(0x10, is_hex=True)])

  output = emitter.emit([inst])
  assert "0x10" in output
