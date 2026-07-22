"""Test suite for the Emitter module."""

from unittest.mock import MagicMock
from ml_switcheroo.core.compiler.backends.sass.emitter import SassEmitter
from ml_switcheroo.core.compiler.frontends.sass.nodes import Instruction, Register, Comment, Label, Directive, Immediate
from ml_switcheroo.core.compiler.backends.sass.synthesizer import SassSynthesizer
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode
from ml_switcheroo.semantics.manager import SemanticsManager


def test_emit_basic_instruction():
  """Emits basic instruction."""
  emitter = SassEmitter()
  inst = Instruction(opcode="FADD", operands=[Register("R0"), Register("R1"), Register("R2")])
  output = emitter.emit([inst])
  assert output.startswith("    ")
  assert "FADD R0, R1, R2;" in output
  assert output.endswith("\n")


def test_emit_label_flush_left():
  """Emits label flush left."""
  emitter = SassEmitter()
  block = [Label("L_START"), Instruction("MOV", [Register("R0"), Register("RZ")])]
  output = emitter.emit(block)
  lines = output.strip().split("\n")
  assert lines[0] == "L_START:"
  assert lines[1] == "    MOV R0, RZ;"


def test_emit_unmapped_op_fallback():
  """Emits unmapped op fallback."""
  mgr = MagicMock(spec=SemanticsManager)
  mgr.resolve_variant.return_value = None
  mgr.get_definition.return_value = None
  synth = SassSynthesizer(mgr)
  graph = LogicalGraph(nodes=[LogicalNode(id="conv1", kind="WeirdOp", metadata={})])
  ast_nodes = synth.from_graph(graph)
  emitter = SassEmitter()
  output = emitter.emit(ast_nodes)
  assert "// Unmapped Op: WeirdOp" in output
  assert output.strip().startswith("//")


def test_emit_manual_directives():
  """Emits manual directives."""
  emitter = SassEmitter()
  nodes = [Directive("headerflags", params=["@0x100"]), Comment("Start of block")]
  output = emitter.emit(nodes)
  assert "    .headerflags @0x100" in output
  assert "    // Start of block" in output


def test_emit_immediate_values():
  """Emits immediate values."""
  emitter = SassEmitter()
  inst = Instruction("MOV", [Register("R0"), Immediate(16, is_hex=True)])
  output = emitter.emit([inst])
  assert "0x10" in output
