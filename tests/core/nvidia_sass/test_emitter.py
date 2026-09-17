"""Test suite for the Emitter module."""

from unittest.mock import MagicMock

from ml_switcheroo.core.compiler.backends.nvidia_sass.emitter import NvidiaSassEmitter
from ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer import NvidiaSassSynthesizer
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassDirective,
  NvidiaSassImmediate,
  NvidiaSassInstruction,
  NvidiaSassLabel,
  NvidiaSassNode,
  NvidiaSassRegister,
)
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode
from ml_switcheroo.semantics.manager import SemanticsManager


def test_emit_basic_instruction() -> None:
  """Emits basic instruction."""
  emitter = NvidiaSassEmitter()
  inst = NvidiaSassInstruction(
    opcode="FADD", operands=[NvidiaSassRegister(name="R0"), NvidiaSassRegister(name="R1"), NvidiaSassRegister(name="R2")]
  )
  output: str = emitter.emit([inst])
  assert output.startswith("    ")
  assert "FADD R0, R1, R2;" in output
  assert output.endswith("\n")


def test_emit_label_flush_left() -> None:
  """Emits label flush left."""
  emitter = NvidiaSassEmitter()
  block: list[NvidiaSassNode] = [
    NvidiaSassLabel(name="L_START"),
    NvidiaSassInstruction(opcode="MOV", operands=[NvidiaSassRegister(name="R0"), NvidiaSassRegister(name="RZ")]),
  ]
  output: str = emitter.emit(block)
  lines: list[str] = output.strip().split("\n")
  assert lines[0] == "L_START:"
  assert lines[1] == "    MOV R0, RZ;"


def test_emit_unmapped_op_fallback() -> None:
  """Emits unmapped op fallback."""
  mgr = MagicMock(spec=SemanticsManager)
  mgr.resolve_variant.return_value = None
  mgr.get_definition.return_value = None
  synth = NvidiaSassSynthesizer(mgr)
  graph = LogicalGraph(nodes={n.id: n for n in [LogicalNode(id="conv1", op_type="WeirdOp", attributes={})]})
  ast_nodes: list[NvidiaSassNode] = synth.from_graph(graph)
  emitter = NvidiaSassEmitter()
  output: str = emitter.emit(ast_nodes)
  assert "// Unmapped Op: WeirdOp" in output
  assert output.strip().startswith("//")


def test_emit_manual_directives() -> None:
  """Emits manual directives."""
  emitter = NvidiaSassEmitter()
  nodes: list[NvidiaSassNode] = [
    NvidiaSassDirective(name="headerflags", params=["@0x100"]),
    NvidiaSassComment(text="Start of block"),
  ]
  output: str = emitter.emit(nodes)
  assert "    .headerflags @0x100" in output
  assert "    // Start of block" in output


def test_emit_immediate_values() -> None:
  """Emits immediate values."""
  emitter = NvidiaSassEmitter()
  inst = NvidiaSassInstruction(
    opcode="MOV", operands=[NvidiaSassRegister(name="R0"), NvidiaSassImmediate(value=16, is_hex=True)]
  )  # type: ignore
  output: str = emitter.emit([inst])
  assert "0x10" in output
