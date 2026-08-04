"""Test suite for the Rdna Emitter module."""

from ml_switcheroo.core.compiler.backends.rdna.emitter import RdnaEmitter
from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaDirective,
  RdnaImmediate,
  RdnaInstruction,
  RdnaLabel,
  RdnaModifier,
  RdnaSGPR,
  RdnaVGPR,
)


def test_emit_basic_instruction() -> None:
  """Emits basic instruction."""
  emitter = RdnaEmitter()
  inst = RdnaInstruction(opcode="v_add_f32", operands=[RdnaVGPR(index=0), RdnaVGPR(index=1), RdnaVGPR(index=2)])
  output = emitter.emit([inst])
  assert output.startswith("    ")
  assert "v_add_f32 v0, v1, v2" in output
  assert output.endswith("\n")


def test_emit_label_flush_left() -> None:
  """Emits label flush left."""
  emitter = RdnaEmitter()
  block = [
    RdnaLabel(name="L_START"),
    RdnaInstruction(opcode="s_mov_b32", operands=[RdnaSGPR(index=0), RdnaImmediate(value=0)]),
  ]
  output = emitter.emit(block)
  lines = output.strip().split("\n")
  assert lines[0] == "L_START:"
  assert lines[1].strip() == "s_mov_b32 s0, 0"
  assert lines[1].startswith("    ")


def test_emit_register_range() -> None:
  """Emits register range."""
  emitter = RdnaEmitter()
  inst = RdnaInstruction(
    opcode="s_load_dwordx4", operands=[RdnaSGPR(index=0, count=4), RdnaSGPR(index=4, count=2), RdnaImmediate(value=0)]
  )
  output = emitter.emit([inst])
  assert "s[0:3], s[4:5], 0" in output


def test_emit_modifiers() -> None:
  """Emits modifiers."""
  emitter = RdnaEmitter()
  inst = RdnaInstruction(
    opcode="buffer_load", operands=[RdnaVGPR(index=0), RdnaModifier(name="off"), RdnaModifier(name="glc")]
  )
  output = emitter.emit([inst])
  assert "v0, off, glc" in output


def test_emit_directives_and_comments() -> None:
  """Emits directives and comments."""
  emitter = RdnaEmitter()
  nodes = [RdnaDirective(name="text"), RdnaComment(text="Init")]
  output = emitter.emit(nodes)
  assert "    .text" in output
  assert "    ; Init" in output
