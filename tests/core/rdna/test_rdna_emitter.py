"""
Tests for RDNA Emitter formatting logic.

Verifies:
1.  **Instruction Formatting**: opcodes, operands, and indentation.
2.  **Control Flow**: Labels flush-left vs indented instructions.
3.  **Complex Operands**: Register ranges.
"""

from ml_switcheroo.core.rdna.emitter import RdnaEmitter
from ml_switcheroo.core.rdna.nodes import (
  Comment,
  Directive,
  Immediate,
  Instruction,
  Label,
  Modifier,
  SGPR,
  VGPR,
)


def test_emit_basic_instruction() -> None:
  """
  Requirement: `v_add_f32 v0, v1, v2` (indented).
  """
  emitter = RdnaEmitter()
  inst = Instruction(opcode="v_add_f32", operands=[VGPR(0), VGPR(1), VGPR(2)])

  output = emitter.emit([inst])

  # Check indentation (4 spaces)
  assert output.startswith("    ")
  # Check syntax (space or comma separated depending on impl, we chose comma in nodes.py)
  assert "v_add_f32 v0, v1, v2" in output
  assert output.endswith("\n")


def test_emit_label_flush_left() -> None:
  """
  Requirement: Labels must be flush-left.
  """
  emitter = RdnaEmitter()
  block = [
    Label("L_START"),
    Instruction("s_mov_b32", [SGPR(0), Immediate(0)]),
  ]

  output = emitter.emit(block)
  lines = output.strip().split("\n")

  # Label: No indent
  assert lines[0] == "L_START:"
  # Instruction: Indented
  assert lines[1].strip() == "s_mov_b32 s0, 0"
  assert lines[1].startswith("    ")


def test_emit_register_range() -> None:
  """
  Verify s[0:3] formatting.
  """
  emitter = RdnaEmitter()
  inst = Instruction("s_load_dwordx4", [SGPR(0, count=4), SGPR(4, count=2), Immediate(0)])
  output = emitter.emit([inst])

  assert "s[0:3], s[4:5], 0" in output


def test_emit_modifiers() -> None:
  """
  Verify modifiers are appended correctly.
  """
  emitter = RdnaEmitter()
  inst = Instruction("buffer_load", [VGPR(0), Modifier("off"), Modifier("glc")])
  output = emitter.emit([inst])

  assert "v0, off, glc" in output


def test_emit_directives_and_comments() -> None:
  """Verify comments and directives."""
  emitter = RdnaEmitter()
  nodes = [
    Directive("text"),
    Comment("Init"),
  ]
  output = emitter.emit(nodes)

  assert "    .text" in output
  assert "    ; Init" in output
