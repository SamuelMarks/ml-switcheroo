"""Test suite for the Nodes module."""

import pytest
from ml_switcheroo.core.compiler.frontends.rdna.nodes import (
  Operand,
  LabelRef,
  SGPR,
  VGPR,
  c_SGPR,
  c_VGPR,
  Immediate,
  Modifier,
  Memory,
  Instruction,
  Label,
  Directive,
  Comment,
)


def test_operand_base():
  """Verifies the behavior of operand base."""
  op = Operand()
  assert str(op) == ""


def test_label_ref():
  """Verifies the behavior of label reference."""
  ref = LabelRef("my_label")
  assert str(ref) == "my_label"


def test_sgpr():
  """Verifies the behavior of sgpr."""
  assert str(SGPR(5)) == "s5"
  assert str(SGPR(0, count=4)) == "s[0:3]"
  assert str(c_SGPR(2)) == "s2"


def test_vgpr():
  """Verifies the behavior of vgpr."""
  assert str(VGPR(10)) == "v10"
  assert str(VGPR(5, count=2)) == "v[5:6]"
  assert str(c_VGPR(1)) == "v1"


def test_immediate():
  """Verifies the behavior of immediate."""
  assert str(Immediate(42)) == "42"
  assert str(Immediate(255, is_hex=True)) == "0xff"


def test_modifier():
  """Verifies the behavior of modifier."""
  assert str(Modifier("glc")) == "glc"


def test_memory():
  """Verifies the behavior of memory."""
  assert str(Memory(base=VGPR(0))) == "v0"
  assert str(Memory(base=SGPR(2), offset=16)) == "s2 offset:16"


def test_instruction():
  """Verifies the behavior of instruction."""
  inst1 = Instruction("v_nop")
  assert str(inst1) == "v_nop"
  inst2 = Instruction("v_add_f32", [VGPR(0), VGPR(1), Immediate(5)])
  assert str(inst2) == "v_add_f32 v0, v1, 5"


def test_instruction_invalid_opcode():
  """Verifies the behavior of instruction invalid opcode."""
  with pytest.raises(ValueError, match="Invalid RDNA opcode"):
    Instruction("invalid opcode")


def test_label():
  """Verifies the behavior of label."""
  lbl = Label("loop_start")
  assert str(lbl) == "loop_start:"


def test_directive():
  """Verifies the behavior of directive."""
  d1 = Directive("text")
  assert str(d1) == ".text"
  d2 = Directive("global_base", ["foo", "bar"])
  assert str(d2) == ".global_base foo, bar"


def test_comment():
  """Verifies the behavior of comment."""
  c = Comment("a comment")
  assert str(c) == "; a comment"
