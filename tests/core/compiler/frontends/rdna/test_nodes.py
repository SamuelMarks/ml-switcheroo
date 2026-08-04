"""Test suite for the Nodes module."""

import pytest
from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaOperand,
  RdnaLabelRef,
  RdnaSGPR,
  RdnaVGPR,
  c_SGPR,
  c_VGPR,
  RdnaImmediate,
  RdnaModifier,
  RdnaMemory,
  RdnaInstruction,
  RdnaLabel,
  RdnaDirective,
  RdnaComment,
)


def test_operand_base():
  """Verifies the behavior of operand base."""
  op = RdnaOperand()
  with __import__("pytest").raises(NotImplementedError):
    str(op)


def test_label_ref():
  """Verifies the behavior of label reference."""
  ref = RdnaLabelRef(name="my_label")
  assert str(ref) == "my_label"


def test_sgpr():
  """Verifies the behavior of sgpr."""
  assert str(RdnaSGPR(index=5)) == "s5"
  assert str(RdnaSGPR(index=0, count=4)) == "s[0:3]"
  assert str(c_SGPR(2)) == "s2"


def test_vgpr():
  """Verifies the behavior of vgpr."""
  assert str(RdnaVGPR(index=10)) == "v10"
  assert str(RdnaVGPR(index=5, count=2)) == "v[5:6]"
  assert str(c_VGPR(1)) == "v1"


def test_immediate():
  """Verifies the behavior of immediate."""
  assert str(RdnaImmediate(value=42)) == "42"
  assert str(RdnaImmediate(value=255, is_hex=True)) == "0xff"


def test_modifier():
  """Verifies the behavior of modifier."""
  assert str(RdnaModifier(name="glc")) == "glc"


def test_memory():
  """Verifies the behavior of memory."""
  assert str(RdnaMemory(base=RdnaVGPR(index=0))) == "v0"
  assert str(RdnaMemory(base=RdnaSGPR(index=2), offset=16)) == "s2 offset:16"


def test_instruction():
  """Verifies the behavior of instruction."""
  inst1 = RdnaInstruction(opcode="v_nop")
  assert str(inst1) == "v_nop"
  inst2 = RdnaInstruction(opcode="v_add_f32", operands=[RdnaVGPR(index=0), RdnaVGPR(index=1), RdnaImmediate(value=5)])
  assert str(inst2) == "v_add_f32 v0, v1, 5"


def test_instruction_invalid_opcode():
  """Verifies the behavior of instruction invalid opcode."""
  with pytest.raises(ValueError, match="Invalid RDNA opcode"):
    RdnaInstruction(opcode="invalid opcode")


def test_label():
  """Verifies the behavior of label."""
  lbl = RdnaLabel(name="loop_start")
  assert str(lbl) == "loop_start:"


def test_directive():
  """Verifies the behavior of directive."""
  d1 = RdnaDirective(name="text")
  assert str(d1) == ".text"
  d2 = RdnaDirective(name="global_base", params=["foo", "bar"])
  assert str(d2) == ".global_base foo, bar"


def test_comment():
  """Verifies the behavior of comment."""
  c = RdnaComment(text="a comment")
  assert str(c) == "; a comment"
