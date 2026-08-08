"""Test rdna nodes."""

import pytest
from ml_switcheroo.core.compiler.frontends.rdna.nodes import (
  RdnaNode,
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


def test_rdna_nodes():
  """Verifies the behavior of rdna nodes."""

  # Cover RdnaNode.__str__
  class DummyRdna(RdnaNode):
    def __str__(self):
      return super().__str__()

  assert str(DummyRdna()) == ""

  # Cover Operand.__str__
  class DummyOperand(Operand):
    def __str__(self):
      return super().__str__()

  assert str(DummyOperand()) == ""

  lref = LabelRef("my_label")
  assert str(lref) == "my_label"

  assert str(c_SGPR(0)) == "s0"
  sgpr2 = SGPR(0, count=2)
  assert str(sgpr2) == "s[0:1]"

  assert str(c_VGPR(1)) == "v1"
  vgpr2 = VGPR(2, count=4)
  assert str(vgpr2) == "v[2:5]"

  imm = Immediate(1.0)
  assert str(imm) == "1.0"
  imm_hex = Immediate(15, is_hex=True)
  assert str(imm_hex) == "0xf"

  mod = Modifier("abs")
  assert str(mod) == "abs"

  mem = Memory("s[0:3]")
  assert str(mem) == "s[0:3]"
  mem_off = Memory("s[0:3]", offset=16)
  assert str(mem_off) == "s[0:3] offset:16"

  inst = Instruction("v_add_f32", operands=[vgpr2, vgpr2, imm])
  assert str(inst).strip() == "v_add_f32 v[2:5], v[2:5], 1.0"

  with pytest.raises(ValueError):
    Instruction("invalid opcode ")

  vgpr_trivia = VGPR(1)
  vgpr_trivia.leading_trivia = " "
  inst_trivia = Instruction("v_add_f32", operands=[vgpr_trivia])
  assert str(inst_trivia) == "v_add_f32 v1"

  label = Label("my_label")
  assert str(label).strip() == "my_label:"

  direc = Directive("text", params=["param1"])
  assert str(direc) == ".text param1"

  comm = Comment("this is a comment")
  assert str(comm).strip() == "; this is a comment"
