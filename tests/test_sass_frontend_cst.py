"""Test module."""

import pytest
from ml_switcheroo.core.compiler.frontends.sass.cst import (
  SassNode,
  SassOperand,
  SassRegister,
  SassPredicate,
  SassImmediate,
  SassMemory,
  SassInstruction,
  SassLabel,
  SassDirective,
  SassComment,
  SassModule,
)
from ml_switcheroo.core.cst.base import Trivia


def test_sass_register():
  """Test element."""
  reg = SassRegister(name="R0")
  assert reg.to_text() == "R0"

  reg_neg = SassRegister(name="R1", negated=True)
  assert reg_neg.to_text() == "-R1"

  reg_abs = SassRegister(name="R2", absolute=True)
  assert reg_abs.to_text() == "|R2|"

  reg_neg_abs = SassRegister(name="R3", negated=True, absolute=True)
  assert reg_neg_abs.to_text() == "-|R3|"

  reg_trivia = SassRegister(name="R0")
  reg_trivia.leading_trivia = [Trivia(" ")]
  reg_trivia.trailing_trivia = [Trivia(" ")]
  assert reg_trivia.to_text() == " R0 "


def test_sass_predicate():
  """Test element."""
  pred = SassPredicate(name="P0")
  assert pred.to_text() == "P0"

  pred_neg = SassPredicate(name="P1", negated=True)
  assert pred_neg.to_text() == "!P1"

  pred_guard = SassPredicate(name="P2", is_guard=True)
  assert pred_guard.to_text() == "@P2"

  pred_all = SassPredicate(name="P3", negated=True, is_guard=True)
  assert pred_all.to_text() == "@!P3"

  pred_trivia = SassPredicate(name="P0")
  pred_trivia.leading_trivia = [Trivia(" ")]
  pred_trivia.trailing_trivia = [Trivia(" ")]
  assert pred_trivia.to_text() == " P0 "


def test_sass_immediate():
  """Test element."""
  imm_int = SassImmediate(value=42)
  assert imm_int.to_text() == "42"

  imm_hex = SassImmediate(value=15, is_hex=True)
  assert imm_hex.to_text() == "0xf"

  imm_trivia = SassImmediate(value=42)
  imm_trivia.leading_trivia = [Trivia(" ")]
  imm_trivia.trailing_trivia = [Trivia(" ")]
  assert imm_trivia.to_text() == " 42 "


def test_sass_memory():
  """Test element."""
  # Constant bank
  mem_c_bank_offset = SassMemory(base="c[0x0]", offset=4)
  assert mem_c_bank_offset.to_text() == "c[0x0][0x4]"

  mem_c_bank_no_offset = SassMemory(base="c[0x0]")
  assert mem_c_bank_no_offset.to_text() == "c[0x0][0x0]"

  # Register based
  reg = SassRegister(name="R5")
  mem_reg_offset = SassMemory(base=reg, offset=16)
  assert mem_reg_offset.to_text() == "[R5 + 0x10]"

  mem_reg_no_offset = SassMemory(base=reg)
  assert mem_reg_no_offset.to_text() == "[R5]"

  mem_trivia = SassMemory(base=reg)
  mem_trivia.leading_trivia = [Trivia(" ")]
  mem_trivia.trailing_trivia = [Trivia(" ")]
  assert mem_trivia.to_text() == " [R5] "


def test_sass_instruction():
  """Test element."""
  # Valid inst
  inst = SassInstruction(opcode="MOV", operands=[SassRegister(name="R0"), SassImmediate(value=1)])
  assert inst.to_text() == "MOV R0, 1;"

  # Validation fail
  with pytest.raises(ValueError, match="Invalid SASS opcode"):
    SassInstruction(opcode="MOV ")

  # With predicate
  pred = SassPredicate(name="P0", is_guard=True)
  inst2 = SassInstruction(opcode="MOV", operands=[SassRegister(name="R0")], predicate=pred)
  assert inst2.to_text() == "@P0 MOV R0;"

  # Test trivia logic
  pred.trailing_trivia = [Trivia(" ")]
  inst3 = SassInstruction(opcode="MOV", operands=[SassRegister(name="R0")], predicate=pred)
  assert inst3.to_text() == "@P0 MOV R0;"

  inst3.trailing_trivia = [Trivia(" // comment")]
  assert inst3.to_text() == "@P0 MOV R0 // comment"

  # Operands spacing logic
  op1 = SassRegister(name="R0")
  op2 = SassRegister(name="R1")
  op2.leading_trivia = [Trivia(" ")]
  inst4 = SassInstruction(opcode="ADD", operands=[op1, op2])
  assert inst4.to_text() == "ADD R0 R1;"


def test_sass_label():
  """Test element."""
  lbl = SassLabel(name="L0")
  assert lbl.to_text() == "L0:"

  lbl.leading_trivia = [Trivia(" ")]
  lbl.trailing_trivia = [Trivia(" ")]
  assert lbl.to_text() == " L0: "


def test_sass_directive():
  """Test element."""
  dir1 = SassDirective(name="headerflags")
  assert dir1.to_text() == ".headerflags"

  dir2 = SassDirective(name="reqntid", params=["128", "1", "1"])
  assert dir2.to_text() == ".reqntid 128, 1, 1"

  dir1.leading_trivia = [Trivia(" ")]
  dir1.trailing_trivia = [Trivia(" ")]
  assert dir1.to_text() == " .headerflags "


def test_sass_comment():
  """Test element."""
  comm = SassComment(text="test")
  assert comm.to_text() == "// test"

  comm.leading_trivia = [Trivia(" ")]
  comm.trailing_trivia = [Trivia(" ")]
  assert comm.to_text() == " // test "


def test_sass_module():
  """Test element."""
  mod = SassModule(statements=[SassLabel(name="L0")])
  assert mod.to_text() == "L0:"

  mod.leading_trivia = [Trivia(" ")]
  mod.trailing_trivia = [Trivia(" ")]
  assert mod.to_text() == " L0: "


def test_base_nodes():
  """Test element."""
  node = SassNode()
  op = SassOperand()
  assert isinstance(node, SassNode)
  assert isinstance(op, SassOperand)
