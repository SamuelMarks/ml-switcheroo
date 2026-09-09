"""Test module."""

import pytest

from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassDirective,
  NvidiaSassImmediate,
  NvidiaSassInstruction,
  NvidiaSassLabel,
  NvidiaSassMemory,
  NvidiaSassModule,
  NvidiaSassNode,
  NvidiaSassOperand,
  NvidiaSassPredicate,
  NvidiaSassRegister,
)
from ml_switcheroo.core.cst.base import Trivia


def test_nvidia_sass_register() -> None:
  """Docstring."""
  reg: NvidiaSassRegister = NvidiaSassRegister(name="R0")
  assert reg.to_text() == "R0"

  reg_neg: NvidiaSassRegister = NvidiaSassRegister(name="R1", negated=True)
  assert reg_neg.to_text() == "-R1"

  reg_abs: NvidiaSassRegister = NvidiaSassRegister(name="R2", absolute=True)
  assert reg_abs.to_text() == "|R2|"

  reg_neg_abs: NvidiaSassRegister = NvidiaSassRegister(name="R3", negated=True, absolute=True)
  assert reg_neg_abs.to_text() == "-|R3|"

  reg_trivia: NvidiaSassRegister = NvidiaSassRegister(name="R0")
  reg_trivia.leading_trivia = [Trivia(" ")]
  reg_trivia.trailing_trivia = [Trivia(" ")]
  assert reg_trivia.to_text() == " R0 "


def test_nvidia_sass_predicate() -> None:
  """Docstring."""
  pred: NvidiaSassPredicate = NvidiaSassPredicate(name="P0")
  assert pred.to_text() == "P0"

  pred_neg: NvidiaSassPredicate = NvidiaSassPredicate(name="P1", negated=True)
  assert pred_neg.to_text() == "!P1"

  pred_guard: NvidiaSassPredicate = NvidiaSassPredicate(name="P2", is_guard=True)
  assert pred_guard.to_text() == "@P2"

  pred_all: NvidiaSassPredicate = NvidiaSassPredicate(name="P3", negated=True, is_guard=True)
  assert pred_all.to_text() == "@!P3"

  pred_trivia: NvidiaSassPredicate = NvidiaSassPredicate(name="P0")
  pred_trivia.leading_trivia = [Trivia(" ")]
  pred_trivia.trailing_trivia = [Trivia(" ")]
  assert pred_trivia.to_text() == " P0 "


def test_nvidia_sass_immediate() -> None:
  """Docstring."""
  imm_int: NvidiaSassImmediate = NvidiaSassImmediate(value=42)
  assert imm_int.to_text() == "42"

  imm_hex: NvidiaSassImmediate = NvidiaSassImmediate(value=15, is_hex=True)
  assert imm_hex.to_text() == "0xf"

  imm_float_hex: NvidiaSassImmediate = NvidiaSassImmediate(value=15.0, is_hex=True)
  assert imm_float_hex.to_text() == "0xf"

  imm_trivia: NvidiaSassImmediate = NvidiaSassImmediate(value=42)
  imm_trivia.leading_trivia = [Trivia(" ")]
  imm_trivia.trailing_trivia = [Trivia(" ")]
  assert imm_trivia.to_text() == " 42 "


def test_nvidia_sass_memory() -> None:
  """Docstring."""
  # Constant bank
  mem_c_bank_offset: NvidiaSassMemory = NvidiaSassMemory(base="c[0x0]", offset=4)
  assert mem_c_bank_offset.to_text() == "c[0x0][0x4]"

  mem_c_bank_no_offset: NvidiaSassMemory = NvidiaSassMemory(base="c[0x0]")
  assert mem_c_bank_no_offset.to_text() == "c[0x0][0x0]"

  # Register based
  reg: NvidiaSassRegister = NvidiaSassRegister(name="R5")
  mem_reg_offset: NvidiaSassMemory = NvidiaSassMemory(base=reg, offset=16)
  assert mem_reg_offset.to_text() == "[R5 + 0x10]"

  mem_reg_no_offset: NvidiaSassMemory = NvidiaSassMemory(base=reg)
  assert mem_reg_no_offset.to_text() == "[R5]"

  mem_trivia: NvidiaSassMemory = NvidiaSassMemory(base=reg)
  mem_trivia.leading_trivia = [Trivia(" ")]
  mem_trivia.trailing_trivia = [Trivia(" ")]
  assert mem_trivia.to_text() == " [R5] "


def test_nvidia_sass_instruction() -> None:
  """Docstring."""
  # Valid inst
  inst: NvidiaSassInstruction = NvidiaSassInstruction(
    opcode="MOV", operands=[NvidiaSassRegister(name="R0"), NvidiaSassImmediate(value=1)]
  )
  assert inst.to_text() == "MOV R0, 1;"

  # Validation fail
  with pytest.raises(ValueError, match="Invalid NVIDIA_SASS opcode"):
    NvidiaSassInstruction(opcode="MOV ")

  # With predicate
  pred: NvidiaSassPredicate = NvidiaSassPredicate(name="P0", is_guard=True)
  inst2: NvidiaSassInstruction = NvidiaSassInstruction(
    opcode="MOV", operands=[NvidiaSassRegister(name="R0")], predicate=pred
  )
  assert inst2.to_text() == "@P0 MOV R0;"

  # Test trivia logic
  pred.trailing_trivia = [Trivia(" ")]
  inst3: NvidiaSassInstruction = NvidiaSassInstruction(
    opcode="MOV", operands=[NvidiaSassRegister(name="R0")], predicate=pred
  )
  assert inst3.to_text() == "@P0 MOV R0;"

  inst3.trailing_trivia = [Trivia(" // comment")]
  assert inst3.to_text() == "@P0 MOV R0 // comment"

  # Operands spacing logic
  op1: NvidiaSassRegister = NvidiaSassRegister(name="R0")
  op2: NvidiaSassRegister = NvidiaSassRegister(name="R1")
  op2.leading_trivia = [Trivia(" ")]
  inst4: NvidiaSassInstruction = NvidiaSassInstruction(opcode="ADD", operands=[op1, op2])
  assert inst4.to_text() == "ADD R0 R1;"


def test_nvidia_sass_label() -> None:
  """Docstring."""
  lbl: NvidiaSassLabel = NvidiaSassLabel(name="L0")
  assert lbl.to_text() == "L0:"

  lbl.leading_trivia = [Trivia(" ")]
  lbl.trailing_trivia = [Trivia(" ")]
  assert lbl.to_text() == " L0: "


def test_nvidia_sass_directive() -> None:
  """Docstring."""
  dir1: NvidiaSassDirective = NvidiaSassDirective(name="headerflags")
  assert dir1.to_text() == ".headerflags"

  dir2: NvidiaSassDirective = NvidiaSassDirective(name="reqntid", params=["128", "1", "1"])
  assert dir2.to_text() == ".reqntid 128, 1, 1"

  dir1.leading_trivia = [Trivia(" ")]
  dir1.trailing_trivia = [Trivia(" ")]
  assert dir1.to_text() == " .headerflags "


def test_nvidia_sass_comment() -> None:
  """Docstring."""
  comm: NvidiaSassComment = NvidiaSassComment(text="test")
  assert comm.to_text() == "// test"

  comm.leading_trivia = [Trivia(" ")]
  comm.trailing_trivia = [Trivia(" ")]
  assert comm.to_text() == " // test "


def test_nvidia_sass_module() -> None:
  """Docstring."""
  mod: NvidiaSassModule = NvidiaSassModule(statements=[NvidiaSassLabel(name="L0")])
  assert mod.to_text() == "L0:"

  mod.leading_trivia = [Trivia(" ")]
  mod.trailing_trivia = [Trivia(" ")]
  assert mod.to_text() == " L0: "


def test_base_nodes() -> None:
  """Docstring."""
  node: NvidiaSassNode = NvidiaSassNode()
  op: NvidiaSassOperand = NvidiaSassOperand()
  assert isinstance(node, NvidiaSassNode)
  assert isinstance(op, NvidiaSassOperand)
