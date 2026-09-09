"""Tests for the NVIDIA_SASS Concrete Syntax Tree nodes."""

import pytest

from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassDirective,
  NvidiaSassImmediate,
  NvidiaSassInstruction,
  NvidiaSassLabel,
  NvidiaSassMemory,
  NvidiaSassModule,
  NvidiaSassPredicate,
  NvidiaSassRegister,
)
from ml_switcheroo.core.cst.base import Trivia


def test_register_to_text() -> None:
  """Docstring."""
  reg = NvidiaSassRegister(name="R0", leading_trivia=[Trivia(" ")], trailing_trivia=[Trivia(",")])
  assert reg.to_text() == " R0,"

  neg_reg = NvidiaSassRegister(name="R1", negated=True)
  assert neg_reg.to_text() == "-R1"

  abs_reg = NvidiaSassRegister(name="R2", absolute=True)
  assert abs_reg.to_text() == "|R2|"

  neg_abs_reg = NvidiaSassRegister(name="R3", absolute=True, negated=True)
  assert neg_abs_reg.to_text() == "-|R3|"


def test_predicate_to_text() -> None:
  """Docstring."""
  pred = NvidiaSassPredicate(name="P0", is_guard=True)
  assert pred.to_text() == "@P0"

  neg_pred = NvidiaSassPredicate(name="PT", negated=True, is_guard=True, leading_trivia=[Trivia("\n")])
  assert neg_pred.to_text() == "\n@!PT"

  op_pred = NvidiaSassPredicate(name="P1")
  assert op_pred.to_text() == "P1"


def test_immediate_to_text() -> None:
  """Docstring."""
  imm_int = NvidiaSassImmediate(value=42)
  assert imm_int.to_text() == "42"

  imm_hex = NvidiaSassImmediate(value=10, is_hex=True)
  assert imm_hex.to_text() == "0xa"

  imm_float_hex = NvidiaSassImmediate(value=15.0, is_hex=True)
  assert imm_float_hex.to_text() == "0xf"


def test_memory_to_text() -> None:
  """Docstring."""
  mem_reg = NvidiaSassMemory(base=NvidiaSassRegister(name="R1"))
  assert mem_reg.to_text() == "[R1]"

  mem_reg_offset = NvidiaSassMemory(base=NvidiaSassRegister(name="R2"), offset=4)
  assert mem_reg_offset.to_text() == "[R2 + 0x4]"

  mem_const = NvidiaSassMemory(base="c[0x0]")
  assert mem_const.to_text() == "c[0x0][0x0]"

  mem_const_offset = NvidiaSassMemory(base="c[0x1]", offset=8)
  assert mem_const_offset.to_text() == "c[0x1][0x8]"


def test_instruction_to_text() -> None:
  """Docstring."""
  inst = NvidiaSassInstruction(
    leading_trivia=[Trivia("  ")],
    predicate=NvidiaSassPredicate(name="P0", is_guard=True, trailing_trivia=[Trivia(" ")]),
    opcode="FADD",
    operands=[
      NvidiaSassRegister(name="R0", leading_trivia=[Trivia(" ")]),
      NvidiaSassRegister(name="R1", leading_trivia=[Trivia(", ")]),
    ],
    trailing_trivia=[Trivia(";")],
  )
  assert inst.to_text() == "  @P0 FADD R0, R1;"


def test_instruction_to_text_no_trivia() -> None:
  """Docstring."""
  inst = NvidiaSassInstruction(
    predicate=NvidiaSassPredicate(name="P0", is_guard=True),
    opcode="FADD",
    operands=[
      NvidiaSassRegister(name="R0"),
      NvidiaSassRegister(name="R1"),
    ],
  )
  assert inst.to_text() == "@P0 FADD R0, R1;"

  inst2 = NvidiaSassInstruction(opcode="NOP")
  assert inst2.to_text() == "NOP;"


def test_instruction_invalid_opcode() -> None:
  """Docstring."""
  with pytest.raises(ValueError, match="Invalid NVIDIA_SASS opcode"):
    NvidiaSassInstruction(opcode="FADD R0")


def test_label_to_text() -> None:
  """Docstring."""
  lbl = NvidiaSassLabel(name="L_0", trailing_trivia=[Trivia("\n")])
  assert lbl.to_text() == "L_0:\n"


def test_directive_to_text() -> None:
  """Docstring."""
  dir1 = NvidiaSassDirective(name="text")
  assert dir1.to_text() == ".text"

  dir2 = NvidiaSassDirective(name="headerflags", params=["0x1", "0x2"])
  assert dir2.to_text() == ".headerflags 0x1, 0x2"


def test_comment_to_text() -> None:
  """Docstring."""
  com = NvidiaSassComment(text="this is a comment", trailing_trivia=[Trivia("\n")])
  assert com.to_text() == "// this is a comment\n"


def test_module_to_text() -> None:
  """Docstring."""
  mod = NvidiaSassModule(
    statements=[
      NvidiaSassDirective(name="text", trailing_trivia=[Trivia("\n")]),
      NvidiaSassLabel(name="L_start", trailing_trivia=[Trivia("\n")]),
      NvidiaSassInstruction(
        opcode="MOV",
        operands=[
          NvidiaSassRegister(name="R0", leading_trivia=[Trivia(" ")]),
          NvidiaSassImmediate(value=0, leading_trivia=[Trivia(", ")]),
        ],
        trailing_trivia=[Trivia(";\n")],
      ),
    ]
  )

  expected = ".text\nL_start:\nMOV R0, 0;\n"
  assert mod.to_text() == expected
