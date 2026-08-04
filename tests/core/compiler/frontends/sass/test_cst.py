"""Tests for the SASS Concrete Syntax Tree nodes."""

from ml_switcheroo.core.compiler.frontends.sass.cst import (
  SassModule,
  SassInstruction,
  SassRegister,
  SassImmediate,
  SassMemory,
  SassPredicate,
  SassLabel,
  SassDirective,
  SassComment,
)
from ml_switcheroo.core.cst.base import Trivia


def test_register_to_text() -> None:
  """Test register serialization."""
  reg = SassRegister(name="R0", leading_trivia=[Trivia(" ")], trailing_trivia=[Trivia(",")])
  assert reg.to_text() == " R0,"

  neg_reg = SassRegister(name="R1", negated=True)
  assert neg_reg.to_text() == "-R1"

  abs_reg = SassRegister(name="R2", absolute=True)
  assert abs_reg.to_text() == "|R2|"

  neg_abs_reg = SassRegister(name="R3", absolute=True, negated=True)
  assert neg_abs_reg.to_text() == "-|R3|"


def test_predicate_to_text() -> None:
  """Test predicate serialization."""
  pred = SassPredicate(name="P0", is_guard=True)
  assert pred.to_text() == "@P0"

  neg_pred = SassPredicate(name="PT", negated=True, is_guard=True, leading_trivia=[Trivia("\n")])
  assert neg_pred.to_text() == "\n@!PT"

  op_pred = SassPredicate(name="P1")
  assert op_pred.to_text() == "P1"


def test_immediate_to_text() -> None:
  """Test immediate serialization."""
  imm_int = SassImmediate(value=42)
  assert imm_int.to_text() == "42"

  imm_hex = SassImmediate(value=10, is_hex=True)
  assert imm_hex.to_text() == "0xa"

  imm_float_hex = SassImmediate(value=15.0, is_hex=True)
  assert imm_float_hex.to_text() == "0xf"


def test_memory_to_text() -> None:
  """Test memory operand serialization."""
  mem_reg = SassMemory(base=SassRegister(name="R1"))
  assert mem_reg.to_text() == "[R1]"

  mem_reg_offset = SassMemory(base=SassRegister(name="R2"), offset=4)
  assert mem_reg_offset.to_text() == "[R2 + 0x4]"

  mem_const = SassMemory(base="c[0x0]")
  assert mem_const.to_text() == "c[0x0][0x0]"

  mem_const_offset = SassMemory(base="c[0x1]", offset=8)
  assert mem_const_offset.to_text() == "c[0x1][0x8]"


def test_instruction_to_text() -> None:
  """Test instruction serialization."""
  inst = SassInstruction(
    leading_trivia=[Trivia("  ")],
    predicate=SassPredicate(name="P0", is_guard=True, trailing_trivia=[Trivia(" ")]),
    opcode="FADD",
    operands=[
      SassRegister(name="R0", leading_trivia=[Trivia(" ")]),
      SassRegister(name="R1", leading_trivia=[Trivia(", ")]),
    ],
    trailing_trivia=[Trivia(";")],
  )
  assert inst.to_text() == "  @P0 FADD R0, R1;"


def test_label_to_text() -> None:
  """Test label serialization."""
  lbl = SassLabel(name="L_0", trailing_trivia=[Trivia("\n")])
  assert lbl.to_text() == "L_0:\n"


def test_directive_to_text() -> None:
  """Test directive serialization."""
  dir1 = SassDirective(name="text")
  assert dir1.to_text() == ".text"

  dir2 = SassDirective(name="headerflags", params=["0x1", "0x2"])
  assert dir2.to_text() == ".headerflags 0x1, 0x2"


def test_comment_to_text() -> None:
  """Test comment serialization."""
  com = SassComment(text="this is a comment", trailing_trivia=[Trivia("\n")])
  assert com.to_text() == "// this is a comment\n"


def test_module_to_text() -> None:
  """Test module serialization."""
  mod = SassModule(
    statements=[
      SassDirective(name="text", trailing_trivia=[Trivia("\n")]),
      SassLabel(name="L_start", trailing_trivia=[Trivia("\n")]),
      SassInstruction(
        opcode="MOV",
        operands=[
          SassRegister(name="R0", leading_trivia=[Trivia(" ")]),
          SassImmediate(value=0, leading_trivia=[Trivia(", ")]),
        ],
        trailing_trivia=[Trivia(";\n")],
      ),
    ]
  )

  expected = ".text\nL_start:\nMOV R0, 0;\n"
  assert mod.to_text() == expected
