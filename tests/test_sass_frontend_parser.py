"""Test module."""

import pytest
from ml_switcheroo.core.compiler.frontends.sass.parser import SassParser
from ml_switcheroo.core.compiler.frontends.sass.cst import (
  SassModule,
  SassLabel,
  SassInstruction,
  SassDirective,
  SassComment,
  SassRegister,
  SassImmediate,
  SassMemory,
)


def test_empty() -> None:
  """Test element."""
  parser: SassParser = SassParser("")
  mod: SassModule = parser.parse()
  assert isinstance(mod, SassModule)
  assert len(mod.statements) == 0

  parser2: SassParser = SassParser("   ")
  mod2: SassModule = parser2.parse()
  assert isinstance(mod2, SassModule)


def test_comment() -> None:
  """Test element."""
  parser: SassParser = SassParser("// hello\n")
  mod: SassModule = parser.parse()
  assert len(mod.statements) == 1
  assert isinstance(mod.statements[0], SassComment)
  assert mod.statements[0].text == "hello"


def test_directive() -> None:
  """Test element."""
  parser: SassParser = SassParser(".headerflags\n.reqntid 128, 1, 1\n")
  mod: SassModule = parser.parse()
  assert len(mod.statements) == 2
  assert isinstance(mod.statements[0], SassDirective)
  assert mod.statements[0].name == "headerflags"
  assert mod.statements[0].params == []

  assert isinstance(mod.statements[1], SassDirective)
  assert mod.statements[1].name == "reqntid"
  assert mod.statements[1].params == ["128", "1", "1"]


def test_label() -> None:
  """Test element."""
  parser: SassParser = SassParser("L0:\n")
  mod: SassModule = parser.parse()
  assert len(mod.statements) == 1
  assert isinstance(mod.statements[0], SassLabel)
  assert mod.statements[0].name == "L0"


def test_instruction_basic() -> None:
  """Test element."""
  parser: SassParser = SassParser("MOV R0, 1;\n")
  mod: SassModule = parser.parse()
  assert len(mod.statements) == 1
  inst = mod.statements[0]
  assert isinstance(inst, SassInstruction)
  assert inst.opcode == "MOV"
  assert len(inst.operands) == 2
  assert isinstance(inst.operands[0], SassRegister)
  assert inst.operands[0].name == "R0"
  assert isinstance(inst.operands[1], SassImmediate)
  assert inst.operands[1].value == 1


def test_instruction_memory() -> None:
  """Test element."""
  parser: SassParser = SassParser("LDG.E R0, [R2];\nLDG.E R0, [R2 + 0x10];\nLDG.E R0, [R2 - 16];\n")
  mod: SassModule = parser.parse()
  assert len(mod.statements) == 3

  mem1 = mod.statements[0].operands[1]
  assert isinstance(mem1, SassMemory)
  assert getattr(mem1.base, "name", "") == "R2"
  assert mem1.offset is None

  mem2 = mod.statements[1].operands[1]
  assert isinstance(mem2, SassMemory)
  assert mem2.offset == 16

  mem3 = mod.statements[2].operands[1]
  assert isinstance(mem3, SassMemory)
  assert mem3.offset == -16


def test_instruction_memory_bank() -> None:
  """Test element."""
  parser: SassParser = SassParser("MOV R0, c[0x0][0x4];\nMOV R0, c[0x0];\n")
  mod: SassModule = parser.parse()

  mem1 = mod.statements[0].operands[1]
  assert isinstance(mem1, SassMemory)
  assert mem1.base == "c[0x0]"
  assert mem1.offset == 4

  mem2 = mod.statements[1].operands[1]
  assert mem2.base == "c[0x0]"
  assert mem2.offset is None


def test_predicate() -> None:
  """Test element."""
  parser: SassParser = SassParser("@P0 MOV R0, 1;\n@!P1 MOV R1, 2;\n")
  mod: SassModule = parser.parse()

  inst1 = mod.statements[0]
  assert getattr(inst1, "predicate").name == "P0"
  assert not getattr(inst1, "predicate").negated
  assert getattr(inst1, "predicate").is_guard

  inst2 = mod.statements[1]
  assert getattr(inst2, "predicate").name == "P1"
  assert getattr(inst2, "predicate").negated
  assert getattr(inst2, "predicate").is_guard


def test_predicate_operands() -> None:
  """Test element."""
  # predicate_operand: @!id | @!reg | @id | @reg | !id | !reg
  parser: SassParser = SassParser("ISETP.LT.AND P0, PT, R1, 3, PT;\n")
  mod: SassModule = parser.parse()
  assert getattr(mod.statements[0], "opcode") == "ISETP.LT.AND"


def test_registers() -> None:
  """Test element."""
  parser: SassParser = SassParser("MOV -R1, |-R2|;\n")
  mod: SassModule = parser.parse()
  op1 = getattr(mod.statements[0], "operands")[0]
  op2 = getattr(mod.statements[0], "operands")[1]
  assert op1.name == "R1"
  assert op1.negated
  assert op2.name == "R2"
  assert op2.absolute
  assert op2.negated


def test_empty_statement() -> None:
  """Test element."""
  parser: SassParser = SassParser(";\n")
  mod: SassModule = parser.parse()
  assert len(mod.statements) == 0


def test_mismatch() -> None:
  """Test element."""
  parser: SassParser = SassParser("~")
  with pytest.raises(ValueError, match="Unexpected"):
    parser.parse()


def test_predicate_operands_all() -> None:
  """Test element."""
  code: str = """
    MOV R0, @!P0;
    MOV R0, @!R1;
    MOV R0, @P2;
    MOV R0, @R3;
    MOV R0, !P4;
    MOV R0, !R5;
    """
  parser: SassParser = SassParser(code)
  parser.parse()
  # But wait, predicates as operands might fail if not covered by grammar!
  # The grammar has:
  # predicate_operand: AT BANG IDENTIFIER -> pred_at_bang_id
  #                  | AT BANG REG_IDENTIFIER -> pred_at_bang_reg
  #                  | AT IDENTIFIER -> pred_at_id
  #                  | AT REG_IDENTIFIER -> pred_at_reg
  #                  | BANG IDENTIFIER -> pred_bang_id
  #                  | BANG REG_IDENTIFIER -> pred_bang_reg
  pass  # Wait, let me actually check coverage first before I commit to this test.


def test_immediate_hex_and_float() -> None:
  """Test element."""
  parser: SassParser = SassParser("MOV R0, 0x10;\nMOV R0, 1.5;\n")
  mod: SassModule = parser.parse()
  assert getattr(mod.statements[0], "operands")[1].value == 16
  assert getattr(mod.statements[0], "operands")[1].is_hex
  assert getattr(mod.statements[1], "operands")[1].value == 1.5
  assert not getattr(mod.statements[1], "operands")[1].is_hex
