"""Test suite for nvidia sass frontend parser."""

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
  NvidiaSassRegister,
)
from ml_switcheroo.core.compiler.frontends.nvidia_sass.parser import NvidiaSassLexer, NvidiaSassParser


def test_empty() -> None:
  """Verify `NvidiaSassParser` returns `0` for empty."""
  parser: NvidiaSassParser = NvidiaSassParser("")
  mod: NvidiaSassModule = parser.parse()
  assert isinstance(mod, NvidiaSassModule)
  assert len(mod.statements) == 0

  parser2: NvidiaSassParser = NvidiaSassParser("   ")
  mod2: NvidiaSassModule = parser2.parse()
  assert isinstance(mod2, NvidiaSassModule)


def test_comment() -> None:
  """Verify `NvidiaSassParser` returns `1` for comment."""
  parser: NvidiaSassParser = NvidiaSassParser("// hello\n")
  mod: NvidiaSassModule = parser.parse()
  assert len(mod.statements) == 1
  assert isinstance(mod.statements[0], NvidiaSassComment)
  assert mod.statements[0].text == "hello"


def test_directive() -> None:
  """Verify `NvidiaSassParser` returns `2` for directive."""
  parser: NvidiaSassParser = NvidiaSassParser(".headerflags\n.reqntid 128, 1, 1\n")
  mod: NvidiaSassModule = parser.parse()
  assert len(mod.statements) == 2
  assert isinstance(mod.statements[0], NvidiaSassDirective)
  assert mod.statements[0].name == "headerflags"
  assert mod.statements[0].params == []

  assert isinstance(mod.statements[1], NvidiaSassDirective)
  assert mod.statements[1].name == "reqntid"
  assert mod.statements[1].params == ["128", "1", "1"]


def test_label() -> None:
  """Verify `NvidiaSassParser` returns `1` for label."""
  parser: NvidiaSassParser = NvidiaSassParser("L0:\n")
  mod: NvidiaSassModule = parser.parse()
  assert len(mod.statements) == 1
  assert isinstance(mod.statements[0], NvidiaSassLabel)
  assert mod.statements[0].name == "L0"


def test_instruction_basic() -> None:
  """Verify `NvidiaSassParser` returns `1` for instruction basic."""
  parser: NvidiaSassParser = NvidiaSassParser("MOV R0, 1;\n")
  mod: NvidiaSassModule = parser.parse()
  assert len(mod.statements) == 1
  inst = mod.statements[0]
  assert isinstance(inst, NvidiaSassInstruction)
  assert inst.opcode == "MOV"
  assert len(inst.operands) == 2
  assert isinstance(inst.operands[0], NvidiaSassRegister)
  assert inst.operands[0].name == "R0"
  assert isinstance(inst.operands[1], NvidiaSassImmediate)
  assert inst.operands[1].value == 1


def test_instruction_memory() -> None:
  """Verify `NvidiaSassParser` returns `3` for instruction memory."""
  parser: NvidiaSassParser = NvidiaSassParser("LDG.E R0, [R2];\nLDG.E R0, [R2 + 0x10];\nLDG.E R0, [R2 - 16];\n")
  mod: NvidiaSassModule = parser.parse()
  assert len(mod.statements) == 3

  inst1: NvidiaSassNode = mod.statements[0]
  assert isinstance(inst1, NvidiaSassInstruction)
  mem1 = inst1.operands[1]
  assert isinstance(mem1, NvidiaSassMemory)
  assert isinstance(mem1.base, NvidiaSassRegister)
  assert mem1.base.name == "R2"
  assert mem1.offset is None

  inst2: NvidiaSassNode = mod.statements[1]
  assert isinstance(inst2, NvidiaSassInstruction)
  mem2 = inst2.operands[1]
  assert isinstance(mem2, NvidiaSassMemory)
  assert mem2.offset == 16

  inst3: NvidiaSassNode = mod.statements[2]
  assert isinstance(inst3, NvidiaSassInstruction)
  mem3 = inst3.operands[1]
  assert isinstance(mem3, NvidiaSassMemory)
  assert mem3.offset == -16


def test_instruction_memory_bank() -> None:
  """Verify `NvidiaSassParser` returns `4` for instruction memory bank."""
  parser: NvidiaSassParser = NvidiaSassParser("MOV R0, c[0x0][0x4];\nMOV R0, c[0x0];\n")
  mod: NvidiaSassModule = parser.parse()

  inst1: NvidiaSassNode = mod.statements[0]
  assert isinstance(inst1, NvidiaSassInstruction)
  mem1 = inst1.operands[1]
  assert isinstance(mem1, NvidiaSassMemory)
  assert mem1.base == "c[0x0]"
  assert mem1.offset == 4

  inst2: NvidiaSassNode = mod.statements[1]
  assert isinstance(inst2, NvidiaSassInstruction)
  mem2 = inst2.operands[1]
  assert isinstance(mem2, NvidiaSassMemory)
  assert mem2.base == "c[0x0]"
  assert mem2.offset is None


def test_predicate() -> None:
  """Verify `NvidiaSassParser` handles predicate as expected."""
  parser: NvidiaSassParser = NvidiaSassParser("@P0 MOV R0, 1;\n@!P1 MOV R1, 2;\n")
  mod: NvidiaSassModule = parser.parse()

  inst1 = mod.statements[0]
  assert isinstance(inst1, NvidiaSassInstruction)
  assert inst1.predicate is not None
  assert inst1.predicate.name == "P0"
  assert not inst1.predicate.negated
  assert inst1.predicate.is_guard

  inst2 = mod.statements[1]
  assert isinstance(inst2, NvidiaSassInstruction)
  assert inst2.predicate is not None
  assert inst2.predicate.name == "P1"
  assert inst2.predicate.negated
  assert inst2.predicate.is_guard


def test_predicate_operands() -> None:
  """Verify `NvidiaSassParser` handles predicate operands as expected."""
  # predicate_operand: @!id | @!reg | @id | @reg | !id | !reg
  parser: NvidiaSassParser = NvidiaSassParser("ISETP.LT.AND P0, PT, R1, 3, PT;\n")
  mod: NvidiaSassModule = parser.parse()
  inst0 = mod.statements[0]
  assert isinstance(inst0, NvidiaSassInstruction)
  assert inst0.opcode == "ISETP.LT.AND"


def test_registers() -> None:
  """Verify `NvidiaSassParser` handles registers as expected."""
  parser: NvidiaSassParser = NvidiaSassParser("MOV -R1, |-R2|;\n")
  mod: NvidiaSassModule = parser.parse()
  inst0 = mod.statements[0]
  assert isinstance(inst0, NvidiaSassInstruction)
  op1 = inst0.operands[0]
  op2 = inst0.operands[1]
  assert isinstance(op1, NvidiaSassRegister)
  assert isinstance(op2, NvidiaSassRegister)
  assert op1.name == "R1"
  assert op1.negated
  assert op2.name == "R2"
  assert op2.absolute
  assert op2.negated


def test_empty_statement() -> None:
  """Verify `NvidiaSassParser` returns `0` for empty statement."""
  parser: NvidiaSassParser = NvidiaSassParser(";\n")
  mod: NvidiaSassModule = parser.parse()
  assert len(mod.statements) == 0


def test_mismatch() -> None:
  """Verify `NvidiaSassParser` raises `ValueError` during mismatch."""
  parser: NvidiaSassParser = NvidiaSassParser("~")
  with pytest.raises(ValueError, match="Unexpected"):
    parser.parse()


def test_predicate_operands_all() -> None:
  """Verify `NvidiaSassParser` returns `6` for predicate operands all."""
  code: str = """
    MOV R0, @!P0;
    MOV R0, @!R1;
    MOV R0, @P2;
    MOV R0, @R3;
    MOV R0, !P4;
    MOV R0, !R5;
    """
  parser: NvidiaSassParser = NvidiaSassParser(code)
  mod = parser.parse()
  assert len(mod.statements) == 6


def test_immediate_hex_and_float() -> None:
  """Verify `NvidiaSassParser` returns `16` for immediate hex and float."""
  parser: NvidiaSassParser = NvidiaSassParser("MOV R0, 0x10;\nMOV R0, 1.5;\n")
  mod: NvidiaSassModule = parser.parse()
  inst0 = mod.statements[0]
  inst1 = mod.statements[1]
  assert isinstance(inst0, NvidiaSassInstruction)
  assert isinstance(inst1, NvidiaSassInstruction)
  imm0 = inst0.operands[1]
  imm1 = inst1.operands[1]
  assert isinstance(imm0, NvidiaSassImmediate)
  assert isinstance(imm1, NvidiaSassImmediate)
  assert imm0.value == 16
  assert imm0.is_hex
  assert imm1.value == 1.5
  assert not imm1.is_hex


def test_tokenize_lexer_state_variants() -> None:
  """Test tokenize with object having text attribute and arbitrary object."""

  class ObjWithText:
    """Sentinel `ObjWithText` fixture class used in nvidia SASS frontend parser tests."""

    text = "FADD R0, R1, R2;\n"

  lexer = NvidiaSassLexer(None)
  tokens1 = list(lexer.lex(ObjWithText()))
  assert len(tokens1) > 0

  class ObjWithoutText:
    """Sentinel `ObjWithoutText` fixture class used in nvidia SASS frontend parser tests."""

    def __str__(self) -> str:
      """Return string representation.

      Returns:
          Resulting `str` from str  .
      """
      return "FADD R0, R1, R2;\n"

  tokens2 = list(lexer.lex(ObjWithoutText()))
  assert len(tokens2) > 0


def test_get_trivia_with_children() -> None:
  """Test _get_trivia traversing children when leading_trivia is absent."""
  from lark import Token, Tree

  from ml_switcheroo.core.compiler.frontends.nvidia_sass.parser import _get_trivia

  tree = Tree("node", [Token("IDENTIFIER", "abc")])
  assert _get_trivia(tree) == []


def test_transformer_directive_branches_and_instruction_empty() -> None:
  """Test NvidiaSassTransformer directive parameter unwrapping and empty instruction children."""
  from lark import Token, Tree

  from ml_switcheroo.core.compiler.frontends.nvidia_sass.parser import NvidiaSassTransformer

  transformer: NvidiaSassTransformer = NvidiaSassTransformer()

  d1: NvidiaSassDirective = transformer.directive(
    [
      Token("DOT", "."),
      Token("IDENTIFIER", "dir1"),
      Tree("directive_params", [[Tree("at_string", [Token("STR", "hello")])]]),
    ]
  )
  assert d1.params == ["hello"]

  d2: NvidiaSassDirective = transformer.directive(
    [
      Token("DOT", "."),
      Token("IDENTIFIER", "dir2"),
      Tree("directive_params", [[[Token("STR", "sublist")]]]),
    ]
  )
  assert d2.params == ["sublist"]

  d3: NvidiaSassDirective = transformer.directive(
    [
      Token("DOT", "."),
      Token("IDENTIFIER", "dir3"),
      Tree("directive_params", [[12345]]),
    ]
  )
  assert d3.params == ["12345"]

  d4: NvidiaSassDirective = transformer.directive(
    [
      Token("DOT", "."),
      Token("IDENTIFIER", "dir4"),
      "not_a_list",
    ]
  )
  assert d4.params == ["not_a_list"]

  at_res = transformer.at_string(["a", "b"])
  assert at_res == ["a", "b"]

  inst: NvidiaSassInstruction = transformer.instruction([None, None])
  assert inst.opcode == ""
