"""Test suite for the Parser module."""

import typing
from ml_switcheroo.core.compiler.frontends.sass.parser import SassParser
from ml_switcheroo.core.compiler.frontends.sass.cst import (
  SassComment,
  SassLabel,
  SassDirective,
  SassInstruction,
  SassRegister,
  SassImmediate,
  SassMemory,
  SassPredicate,
  SassNode,
)


def test_parse_empty() -> None:
  """Parses empty."""
  parser = SassParser("")
  assert len(parser.parse().statements) == 0


def test_parse_semicolon() -> None:
  """Parses semicolon."""
  parser = SassParser(";")
  assert len(parser.parse().statements) == 0


def test_parse_comment() -> None:
  """Parses comment."""
  parser = SassParser("// comment")
  nodes: list[SassNode] = parser.parse().statements
  assert len(nodes) == 1
  assert isinstance(nodes[0], SassComment)
  assert nodes[0].text == "comment"


def test_parse_label() -> None:
  """Parses label."""
  parser = SassParser("L_1:")
  nodes: list[SassNode] = parser.parse().statements
  assert len(nodes) == 1
  assert isinstance(nodes[0], SassLabel)
  assert nodes[0].name == "L_1"


def test_parse_directive() -> None:
  """Parses directive."""
  parser = SassParser(".headerflags 1, 2;")
  nodes: list[SassNode] = parser.parse().statements
  assert len(nodes) == 1
  assert isinstance(nodes[0], SassDirective)
  assert nodes[0].name == "headerflags"
  assert nodes[0].params == ["1", "2"]


def test_parse_directive_multiline() -> None:
  """Parses directive multiline."""
  parser = SassParser(".headerflags 1\n.global_base")
  nodes: list[SassNode] = parser.parse().statements
  assert len(nodes) == 2


def test_parse_instruction_simple() -> None:
  """Parses instruction simple."""
  parser = SassParser("NOP;")
  nodes: list[SassNode] = parser.parse().statements
  assert len(nodes) == 1
  assert isinstance(nodes[0], SassInstruction)
  assert nodes[0].opcode == "NOP"


def test_parse_instruction_predicate() -> None:
  """Parses instruction predicate."""
  parser = SassParser("@P0 FADD R0, R1;")
  nodes: list[SassNode] = parser.parse().statements
  assert len(nodes) == 1
  inst = typing.cast(SassInstruction, nodes[0])
  assert inst.opcode == "FADD"
  assert isinstance(inst.predicate, SassPredicate)
  assert inst.predicate.name == "P0"
  assert inst.predicate.negated is False
  parser2 = SassParser("@!P1 FADD R0, R1;")
  inst2 = typing.cast(SassInstruction, parser2.parse().statements[0])
  assert inst2.predicate is not None
  assert inst2.predicate.name == "P1"
  assert inst2.predicate.negated is True


def test_parse_operands() -> None:
  """Parses operands."""
  parser = SassParser("FADD R0, -R1, |R2|, -|R3|, c[0x0][0x4], [R1 + 0x4], [R2], 0x1, 1.5, @P0, L_1, L_2")
  nodes: list[SassNode] = parser.parse().statements
  inst = typing.cast(SassInstruction, nodes[0])
  assert len(inst.operands) == 12
  assert isinstance(inst.operands[0], SassRegister) and inst.operands[0].name == "R0"
  assert isinstance(inst.operands[1], SassRegister) and inst.operands[1].negated is True
  assert isinstance(inst.operands[2], SassRegister) and inst.operands[2].absolute is True
  assert (
    isinstance(inst.operands[3], SassRegister)
    and inst.operands[3].absolute is True
    and (inst.operands[3].negated is True)
  )
  assert isinstance(inst.operands[4], SassMemory) and inst.operands[4].base == "c[0x0]" and (inst.operands[4].offset == 4)
  assert (
    isinstance(inst.operands[5], SassMemory)
    and getattr(inst.operands[5].base, "name", None) == "R1"
    and (inst.operands[5].offset == 4)
  )
  assert (
    isinstance(inst.operands[6], SassMemory)
    and getattr(inst.operands[6].base, "name", None) == "R2"
    and (inst.operands[6].offset is None)
  )
  assert isinstance(inst.operands[7], SassImmediate) and inst.operands[7].value == 1 and (inst.operands[7].is_hex is True)
  assert isinstance(inst.operands[8], SassImmediate) and inst.operands[8].value == 1.5
  assert isinstance(inst.operands[9], SassPredicate) and inst.operands[9].name == "P0"
  assert isinstance(inst.operands[10], SassLabel) and inst.operands[10].name == "L_1"
  assert isinstance(inst.operands[11], SassLabel) and inst.operands[11].name == "L_2"


def test_parse_memory_missing_offset() -> None:
  """Parses memory missing offset."""
  parser = SassParser("FADD c[0x0]")
  nodes: list[SassNode] = parser.parse().statements
  inst = typing.cast(SassInstruction, nodes[0])
  assert isinstance(inst.operands[0], SassMemory)
  assert inst.operands[0].base == "c[0x0]"
  assert inst.operands[0].offset is None


def test_parse_memory_dec_offset() -> None:
  """Parses memory dec offset."""
  parser = SassParser("FADD [R1 + 10]")
  nodes: list[SassNode] = parser.parse().statements
  inst = typing.cast(SassInstruction, nodes[0])
  assert isinstance(inst.operands[0], SassMemory)
  assert inst.operands[0].offset == 10


def test_instruction_multiline() -> None:
  """Verifies the behavior of instruction multiline."""
  # Changed from FADD\nNOP because Lark might parse it as FADD NOP (operands)
  parser = SassParser("FADD;\nNOP")
  assert len(parser.parse().statements) == 2
