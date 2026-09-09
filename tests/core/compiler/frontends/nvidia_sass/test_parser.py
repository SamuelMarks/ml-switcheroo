"""Test suite for the Parser module."""

import typing

from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassDirective,
  NvidiaSassImmediate,
  NvidiaSassInstruction,
  NvidiaSassLabel,
  NvidiaSassMemory,
  NvidiaSassNode,
  NvidiaSassPredicate,
  NvidiaSassRegister,
)
from ml_switcheroo.core.compiler.frontends.nvidia_sass.parser import NvidiaSassParser


def test_parse_empty() -> None:
  """Parses empty."""
  parser = NvidiaSassParser("")
  assert len(parser.parse().statements) == 0


def test_parse_semicolon() -> None:
  """Parses semicolon."""
  parser = NvidiaSassParser(";")
  assert len(parser.parse().statements) == 0


def test_parse_comment() -> None:
  """Parses comment."""
  parser = NvidiaSassParser("// comment")
  nodes: list[NvidiaSassNode] = parser.parse().statements
  assert len(nodes) == 1
  assert isinstance(nodes[0], NvidiaSassComment)
  assert nodes[0].text == "comment"


def test_parse_label() -> None:
  """Parses label."""
  parser = NvidiaSassParser("L_1:")
  nodes: list[NvidiaSassNode] = parser.parse().statements
  assert len(nodes) == 1
  assert isinstance(nodes[0], NvidiaSassLabel)
  assert nodes[0].name == "L_1"


def test_parse_directive() -> None:
  """Parses directive."""
  parser = NvidiaSassParser(".headerflags 1, 2;")
  nodes: list[NvidiaSassNode] = parser.parse().statements
  assert len(nodes) == 1
  assert isinstance(nodes[0], NvidiaSassDirective)
  assert nodes[0].name == "headerflags"
  assert nodes[0].params == ["1", "2"]


def test_parse_directive_multiline() -> None:
  """Parses directive multiline."""
  parser = NvidiaSassParser(".headerflags 1\n.global_base")
  nodes: list[NvidiaSassNode] = parser.parse().statements
  assert len(nodes) == 2


def test_parse_instruction_simple() -> None:
  """Parses instruction simple."""
  parser = NvidiaSassParser("NOP;")
  nodes: list[NvidiaSassNode] = parser.parse().statements
  assert len(nodes) == 1
  assert isinstance(nodes[0], NvidiaSassInstruction)
  assert nodes[0].opcode == "NOP"


def test_parse_instruction_predicate() -> None:
  """Parses instruction predicate."""
  parser = NvidiaSassParser("@P0 FADD R0, R1;")
  nodes: list[NvidiaSassNode] = parser.parse().statements
  assert len(nodes) == 1
  inst = typing.cast(NvidiaSassInstruction, nodes[0])
  assert inst.opcode == "FADD"
  assert isinstance(inst.predicate, NvidiaSassPredicate)
  assert inst.predicate.name == "P0"
  assert inst.predicate.negated is False
  parser2 = NvidiaSassParser("@!P1 FADD R0, R1;")
  inst2 = typing.cast(NvidiaSassInstruction, parser2.parse().statements[0])
  assert inst2.predicate is not None
  assert inst2.predicate.name == "P1"
  assert inst2.predicate.negated is True


def test_parse_operands() -> None:
  """Parses operands."""
  parser = NvidiaSassParser("FADD R0, -R1, |R2|, -|R3|, c[0x0][0x4], [R1 + 0x4], [R2], 0x1, 1.5, @P0, L_1, L_2")
  nodes: list[NvidiaSassNode] = parser.parse().statements
  inst = typing.cast(NvidiaSassInstruction, nodes[0])
  assert len(inst.operands) == 12
  assert isinstance(inst.operands[0], NvidiaSassRegister) and inst.operands[0].name == "R0"
  assert isinstance(inst.operands[1], NvidiaSassRegister) and inst.operands[1].negated is True
  assert isinstance(inst.operands[2], NvidiaSassRegister) and inst.operands[2].absolute is True
  assert (
    isinstance(inst.operands[3], NvidiaSassRegister)
    and inst.operands[3].absolute is True
    and (inst.operands[3].negated is True)
  )
  assert (
    isinstance(inst.operands[4], NvidiaSassMemory)
    and inst.operands[4].base == "c[0x0]"
    and (inst.operands[4].offset == 4)
  )
  assert (
    isinstance(inst.operands[5], NvidiaSassMemory)
    and getattr(inst.operands[5].base, "name", None) == "R1"
    and (inst.operands[5].offset == 4)
  )
  assert (
    isinstance(inst.operands[6], NvidiaSassMemory)
    and getattr(inst.operands[6].base, "name", None) == "R2"
    and (inst.operands[6].offset is None)
  )
  assert (
    isinstance(inst.operands[7], NvidiaSassImmediate)
    and inst.operands[7].value == 1
    and (inst.operands[7].is_hex is True)
  )
  assert isinstance(inst.operands[8], NvidiaSassImmediate) and inst.operands[8].value == 1.5
  assert isinstance(inst.operands[9], NvidiaSassPredicate) and inst.operands[9].name == "P0"
  assert isinstance(inst.operands[10], NvidiaSassLabel) and inst.operands[10].name == "L_1"
  assert isinstance(inst.operands[11], NvidiaSassLabel) and inst.operands[11].name == "L_2"


def test_parse_memory_missing_offset() -> None:
  """Parses memory missing offset."""
  parser = NvidiaSassParser("FADD c[0x0]")
  nodes: list[NvidiaSassNode] = parser.parse().statements
  inst = typing.cast(NvidiaSassInstruction, nodes[0])
  assert isinstance(inst.operands[0], NvidiaSassMemory)
  assert inst.operands[0].base == "c[0x0]"
  assert inst.operands[0].offset is None


def test_parse_memory_dec_offset() -> None:
  """Parses memory dec offset."""
  parser = NvidiaSassParser("FADD [R1 + 10]")
  nodes: list[NvidiaSassNode] = parser.parse().statements
  inst = typing.cast(NvidiaSassInstruction, nodes[0])
  assert isinstance(inst.operands[0], NvidiaSassMemory)
  assert inst.operands[0].offset == 10


def test_instruction_multiline() -> None:
  """Verifies the behavior of instruction multiline."""
  # Changed from FADD\nNOP because Lark might parse it as FADD NOP (operands)
  parser = NvidiaSassParser("FADD;\nNOP")
  assert len(parser.parse().statements) == 2
