"""Test suite for the Parser module."""

import typing

from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaDirective,
  RdnaImmediate,
  RdnaInstruction,
  RdnaLabel,
  RdnaLabelRef,
  RdnaMemory,
  RdnaModifier,
  RdnaNode,
  RdnaSGPR,
  RdnaVGPR,
)
from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser


def test_parse_comment() -> None:
  """Parses comment."""
  parser = RdnaParser("; hello world")
  nodes: list[RdnaNode] = parser.parse().statements
  assert len(nodes) == 1
  assert isinstance(nodes[0], RdnaComment)
  assert nodes[0].text == " hello world"


def test_parse_label() -> None:
  """Parses label."""
  parser = RdnaParser("loop:")
  nodes: list[RdnaNode] = parser.parse().statements
  assert len(nodes) == 1
  assert isinstance(nodes[0], RdnaLabel)
  assert nodes[0].name == "loop"


def test_parse_directive() -> None:
  """Parses directive."""
  parser = RdnaParser(".global_base 1, 2")
  nodes: list[RdnaNode] = parser.parse().statements
  assert len(nodes) == 1
  assert isinstance(nodes[0], RdnaDirective)
  assert nodes[0].name == "global_base"
  assert nodes[0].params == ["1", "2"]


def test_parse_instruction_simple() -> None:
  """Parses instruction simple."""
  parser = RdnaParser("v_nop")
  nodes: list[RdnaNode] = parser.parse().statements
  assert len(nodes) == 1
  assert isinstance(nodes[0], RdnaInstruction)
  assert nodes[0].opcode == "v_nop"
  assert len(nodes[0].operands) == 0


def test_parse_instruction_operands() -> None:
  """Parses instruction operands."""
  parser = RdnaParser("v_add_f32 v0, v1, s0, 42, 0xff, my_label")
  nodes: list[RdnaNode] = parser.parse().statements
  assert len(nodes) == 1
  inst = typing.cast(RdnaInstruction, nodes[0])
  assert inst.opcode == "v_add_f32"
  assert len(inst.operands) == 6
  assert isinstance(inst.operands[0], RdnaVGPR)
  assert inst.operands[0].index == 0
  assert isinstance(inst.operands[1], RdnaVGPR)
  assert inst.operands[1].index == 1
  assert isinstance(inst.operands[2], RdnaSGPR)
  assert inst.operands[2].index == 0
  assert isinstance(inst.operands[3], RdnaImmediate)
  assert inst.operands[3].value == 42
  assert isinstance(inst.operands[4], RdnaImmediate)
  assert inst.operands[4].value == 255
  assert inst.operands[4].is_hex is True
  assert isinstance(inst.operands[5], RdnaLabelRef)
  assert inst.operands[5].name == "my_label"


def test_parse_modifiers() -> None:
  """Parses modifiers."""
  parser = RdnaParser("v_add_f32 v0, glc")
  nodes: list[RdnaNode] = parser.parse().statements
  inst = typing.cast(RdnaInstruction, nodes[0])
  assert len(inst.operands) == 2
  assert isinstance(inst.operands[1], RdnaModifier)
  assert inst.operands[1].name == "glc"


def test_parse_memory() -> None:
  """Parses memory."""
  parser = RdnaParser("v_add [v0 + 4]")
  nodes: list[RdnaNode] = parser.parse().statements
  inst = typing.cast(RdnaInstruction, nodes[0])
  assert isinstance(inst.operands[0], RdnaMemory)
  assert isinstance(inst.operands[0].base, RdnaVGPR)
  assert inst.operands[0].offset == 4
  parser2 = RdnaParser("v_add [v1 - 0x2]")
  nodes2: list[RdnaNode] = parser2.parse().statements
  inst2 = typing.cast(RdnaInstruction, nodes2[0])
  assert getattr(inst2.operands[0], "offset", None) == -2


def test_parse_memory_no_offset() -> None:
  """Parses memory no offset."""
  parser = RdnaParser("v_add [v0]")
  nodes: list[RdnaNode] = parser.parse().statements
  inst = typing.cast(RdnaInstruction, nodes[0])
  assert isinstance(inst.operands[0], RdnaMemory)
  assert inst.operands[0].offset == 0


def test_parse_register_range() -> None:
  """Parses register range."""
  parser = RdnaParser("s_mov_b64 s[0:1], v[10:11]")
  nodes: list[RdnaNode] = parser.parse().statements
  inst = typing.cast(RdnaInstruction, nodes[0])
  assert isinstance(inst.operands[0], RdnaSGPR)
  assert inst.operands[0].index == 0
  assert inst.operands[0].count == 2
  assert isinstance(inst.operands[1], RdnaVGPR)
  assert inst.operands[1].index == 10
  assert inst.operands[1].count == 2


def test_parse_special_reg() -> None:
  """Parses special reg."""
  parser = RdnaParser("s_mov_b32 exec, 1")
  nodes: list[RdnaNode] = parser.parse().statements
  inst = typing.cast(RdnaInstruction, nodes[0])
  assert isinstance(inst.operands[0], RdnaLabelRef)
  assert inst.operands[0].name == "exec"


def test_parse_directive_multiline() -> None:
  """Parses directive multiline."""
  parser = RdnaParser(".global_base 1 \n .global_base 2")
  nodes: list[RdnaNode] = parser.parse().statements
  assert len(nodes) == 2


def test_parse_instruction_multiline() -> None:
  """Parses instruction multiline."""
  parser = RdnaParser("v_add \n v_sub")
  nodes: list[RdnaNode] = parser.parse().statements
  assert len(nodes) == 2
