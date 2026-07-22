"""Test suite for the Parser module."""

import pytest
from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser
from ml_switcheroo.core.compiler.frontends.rdna.nodes import (
  Comment,
  Label,
  Directive,
  Instruction,
  SGPR,
  VGPR,
  Immediate,
  LabelRef,
  Modifier,
  Memory,
)


def test_parse_comment():
  """Parses comment."""
  parser = RdnaParser("; hello world")
  nodes = parser.parse()
  assert len(nodes) == 1
  assert isinstance(nodes[0], Comment)
  assert nodes[0].text == "hello world"


def test_parse_label():
  """Parses label."""
  parser = RdnaParser("loop:")
  nodes = parser.parse()
  assert len(nodes) == 1
  assert isinstance(nodes[0], Label)
  assert nodes[0].name == "loop"


def test_parse_directive():
  """Parses directive."""
  parser = RdnaParser(".global_base 1, 2")
  nodes = parser.parse()
  assert len(nodes) == 1
  assert isinstance(nodes[0], Directive)
  assert nodes[0].name == "global_base"
  assert nodes[0].params == ["1", "2"]


def test_parse_instruction_simple():
  """Parses instruction simple."""
  parser = RdnaParser("v_nop")
  nodes = parser.parse()
  assert len(nodes) == 1
  assert isinstance(nodes[0], Instruction)
  assert nodes[0].opcode == "v_nop"
  assert len(nodes[0].operands) == 0


def test_parse_instruction_operands():
  """Parses instruction operands."""
  parser = RdnaParser("v_add_f32 v0, v1, s0, 42, 0xff, my_label")
  nodes = parser.parse()
  assert len(nodes) == 1
  inst = nodes[0]
  assert inst.opcode == "v_add_f32"
  assert len(inst.operands) == 6
  assert isinstance(inst.operands[0], VGPR)
  assert inst.operands[0].index == 0
  assert isinstance(inst.operands[1], VGPR)
  assert inst.operands[1].index == 1
  assert isinstance(inst.operands[2], SGPR)
  assert inst.operands[2].index == 0
  assert isinstance(inst.operands[3], Immediate)
  assert inst.operands[3].value == 42
  assert isinstance(inst.operands[4], Immediate)
  assert inst.operands[4].value == 255
  assert inst.operands[4].is_hex is True
  assert isinstance(inst.operands[5], LabelRef)
  assert inst.operands[5].name == "my_label"


def test_parse_modifiers():
  """Parses modifiers."""
  parser = RdnaParser("v_add_f32 v0, glc")
  nodes = parser.parse()
  assert len(nodes[0].operands) == 2
  assert isinstance(nodes[0].operands[1], Modifier)
  assert nodes[0].operands[1].name == "glc"


def test_parse_memory():
  """Parses memory."""
  parser = RdnaParser("v_add [v0 + 4]")
  nodes = parser.parse()
  assert isinstance(nodes[0].operands[0], Memory)
  assert isinstance(nodes[0].operands[0].base, VGPR)
  assert nodes[0].operands[0].offset == 4
  parser2 = RdnaParser("v_add [v1 - 0x2]")
  nodes2 = parser2.parse()
  assert nodes2[0].operands[0].offset == -2


def test_parse_memory_no_offset():
  """Parses memory no offset."""
  parser = RdnaParser("v_add [v0]")
  nodes = parser.parse()
  assert isinstance(nodes[0].operands[0], Memory)
  assert nodes[0].operands[0].offset == 0


def test_parse_register_range():
  """Parses register range."""
  parser = RdnaParser("s_mov_b64 s[0:1], v[10:11]")
  nodes = parser.parse()
  inst = nodes[0]
  assert isinstance(inst.operands[0], SGPR)
  assert inst.operands[0].index == 0
  assert inst.operands[0].count == 2
  assert isinstance(inst.operands[1], VGPR)
  assert inst.operands[1].index == 10
  assert inst.operands[1].count == 2


def test_parse_special_reg():
  """Parses special reg."""
  parser = RdnaParser("s_mov_b32 exec, 1")
  nodes = parser.parse()
  assert isinstance(nodes[0].operands[0], LabelRef)
  assert nodes[0].operands[0].name == "exec"


def test_parse_unexpected_eof():
  """Parses unexpected eof."""
  parser = RdnaParser("v_add [v0 +")
  with pytest.raises(SyntaxError):
    parser.parse()


def test_parse_unexpected_token():
  """Parses unexpected token."""
  parser = RdnaParser(",")
  with pytest.raises(SyntaxError):
    parser.parse()


def test_parse_missing_bracket():
  """Parses missing bracket."""
  parser = RdnaParser("v_add [v0 + 4 foo")
  with pytest.raises(SyntaxError, match="Expected ]"):
    parser.parse()


def test_parse_immediate_float():
  """Parses immediate float."""
  parser = RdnaParser("v_add 1.5")
  nodes = parser.parse()
  assert nodes[0].operands[0].value == 1.5


def test_parse_memory_bad_imm():
  """Parses memory bad imm."""
  parser = RdnaParser("v_add [v0 + v1]")
  with pytest.raises(SyntaxError, match="Expected immediate after"):
    parser.parse()


def test_parse_bad_operand():
  """Parses bad operand."""
  parser = RdnaParser("v_add +")
  with pytest.raises(SyntaxError):
    parser.parse()


def test_consume_unexpected():
  """Verifies the behavior of consume unexpected."""
  from ml_switcheroo.core.compiler.frontends.rdna.tokens import TokenType

  parser = RdnaParser("v_add")
  with pytest.raises(SyntaxError, match="Expected"):
    parser._consume(kind=TokenType.PLUS)


def test_parse_directive_multiline():
  """Parses directive multiline."""
  parser = RdnaParser(".global_base 1 \n .global_base 2")
  nodes = parser.parse()
  assert len(nodes) == 2


def test_parse_instruction_multiline():
  """Parses instruction multiline."""
  parser = RdnaParser("v_add \n v_sub")
  nodes = parser.parse()
  assert len(nodes) == 2
