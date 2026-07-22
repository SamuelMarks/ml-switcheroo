"""Test suite for the Rdna Parser module."""

import pytest
from ml_switcheroo.core.compiler.frontends.rdna.tokens import RdnaLexer, TokenType
from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser
from ml_switcheroo.core.compiler.frontends.rdna.nodes import (
  Instruction,
  Label,
  Directive,
  Comment,
  SGPR,
  VGPR,
  Modifier,
  LabelRef,
)


def test_lexer_simple_instruction() -> None:
  """Verifies the behavior of lexer simple instruction."""
  code = "v_add_f32 v0, v1, v2"
  lexer = RdnaLexer()
  tokens = list(lexer.tokenize(code))
  assert len(tokens) == 6
  assert tokens[0].kind == TokenType.IDENTIFIER
  assert tokens[0].value == "v_add_f32"
  assert tokens[1].kind == TokenType.VGPR
  assert tokens[1].value == "v0"


def test_lexer_modifiers() -> None:
  """Verifies the behavior of lexer modifiers."""
  code = "global_load_dword v1, v2, off glc"
  lexer = RdnaLexer()
  tokens = list(lexer.tokenize(code))
  assert tokens[0].value == "global_load_dword"
  assert tokens[5].kind == TokenType.MODIFIER
  assert tokens[5].value == "off"
  assert tokens[6].kind == TokenType.MODIFIER
  assert tokens[6].value == "glc"


def test_lexer_range_syntax() -> None:
  """Verifies the behavior of lexer range syntax."""
  code = "s[0:3]"
  lexer = RdnaLexer()
  tokens = list(lexer.tokenize(code))
  assert tokens[0].kind == TokenType.IDENTIFIER
  assert tokens[0].value == "s"
  assert tokens[1].kind == TokenType.LBRACKET
  assert tokens[2].kind == TokenType.IMMEDIATE
  assert tokens[3].kind == TokenType.COLON
  assert tokens[5].kind == TokenType.RBRACKET


def test_lexer_comment() -> None:
  """Verifies the behavior of lexer comment."""
  code = "s_mov_b32 s0, 1 ; set s0"
  lexer = RdnaLexer()
  tokens = list(lexer.tokenize(code))
  assert tokens[-1].kind == TokenType.COMMENT
  assert tokens[-1].value == "; set s0"


def test_lexer_immediate_hex() -> None:
  """Verifies the behavior of lexer immediate hex."""
  code = "0xFF"
  lexer = RdnaLexer()
  tokens = list(lexer.tokenize(code))
  assert tokens[0].kind == TokenType.IMMEDIATE
  assert tokens[0].value == "0xFF"


def test_parser_basic_instruction() -> None:
  """Verifies the behavior of parser basic instruction."""
  code = "v_add_f32 v0, v1, v2"
  parser = RdnaParser(code)
  nodes = parser.parse()
  assert len(nodes) == 1
  inst = nodes[0]
  assert isinstance(inst, Instruction)
  assert inst.opcode == "v_add_f32"
  assert len(inst.operands) == 3
  assert isinstance(inst.operands[0], VGPR)
  assert inst.operands[0].index == 0


def test_parser_register_range() -> None:
  """Verifies the behavior of parser register range."""
  code = "s_load_dwordx4 s[4:7], s[0:1], 0x10"
  parser = RdnaParser(code)
  nodes = parser.parse()
  inst = nodes[0]
  assert isinstance(inst, Instruction)
  op0 = inst.operands[0]
  assert isinstance(op0, SGPR)
  assert op0.index == 4
  assert op0.count == 4
  op1 = inst.operands[1]
  assert isinstance(op1, SGPR)
  assert op1.index == 0
  assert op1.count == 2


def test_parser_modifiers() -> None:
  """Verifies the behavior of parser modifiers."""
  code = "buffer_load_dword v0, v1, s[0:3], 0 offen glc"
  parser = RdnaParser(code)
  nodes = parser.parse()
  inst = nodes[0]
  assert len(inst.operands) == 6
  assert isinstance(inst.operands[-2], LabelRef)
  assert str(inst.operands[-2]) == "offen"
  mod = inst.operands[-1]
  assert isinstance(mod, Modifier)
  assert mod.name == "glc"


def test_parser_labels_and_structure() -> None:
  """Verifies the behavior of parser labels and structure."""
  code = "; Start\nL_ENTRY:\n    s_endpgm"
  parser = RdnaParser(code)
  nodes = parser.parse()
  assert isinstance(nodes[0], Comment)
  assert nodes[0].text == "Start"
  assert isinstance(nodes[1], Label)
  assert nodes[1].name == "L_ENTRY"
  assert isinstance(nodes[2], Instruction)
  assert nodes[2].opcode == "s_endpgm"


def test_parser_directives() -> None:
  """Verifies the behavior of parser directives."""
  code = ".text\n.globl func"
  parser = RdnaParser(code)
  nodes = parser.parse()
  assert isinstance(nodes[0], Directive)
  assert nodes[0].name == "text"
  assert isinstance(nodes[1], Directive)
  assert nodes[1].name == "globl"
  assert nodes[1].params == ["func"]


def test_parser_unexpected_token() -> None:
  """Verifies the behavior of parser unexpected token."""
  code = "v_add_f32 , "
  parser = RdnaParser(code)
  with pytest.raises(SyntaxError):
    parser.parse()


def test_rdna_directive_with_comma():
  """Verifies the behavior of RDNA directive with comma."""
  parser = RdnaParser(".amdgcn_target gfx90a, param2")
  nodes = parser.parse()
  assert len(nodes) == 1
  assert isinstance(nodes[0], Directive)


def test_rdna_directive_followed_by_directive():
  """Verifies the behavior of RDNA directive followed by directive."""
  parser = RdnaParser(".amdgcn_target gfx90a .another")
  nodes = parser.parse()
  assert len(nodes) == 2


def test_rdna_instruction_with_comma():
  """Verifies the behavior of RDNA instruction with comma."""
  parser = RdnaParser("v_add_f32 v0, v1, v2")
  nodes = parser.parse()
  assert len(nodes) == 1
  assert isinstance(nodes[0], Instruction)
