"""
Tests for RDNA Lexer and Parser.

Verifies:
1. Tokenization of instruction streams.
2. Parsing of register ranges (s[0:3]).
3. Parsing of explicit modifiers (glc, slc).
4. Extraction of Basic Blocks markers (Labels).
5. Comment preservation.
"""

import pytest
from ml_switcheroo.core.rdna.tokens import RdnaLexer, TokenType
from ml_switcheroo.core.rdna.parser import RdnaParser
from ml_switcheroo.core.rdna.nodes import (
  Instruction,
  Label,
  Directive,
  Comment,
  SGPR,
  VGPR,
  Immediate,
  Modifier,
  LabelRef,
)

# --- Lexer Tests ---


def test_lexer_simple_instruction() -> None:
  code = "v_add_f32 v0, v1, v2"
  lexer = RdnaLexer()
  tokens = list(lexer.tokenize(code))

  assert len(tokens) == 6  # ID, REG, COMMA, REG, COMMA, REG
  assert tokens[0].kind == TokenType.IDENTIFIER
  assert tokens[0].value == "v_add_f32"
  assert tokens[1].kind == TokenType.VGPR
  assert tokens[1].value == "v0"


def test_lexer_modifiers() -> None:
  """Verify modifiers are tokenized correctly."""
  code = "global_load_dword v1, v2, off glc"
  lexer = RdnaLexer()
  tokens = list(lexer.tokenize(code))

  # ID, VGPR, COMMA, VGPR, COMMA, MOD, MOD
  assert tokens[0].value == "global_load_dword"
  assert tokens[5].kind == TokenType.MODIFIER
  assert tokens[5].value == "off"
  assert tokens[6].kind == TokenType.MODIFIER
  assert tokens[6].value == "glc"


def test_lexer_range_syntax() -> None:
  """Verify structurizer tokens for ranges s[0:3]."""
  code = "s[0:3]"
  lexer = RdnaLexer()
  tokens = list(lexer.tokenize(code))

  # s (ID), [, 0, :, 3, ]
  assert tokens[0].kind == TokenType.IDENTIFIER
  assert tokens[0].value == "s"
  assert tokens[1].kind == TokenType.LBRACKET
  assert tokens[2].kind == TokenType.IMMEDIATE
  assert tokens[3].kind == TokenType.COLON
  assert tokens[5].kind == TokenType.RBRACKET


def test_lexer_comment() -> None:
  code = "s_mov_b32 s0, 1 ; set s0"
  lexer = RdnaLexer()
  tokens = list(lexer.tokenize(code))

  assert tokens[-1].kind == TokenType.COMMENT
  assert tokens[-1].value == "; set s0"


def test_lexer_immediate_hex() -> None:
  code = "0xFF"
  lexer = RdnaLexer()
  tokens = list(lexer.tokenize(code))
  assert tokens[0].kind == TokenType.IMMEDIATE
  assert tokens[0].value == "0xFF"


# --- Parser Tests ---


def test_parser_basic_instruction() -> None:
  """Verify simple instruction parsing."""
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
  """Verify parsing of s[4:7] syntax."""
  code = "s_load_dwordx4 s[4:7], s[0:1], 0x10"
  parser = RdnaParser(code)
  nodes = parser.parse()

  inst = nodes[0]
  assert isinstance(inst, Instruction)
  op0 = inst.operands[0]
  assert isinstance(op0, SGPR)
  assert op0.index == 4
  assert op0.count == 4  # 4,5,6,7 spans 4 indices

  op1 = inst.operands[1]
  assert isinstance(op1, SGPR)
  assert op1.index == 0
  assert op1.count == 2


def test_parser_modifiers() -> None:
  """Verify parsing of mixed operands including modifiers."""
  code = "buffer_load_dword v0, v1, s[0:3], 0 offen glc"
  parser = RdnaParser(code)
  nodes = parser.parse()

  inst = nodes[0]
  # Operands: v0, v1, s[0:3], 0, offen, glc
  assert len(inst.operands) == 6
  assert isinstance(inst.operands[-2], LabelRef)  # offen is Identifier/Ref not keyword in lexer
  # Wait, 'offen' isn't in MODIFIER list in lexical spec logic (glc, slc, dlc, off).
  # So it parses as identifier (LabelRef in node context).
  assert str(inst.operands[-2]) == "offen"

  # 'glc' is in MODIFIER list
  mod = inst.operands[-1]
  assert isinstance(mod, Modifier)
  assert mod.name == "glc"


def test_parser_labels_and_structure() -> None:
  """Verify labels and comments."""
  code = """; Start
L_ENTRY:
    s_endpgm"""
  parser = RdnaParser(code)
  nodes = parser.parse()

  assert isinstance(nodes[0], Comment)
  assert nodes[0].text == "Start"

  assert isinstance(nodes[1], Label)
  assert nodes[1].name == "L_ENTRY"

  assert isinstance(nodes[2], Instruction)
  assert nodes[2].opcode == "s_endpgm"


def test_parser_directives() -> None:
  """Verify directive parsing."""
  code = ".text\n.globl func"
  parser = RdnaParser(code)
  nodes = parser.parse()

  assert isinstance(nodes[0], Directive)
  assert nodes[0].name == "text"

  assert isinstance(nodes[1], Directive)
  assert nodes[1].name == "globl"
  assert nodes[1].params == ["func"]


def test_parser_unexpected_token() -> None:
  """Verify failure on bad syntax."""
  # Use valid tokens in invalid order (Opcode followed by Comma)
  # v_add_f32 (Identifier), , (Comma)
  code = "v_add_f32 , "
  parser = RdnaParser(code)
  with pytest.raises(SyntaxError):
    parser.parse()
