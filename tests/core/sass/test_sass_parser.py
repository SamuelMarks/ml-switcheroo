"""
Tests for SASS Lexer and Parser.

Verifies:
1. Tokenization of instruction streams.
2. Parsing of complex modifiers (FADD.FTZ).
3. Extraction of Basic Blocks markers (Labels).
4. Comment preservation.
5. Operand handling (Memory, Predicates, Negated Registers).
"""

import pytest
from ml_switcheroo.core.sass.tokens import SassLexer, TokenType
from ml_switcheroo.core.sass.parser import SassParser
from ml_switcheroo.core.sass.nodes import Instruction, Label, Directive, Comment, Register, Immediate, Memory, Predicate

# --- Lexer Tests ---


def test_lexer_simple_instruction():
  code = "FADD R1, R2, R3;"
  lexer = SassLexer()
  tokens = list(lexer.tokenize(code))

  assert len(tokens) == 7  # ID, REG, COMMA, REG, COMMA, REG, SEMI
  assert tokens[0].kind == TokenType.IDENTIFIER
  assert tokens[0].value == "FADD"
  assert tokens[1].kind == TokenType.REGISTER
  assert tokens[1].value == "R1"


def test_lexer_modifiers():
  """Verify dot notation in Opcodes is preserved as one Identifier."""
  code = "FFMA.FTZ.RN R0, R1, R2, RZ;"
  lexer = SassLexer()
  tokens = list(lexer.tokenize(code))

  assert tokens[0].kind == TokenType.IDENTIFIER
  assert tokens[0].value == "FFMA.FTZ.RN"


def test_lexer_memory_operands():
  """Verify constant and global memory tokenization."""
  code = "LD R0, [R1 + 0x4]; LDC R2, c[0x0][0x140];"
  lexer = SassLexer()
  tokens = list(lexer.tokenize(code))

  # [R1 + 0x4] - 4th token (indices 0-3)
  mem_glob = tokens[3]
  assert mem_glob.kind == TokenType.MEMORY
  assert mem_glob.value == "[R1 + 0x4]"

  # c[0x0][0x140]
  # Previous: LD, R0, ,, MEM, ;, LDC, R2, , -> 8 tokens.
  # This memory token is at index 8. Semicolon is at 9.
  mem_const = tokens[8]
  assert mem_const.kind == TokenType.MEMORY
  assert mem_const.value == "c[0x0][0x140]"


def test_lexer_predicates_and_labels():
  code = "@P0 BRA L_EXIT;"
  lexer = SassLexer()
  tokens = list(lexer.tokenize(code))

  assert tokens[0].kind == TokenType.PREDICATE
  assert tokens[0].value == "@P0"

  assert tokens[2].kind == TokenType.IDENTIFIER  # L_EXIT is ref


# --- Parser Tests ---


def test_parser_basic_block_structure():
  """Verify Labels and Instructions are parsed linearizing the block."""
  code = """ 
    L_START: 
        FADD R0, R1, R2; 
        BRA L_END; 
    L_END: 
        EXIT; 
    """
  parser = SassParser(code)
  nodes = parser.parse()

  # Should be 5 nodes: Label, Inst, Inst, Label, Inst
  assert len(nodes) == 5
  assert isinstance(nodes[0], Label)
  assert nodes[0].name == "L_START"

  assert isinstance(nodes[1], Instruction)
  assert nodes[1].opcode == "FADD"

  assert isinstance(nodes[2], Instruction)
  assert nodes[2].opcode == "BRA"
  # Check Label Reference handling
  # BRA target 'L_END' should be parsed as LabelRef (operand)
  assert str(nodes[2].operands[0]) == "L_END"

  assert isinstance(nodes[3], Label)
  assert nodes[3].name == "L_END"


def test_parser_operands_complexity():
  """Verify negation, absolute value, and immediate handling."""
  code = "IADD3 R0, -R1, |R2|, 0x10;"
  parser = SassParser(code)
  nodes = parser.parse()

  inst = nodes[0]
  op1 = inst.operands[1]  # -R1
  op2 = inst.operands[2]  # |R2|
  op3 = inst.operands[3]  # 0x10

  assert isinstance(op1, Register)
  assert op1.negated is True
  assert op1.name == "R1"

  assert isinstance(op2, Register)
  assert op2.absolute is True
  assert op2.name == "R2"

  assert isinstance(op3, Immediate)
  assert op3.value == 16
  assert op3.is_hex is True


def test_parser_predicates():
  """Verify predicated execution parsing."""
  code = "@!P0 MOV R0, RZ;"
  parser = SassParser(code)
  nodes = parser.parse()

  inst = nodes[0]
  assert inst.predicate is not None
  assert inst.predicate.name == "P0"
  assert inst.predicate.negated is True
  assert inst.opcode == "MOV"


def test_parser_comments_preservation():
  """Verify comments are extracted into AST."""
  code = """ 
    // Init Loop
    MOV R0, RZ; // Clear Accumulator
    """
  parser = SassParser(code)
  nodes = parser.parse()

  assert isinstance(nodes[0], Comment)
  assert nodes[0].text == "Init Loop"

  # Inline comment usually comes AFTER the instruction in stream
  # Lexer: MOV, ..., SEMI, COMMENT
  assert isinstance(nodes[1], Instruction)
  assert isinstance(nodes[2], Comment)
  assert nodes[2].text == "Clear Accumulator"


def test_parser_directives():
  """Verify directives parsing."""
  code = ".headerflags @0x100;"
  parser = SassParser(code)
  nodes = parser.parse()

  assert isinstance(nodes[0], Directive)
  assert nodes[0].name == "headerflags"
  # Arguments might vary, checking generic consumption
  assert nodes[0].params == ["@0x100"]
